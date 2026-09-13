#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`compile_model` changes how ACT's training step runs, not what it computes or checkpoints."""

import copy

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMERA = "observation.images.cam"


def _policy(compile_model, **overrides):
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=8,
        n_action_steps=8,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        device="cpu",
        compile_model=compile_model,
        **overrides,
    )
    stats = {
        CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    return ACTPolicy(config, dataset_stats=stats)


def _batch(batch_size=2):
    torch.manual_seed(1)
    return {
        CAMERA: torch.rand(batch_size, 3, 96, 96),
        "observation.state": torch.rand(batch_size, 7),
        "action": torch.rand(batch_size, 8, 7),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
    }


def test_off_by_default():
    assert ACTConfig().compile_model is False
    assert ACTConfig().compile_mode is None


def test_checkpoint_keys_are_unchanged():
    """Compiling the bound forward, not the module, is what keeps `_orig_mod` out of the keys."""
    eager, compiled = _policy(False), _policy(True)
    assert set(eager.state_dict()) == set(compiled.state_dict())
    assert not any("_orig_mod" in key for key in compiled.state_dict())
    compiled.load_state_dict(eager.state_dict())


def test_loss_and_gradients_match_eager():
    # No VAE sampling and no dropout: with randomness in the graph the two paths consume their
    # generators differently and the comparison would say nothing.
    eager = _policy(False, use_vae=False, dropout=0.0)
    compiled = _policy(True, use_vae=False, dropout=0.0)
    compiled.load_state_dict(eager.state_dict())

    eager_loss, _ = eager(copy.deepcopy(_batch()))
    compiled_loss, _ = compiled(copy.deepcopy(_batch()))
    torch.testing.assert_close(compiled_loss, eager_loss, atol=1e-5, rtol=0)

    eager_loss.backward()
    compiled_loss.backward()
    for (name, expected), (_, actual) in zip(
        eager.named_parameters(), compiled.named_parameters(), strict=True
    ):
        if expected.grad is None:
            continue
        torch.testing.assert_close(
            actual.grad, expected.grad, atol=1e-4, rtol=0, msg=lambda m, name=name: f"{name}: {m}"
        )


def test_a_second_batch_shape_is_accepted():
    """Static shapes mean one graph per shape, not a failure on the second one."""
    policy = _policy(True, use_vae=False, dropout=0.0)
    assert policy(_batch(2))[0].ndim == 0
    assert policy(_batch(3))[0].ndim == 0


def test_select_action_is_left_uncompiled():
    """Inference keeps the latency profile it had; this flag is about the training step."""
    policy = _policy(True)
    policy.eval()
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(
            {CAMERA: torch.rand(1, 3, 96, 96), "observation.state": torch.rand(1, 7)}
        )
    assert action.shape == (1, 7)
