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

"""`compile_model` changes how VQ-BeT's second training phase runs, not what it checkpoints."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

CAMERA = "observation.images.cam"


def _policy(compile_model):
    torch.manual_seed(0)
    config = VQBeTConfig(
        input_features={
            CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        crop_shape=(84, 84),
        device="cpu",
        compile_model=compile_model,
    )
    stats = {
        CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    return VQBeTPolicy(config, dataset_stats=stats)


def test_off_by_default():
    assert VQBeTConfig().compile_model is False
    assert VQBeTConfig().compile_mode is None


def test_checkpoint_keys_are_unchanged():
    """Compiling the bound forward, not the module, is what keeps `_orig_mod` out of the keys."""
    eager, compiled = _policy(False), _policy(True)
    assert set(eager.state_dict()) == set(compiled.state_dict())
    assert not any("_orig_mod" in key for key in compiled.state_dict())


def test_checkpoints_are_interchangeable():
    eager, compiled = _policy(False), _policy(True)
    compiled.load_state_dict(eager.state_dict())
    eager.load_state_dict(compiled.state_dict())


def test_the_residual_vq_phase_does_not_go_through_the_compiled_call():
    """Phase one trains the residual VQ on its own path; only phase two is compiled."""
    policy = _policy(True)
    assert policy.vqbet.action_head.vqvae_model.discretized.item() is False
    assert policy._vqbet_forward is not policy.vqbet.forward
