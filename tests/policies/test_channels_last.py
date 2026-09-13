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

"""`channels_last` changes a vision backbone's layout, not what it computes."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.import_utils import is_package_available

CAMERA = "observation.images.cam"


def _policy(channels_last):
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
        pretrained_backbone_weights=None,
        use_vae=False,
        dropout=0.0,
        channels_last=channels_last,
        device="cpu",
    )
    stats = {
        CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    torch.manual_seed(0)
    return ACTPolicy(config, dataset_stats=stats)


def _batch():
    torch.manual_seed(1)
    return {
        CAMERA: torch.rand(2, 3, 96, 96),
        "observation.state": torch.rand(2, 7),
        "action": torch.rand(2, 8, 7),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }


def test_defaults_to_off():
    assert ACTConfig().channels_last is False
    assert (
        not _policy(channels_last=False)
        .model.backbone["conv1"]
        .weight.is_contiguous(memory_format=torch.channels_last)
    )


def test_backbone_weights_carry_the_layout():
    backbone = _policy(channels_last=True).model.backbone
    assert backbone["conv1"].weight.is_contiguous(memory_format=torch.channels_last)
    assert backbone["layer4"][0].conv1.weight.is_contiguous(memory_format=torch.channels_last)


def test_the_rest_of_the_model_is_untouched():
    """Only convolutions benefit; reshaping the transformer's weights would be pointless."""
    policy = _policy(channels_last=True)
    assert policy.model.decoder.layers[0].linear1.weight.stride() == (
        policy.model.decoder.layers[0].linear1.weight.shape[1],
        1,
    )


def test_loss_matches_the_contiguous_layout():
    contiguous, channels_last = _policy(False), _policy(True)
    channels_last.load_state_dict(contiguous.state_dict())
    batch = _batch()
    loss_a, _ = contiguous(dict(batch))
    loss_b, _ = channels_last(dict(batch))
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-4, atol=1e-5)


def test_gradients_match_the_contiguous_layout():
    contiguous, channels_last = _policy(False), _policy(True)
    channels_last.load_state_dict(contiguous.state_dict())
    batch = _batch()
    contiguous(dict(batch))[0].backward()
    channels_last(dict(batch))[0].backward()
    pairs = dict(channels_last.named_parameters())
    checked = 0
    for name, param in contiguous.named_parameters():
        if param.grad is None:
            continue
        torch.testing.assert_close(param.grad, pairs[name].grad, rtol=1e-3, atol=1e-5, msg=name)
        checked += 1
    assert checked > 20


def test_state_dict_keys_are_unchanged():
    """A checkpoint must be interchangeable between the two layouts."""
    assert set(_policy(True).state_dict()) == set(_policy(False).state_dict())


@pytest.mark.parametrize("channels_last", [False, True])
def test_select_action_runs(channels_last):
    policy = _policy(channels_last)
    policy.reset()
    policy.eval()
    with torch.no_grad():
        action = policy.select_action(
            {CAMERA: torch.rand(1, 3, 96, 96), "observation.state": torch.rand(1, 7)}
        )
    assert action.shape == (1, 7)


class TestEveryConvBackbonePolicy:
    """The same option on diffusion and VQ-BeT, which use the same ResNet encoder."""

    def _diffusion(self, channels_last):
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        config = DiffusionConfig(
            input_features={
                CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
                "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            n_obs_steps=1,
            horizon=8,
            n_action_steps=4,
            crop_shape=None,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            channels_last=channels_last,
            device="cpu",
        )
        stats = {
            CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
            "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
            "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        }
        torch.manual_seed(0)
        return DiffusionPolicy(config, dataset_stats=stats)

    def test_diffusion_defaults_to_off(self):
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        assert DiffusionConfig().channels_last is False

    @pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
    def test_diffusion_backbone_carries_the_layout(self):
        backbone = self._diffusion(channels_last=True).diffusion.rgb_encoder.backbone
        weights = [m.weight for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
        assert weights, "expected convolutions in the backbone"
        assert all(w.is_contiguous(memory_format=torch.channels_last) for w in weights)

    @pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
    def test_diffusion_state_dict_keys_are_unchanged(self):
        assert set(self._diffusion(True).state_dict()) == set(self._diffusion(False).state_dict())

    def test_vqbet_defaults_to_off(self):
        from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

        assert VQBeTConfig().channels_last is False
