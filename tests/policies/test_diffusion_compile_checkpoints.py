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

"""A diffusion checkpoint is the same whether or not the run that wrote it compiled the U-Net."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature

pytest.importorskip("diffusers")

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

CAMERA = "observation.images.cam"


def _policy(compile_model):
    torch.manual_seed(0)
    config = DiffusionConfig(
        input_features={
            CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        crop_shape=(84, 84),
        device="cpu",
        compile_model=compile_model,
    )
    stats = {
        CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    return DiffusionPolicy(config, dataset_stats=stats)


def test_compiling_does_not_rename_checkpoint_keys():
    """`torch.compile(module)` would put every U-Net parameter under `_orig_mod`."""
    eager, compiled = _policy(False), _policy(True)
    assert set(eager.state_dict()) == set(compiled.state_dict())
    assert not any("_orig_mod" in key for key in compiled.state_dict())


def test_checkpoints_are_interchangeable_between_the_two_settings():
    eager, compiled = _policy(False), _policy(True)
    eager.load_state_dict(compiled.state_dict())
    compiled.load_state_dict(eager.state_dict())


def test_compile_is_off_by_default():
    assert DiffusionConfig().compile_model is False
