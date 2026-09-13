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

"""Training only the vision tower's last layers changes what gets gradients, not what it computes."""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (  # noqa: E402
    MultiTaskDiTConfig,
)
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import CLIPVisionEncoder  # noqa: E402

MODEL = "openai/clip-vit-base-patch16"


@pytest.fixture(scope="module")
def full():
    return CLIPVisionEncoder(MODEL)


def _trainable(encoder):
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def test_default_trains_the_whole_tower(full):
    assert MultiTaskDiTConfig().vision_encoder_trainable_layers is None
    assert _trainable(full) == sum(p.numel() for p in full.parameters())


@pytest.mark.parametrize("trainable_layers", [0, 1, 4, 12])
def test_only_the_last_layers_require_gradients(full, trainable_layers):
    encoder = CLIPVisionEncoder(MODEL, trainable_layers=trainable_layers)
    layers = encoder.model.vision_model.encoder.layers
    for position, layer in enumerate(layers):
        expected = position >= len(layers) - trainable_layers if trainable_layers else False
        assert all(p.requires_grad is expected for p in layer.parameters()), position
    assert not any(p.requires_grad for p in encoder.model.vision_model.embeddings.parameters())
    if trainable_layers == 0:
        assert _trainable(encoder) == 0
    else:
        assert 0 < _trainable(encoder) <= _trainable(full)


def test_fewer_layers_means_fewer_trainable_parameters(full):
    counts = [_trainable(CLIPVisionEncoder(MODEL, trainable_layers=n)) for n in (1, 4)]
    assert counts[0] < counts[1] < _trainable(full)


def test_the_forward_pass_is_unchanged(full):
    """Freezing changes gradients only; the same weights must give the same CLS token."""
    partial = CLIPVisionEncoder(MODEL, trainable_layers=4)
    partial.load_state_dict(full.state_dict())
    full.eval()
    partial.eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(partial(images), full(images))


def test_the_backward_stops_below_the_trainable_layers():
    encoder = CLIPVisionEncoder(MODEL, trainable_layers=2)
    encoder(torch.rand(1, 3, 224, 224)).sum().backward()
    layers = encoder.model.vision_model.encoder.layers
    assert all(p.grad is None for p in layers[0].parameters())
    assert any(p.grad is not None for p in layers[-1].parameters())
    assert all(p.grad is None for p in encoder.model.vision_model.embeddings.parameters())


def test_asking_for_more_layers_than_exist_is_rejected():
    with pytest.raises(ValueError, match="exceeds the 12 layers"):
        CLIPVisionEncoder(MODEL, trainable_layers=99)


def test_a_negative_count_is_rejected():
    with pytest.raises(ValueError, match="vision_encoder_trainable_layers"):
        MultiTaskDiTConfig(vision_encoder_trainable_layers=-1)
