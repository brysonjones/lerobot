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

"""Cameras sharing an image encoder go through it together, and get back what they would alone."""

import pytest
import torch

from lerobot.policies.utils import embed_images_batched


class RecordingEncoder:
    """A batch-independent stand-in for an image encoder, counting the batch size of each call."""

    def __init__(self):
        self.calls: list[int] = []
        torch.manual_seed(0)
        self.weight = torch.randn(3 * 8 * 8, 5)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append(images.shape[0])
        flat = images.flatten(1)
        projection = flat @ self.weight[: flat.shape[1]]
        return projection.unsqueeze(1).expand(-1, 4, -1).contiguous()


@pytest.fixture
def encoder():
    return RecordingEncoder()


def _images(count, height=8, width=8, batch_size=2):
    torch.manual_seed(1)
    return [torch.rand(batch_size, 3, height, width) for _ in range(count)]


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_output_matches_one_call_per_camera(encoder, count):
    images = _images(count)
    expected = [encoder(image) for image in images]
    encoder.calls.clear()
    actual = embed_images_batched(encoder, images)
    assert len(actual) == count
    for one, other in zip(actual, expected, strict=True):
        torch.testing.assert_close(one, other)


def test_same_resolution_cameras_take_one_call(encoder):
    embed_images_batched(encoder, _images(4))
    assert encoder.calls == [8]


def test_a_single_camera_is_not_wrapped(encoder):
    embed_images_batched(encoder, _images(1))
    assert encoder.calls == [2]


def test_no_cameras_is_no_call(encoder):
    assert embed_images_batched(encoder, []) == []
    assert encoder.calls == []


def test_a_resolution_change_starts_a_new_call(encoder):
    images = [*_images(2), *_images(1, height=4, width=16), *_images(2)]
    expected = [encoder(image) for image in images]
    encoder.calls.clear()
    actual = embed_images_batched(encoder, images)
    assert encoder.calls == [4, 2, 4]
    for one, other in zip(actual, expected, strict=True):
        torch.testing.assert_close(one, other)


def test_camera_order_is_preserved(encoder):
    """A policy lays its camera tokens out in order; a permuted result would be silently wrong."""
    images = _images(3)
    actual = embed_images_batched(encoder, images)
    for position, image in enumerate(images):
        encoder.calls.clear()
        torch.testing.assert_close(actual[position], encoder(image))


def test_gradients_flow_to_every_camera(encoder):
    images = [image.requires_grad_(True) for image in _images(3)]
    torch.stack([embedding.sum() for embedding in embed_images_batched(encoder, images)]).sum().backward()
    assert all(image.grad is not None and image.grad.abs().sum() > 0 for image in images)
