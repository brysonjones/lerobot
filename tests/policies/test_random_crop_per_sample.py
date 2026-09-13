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

"""Each image of a batch is cropped at its own position, and each crop is a real sub-window."""

import itertools

import pytest
import torch

from lerobot.policies.utils import RandomCropPerSample
from lerobot.utils.random_utils import seeded_context


def _offset_of(image, crop):
    """Find where `crop` sits inside `image`, or None if it is not a sub-window of it."""
    crop_h, crop_w = crop.shape[-2:]
    height, width = image.shape[-2:]
    for top, left in itertools.product(range(height - crop_h + 1), range(width - crop_w + 1)):
        if torch.equal(image[..., top : top + crop_h, left : left + crop_w], crop):
            return top, left
    return None


@pytest.mark.parametrize("size", [(5, 6), 5])
def test_output_shape(size):
    crop = RandomCropPerSample(size)
    assert crop(torch.rand(4, 3, 8, 9)).shape == (4, 3, *crop.size)


def test_every_crop_is_a_sub_window_of_its_own_image():
    images = torch.rand(32, 3, 8, 9)
    with seeded_context(0):
        crops = RandomCropPerSample((5, 6))(images)
    offsets = [_offset_of(images[i], crops[i]) for i in range(len(images))]
    assert all(offset is not None for offset in offsets)


def test_positions_differ_between_images():
    """The whole point: torchvision's RandomCrop would give all 64 images the same position."""
    images = torch.rand(64, 3, 16, 16)
    with seeded_context(0):
        crops = RandomCropPerSample((8, 8))(images)
    offsets = {_offset_of(images[i], crops[i]) for i in range(len(images))}
    assert len(offsets) > 1


def test_channels_share_their_image_position():
    images = torch.rand(8, 3, 12, 12)
    with seeded_context(1):
        crops = RandomCropPerSample((6, 6))(images)
    for image, crop in zip(images, crops, strict=True):
        assert _offset_of(image[0], crop[0]) == _offset_of(image[2], crop[2])


def test_full_size_crop_is_the_identity():
    images = torch.rand(4, 3, 7, 7)
    torch.testing.assert_close(RandomCropPerSample((7, 7))(images), images)


def test_crop_larger_than_the_image_is_rejected():
    with pytest.raises(ValueError, match="larger than"):
        RandomCropPerSample((9, 9))(torch.rand(2, 3, 8, 8))


def test_gradients_reach_the_cropped_region_only():
    images = torch.zeros(2, 1, 4, 4, requires_grad=True)
    with seeded_context(2):
        RandomCropPerSample((2, 2))(images).sum().backward()
    assert images.grad.sum() == 2 * 2 * 2  # one crop's worth of ones per image
