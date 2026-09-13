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

"""Images embedded in the parquet follow the same uint8 contract as frames decoded from video."""

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from lerobot.datasets.io_utils import hf_transform_to_torch, pil_to_chw_tensor


@pytest.fixture
def rgb():
    rng = np.random.default_rng(0)
    return PILImage.fromarray(rng.integers(0, 256, (5, 7, 3), dtype=np.uint8))


def test_uint8_and_float_carry_the_same_picture(rgb):
    as_float = pil_to_chw_tensor(rgb)
    as_uint8 = pil_to_chw_tensor(rgb, return_uint8=True)
    assert as_uint8.dtype == torch.uint8
    assert as_uint8.shape == as_float.shape == (3, 5, 7)
    torch.testing.assert_close(as_uint8.float() / 255.0, as_float)


def test_uint8_is_a_quarter_of_the_bytes(rgb):
    as_float = pil_to_chw_tensor(rgb)
    as_uint8 = pil_to_chw_tensor(rgb, return_uint8=True)
    assert as_uint8.nbytes * 4 == as_float.nbytes


def test_uint8_tensor_is_contiguous_and_writable(rgb):
    """A tensor sharing PIL's read-only buffer would break pinning and in-place transforms."""
    tensor = pil_to_chw_tensor(rgb, return_uint8=True)
    assert tensor.is_contiguous()
    tensor[0, 0, 0] = 1  # must not raise


def test_grayscale_keeps_its_single_channel():
    rng = np.random.default_rng(1)
    gray = PILImage.fromarray(rng.integers(0, 256, (5, 7), dtype=np.uint8))
    assert pil_to_chw_tensor(gray, return_uint8=True).shape == (1, 5, 7)
    torch.testing.assert_close(
        pil_to_chw_tensor(gray, return_uint8=True).float() / 255.0, pil_to_chw_tensor(gray)
    )


def test_depth_maps_ignore_the_flag():
    """A uint16 depth map is in native units, not a 0-255 range, so it stays float32."""
    depth = PILImage.fromarray(np.full((5, 7), 40_000, dtype=np.uint16))
    for flag in (False, True):
        tensor = pil_to_chw_tensor(depth, return_uint8=flag)
        assert tensor.dtype == torch.float32 and tensor.shape == (1, 5, 7)
        assert tensor[0, 0, 0] == 40_000


@pytest.mark.parametrize("return_uint8", [False, True])
def test_batch_transform_passes_the_flag_through(rgb, return_uint8):
    batch = {"observation.images.cam": [rgb, rgb], "observation.state": [[1.0], [2.0]]}
    out = hf_transform_to_torch(batch, return_uint8=return_uint8)
    expected = torch.uint8 if return_uint8 else torch.float32
    assert all(frame.dtype == expected for frame in out["observation.images.cam"])
    assert out["observation.state"][0].dtype == torch.float32


def test_batch_transform_defaults_to_float(rgb):
    """Every caller that does not ask for uint8 keeps the behaviour it had."""
    out = hf_transform_to_torch({"observation.images.cam": [rgb]})
    assert out["observation.images.cam"][0].dtype == torch.float32
