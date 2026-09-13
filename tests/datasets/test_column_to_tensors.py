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

"""Converting a column in one call must give what converting it row by row gives."""

import numpy as np
import pytest
import torch

from lerobot.datasets.io_utils import _column_to_tensors

COLUMNS = [
    pytest.param([[0.0, 1.5], [2.0, -3.25]], id="float-rows"),
    pytest.param([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], id="nested-rows"),
    pytest.param([0.5, 1.5, 2.5], id="float-scalars"),
    pytest.param([0, 1, 2], id="int-scalars"),
    pytest.param([True, False, True], id="bool-scalars"),
    pytest.param([np.array([1.0, 2.0], dtype=np.float32)] * 3, id="numpy-rows"),
    pytest.param([np.float32(1.5), np.float32(2.5)], id="numpy-scalars"),
    pytest.param([[1.0]], id="single-row"),
]


def _row_by_row(values):
    return [x if isinstance(x, str) else torch.tensor(x) for x in values]


@pytest.mark.parametrize("values", COLUMNS)
def test_matches_row_by_row(values):
    batched = _column_to_tensors(list(values))
    expected = _row_by_row(list(values))
    assert len(batched) == len(expected)
    for got, want in zip(batched, expected, strict=True):
        assert got.dtype == want.dtype, f"{got.dtype} != {want.dtype}"
        assert got.shape == want.shape
        torch.testing.assert_close(got, want)


def test_strings_pass_through():
    values = ["pick the cube", "place it"]
    assert _column_to_tensors(list(values)) == values


def test_mixed_strings_and_numbers_fall_back():
    values = ["a", 1.0]
    out = _column_to_tensors(list(values))
    assert out[0] == "a"
    torch.testing.assert_close(out[1], torch.tensor(1.0))


def test_ragged_rows_fall_back():
    """A window is normally rectangular; if one is not, each row still converts."""
    out = _column_to_tensors([[1.0, 2.0], [3.0]])
    assert [tuple(t.shape) for t in out] == [(2,), (1,)]


def test_nulls_fail_the_same_way_as_before():
    """A null in a numeric column was never convertible; the fallback must not paper over it."""
    values = [[1.0], None]
    with pytest.raises(RuntimeError):
        _row_by_row(list(values))
    with pytest.raises(RuntimeError):
        _column_to_tensors(list(values))


def test_empty_column():
    assert _column_to_tensors([]) == []


def test_rows_are_independent_of_each_other():
    """The batched path returns views; writing to one must not reach the others."""
    out = _column_to_tensors([[1.0, 2.0], [3.0, 4.0]])
    out[0].add_(100.0)
    torch.testing.assert_close(out[1], torch.tensor([3.0, 4.0]))
