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

"""Tables that depend only on the configuration are built once, and still produce the same values."""

import math

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (  # noqa: E402
    RotaryPositionalEmbedding,
    SinusoidalPosEmb,
)


def _reference_sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    """What the forward pass used to rebuild on every call."""
    half_dim = dim // 2
    decay = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -decay)
    emb = x[:, None] * frequencies[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


@pytest.mark.parametrize("dim", [64, 256])
def test_sinusoidal_values_are_unchanged(dim):
    x = torch.rand(8) * 100
    torch.testing.assert_close(SinusoidalPosEmb(dim)(x), _reference_sinusoidal(x, dim))


def test_sinusoidal_frequencies_are_not_checkpointed():
    """They are derived from the configuration, so a checkpoint must not carry them."""
    embedding = SinusoidalPosEmb(64)
    assert "frequencies" not in embedding.state_dict()
    assert embedding.frequencies.shape == (32,)


def test_sinusoidal_follows_the_input_dtype():
    embedding = SinusoidalPosEmb(64)
    assert embedding(torch.rand(4, dtype=torch.float64)).dtype == torch.float64
    assert embedding(torch.rand(4)).dtype == torch.float32


class TestRotaryCache:
    def _tensors(self, dtype=torch.float32):
        torch.manual_seed(1)
        return torch.rand(2, 4, 16, 32, dtype=dtype), torch.rand(2, 4, 16, 32, dtype=dtype)

    def test_values_are_unchanged(self):
        rope = RotaryPositionalEmbedding(32, max_seq_len=64)
        q, k = self._tensors()
        cos = rope._cos_cached[:, :, : q.shape[2], :].to(q.dtype)
        sin = rope._sin_cached[:, :, : q.shape[2], :].to(q.dtype)
        expected_q = (q * cos) + (rope._rotate_half(q) * sin)
        rotated_q, _ = rope(q, k)
        torch.testing.assert_close(rotated_q, expected_q)

    def test_the_cast_table_follows_the_query_dtype(self):
        rope = RotaryPositionalEmbedding(32, max_seq_len=64)
        q, k = self._tensors()
        rope(q, k)
        assert rope._cos_cast.dtype == torch.float32
        rope(q.half(), k.half())
        assert rope._cos_cast.dtype == torch.float16

    def test_switching_dtypes_back_and_forth_stays_correct(self):
        """A cache keyed on dtype must not hand a later call the previous dtype's table."""
        rope = RotaryPositionalEmbedding(32, max_seq_len=64)
        q, k = self._tensors()
        first, _ = rope(q, k)
        rope(q.half(), k.half())
        again, _ = rope(q, k)
        torch.testing.assert_close(again, first)

    def test_a_shorter_sequence_takes_a_prefix(self):
        rope = RotaryPositionalEmbedding(32, max_seq_len=64)
        q, k = self._tensors()
        full, _ = rope(q, k)
        short, _ = rope(q[:, :, :8], k[:, :, :8])
        torch.testing.assert_close(short, full[:, :, :8])

    def test_too_long_a_sequence_is_rejected(self):
        rope = RotaryPositionalEmbedding(32, max_seq_len=8)
        q, k = self._tensors()
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            rope(q, k)
