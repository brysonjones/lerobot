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

"""An embedded image is decoded once per sample that needs it, not once more for the current row."""

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_CHW

CAMERA = "observation.images.cam"


@pytest.fixture
def image_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        CAMERA: {"dtype": "image", "shape": DUMMY_CHW, "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "decode", features=features)
    rng = np.random.default_rng(0)
    for _ in range(6):
        dataset.add_frame(
            {
                CAMERA: rng.integers(0, 256, DUMMY_CHW[1:] + (3,), dtype=np.uint8),
                "observation.state": torch.zeros(2),
                "action": torch.zeros(2),
                "task": "dummy",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return dataset


def _reload(dataset, delta_timestamps):
    return LeRobotDataset(
        dataset.repo_id, root=dataset.root, delta_timestamps=delta_timestamps, download_videos=False
    )


def _count_decoded_images(monkeypatch):
    """Count the embedded images turned into tensors, one call per image the row query decoded."""
    from lerobot.datasets import io_utils

    calls = []
    original = io_utils.pil_to_chw_tensor

    def counting(img, *args, **kwargs):
        calls.append(1)
        return original(img, *args, **kwargs)

    monkeypatch.setattr(io_utils, "pil_to_chw_tensor", counting)
    return calls


def test_row_query_skips_the_columns_the_delta_query_re_reads(image_dataset):
    fps = image_dataset.fps
    dataset = _reload(image_dataset, {CAMERA: [-1 / fps, 0.0]})
    dataset[2]
    view = dataset.reader._current_row_view
    assert CAMERA not in view.column_names
    assert "observation.state" in view.column_names and "episode_index" in view.column_names


def test_row_query_keeps_columns_without_delta_indices(image_dataset):
    fps = image_dataset.fps
    dataset = _reload(image_dataset, {"action": [0.0, 1 / fps]})
    dataset[2]
    assert CAMERA in dataset.reader._current_row_view.column_names


def test_no_delta_timestamps_uses_the_dataset_itself(image_dataset):
    dataset = _reload(image_dataset, None)
    dataset[2]
    assert dataset.reader._current_row_view is dataset.reader.hf_dataset


@pytest.mark.parametrize("window", [1, 2, 3])
def test_one_decode_per_needed_frame(image_dataset, monkeypatch, window):
    fps = image_dataset.fps
    deltas = [(-offset) / fps for offset in reversed(range(window))]
    dataset = _reload(image_dataset, {CAMERA: deltas})
    calls = _count_decoded_images(monkeypatch)
    dataset[3]
    assert len(calls) == window


def test_frames_are_unchanged(image_dataset):
    fps = image_dataset.fps
    deltas = {CAMERA: [-1 / fps, 0.0]}
    dataset = _reload(image_dataset, deltas)
    item = dataset[3]
    reference = _reload(image_dataset, None)
    torch.testing.assert_close(item[CAMERA][-1], reference[3][CAMERA])
    torch.testing.assert_close(item[CAMERA][0], reference[2][CAMERA])


def test_other_columns_are_unchanged(image_dataset):
    fps = image_dataset.fps
    dataset = _reload(image_dataset, {CAMERA: [-1 / fps, 0.0]})
    item = dataset[3]
    reference = _reload(image_dataset, None)[3]
    for key in ("episode_index", "index", "timestamp", "task_index"):
        assert item[key] == reference[key]
    assert item["task"] == reference["task"]
