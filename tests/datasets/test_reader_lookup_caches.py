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

"""Lookups that do not change while a dataset is read happen once, not once per sample."""

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_CHW

CAMERA = "observation.images.cam"


@pytest.fixture
def dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        CAMERA: {"dtype": "image", "shape": DUMMY_CHW, "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    built = empty_lerobot_dataset_factory(root=tmp_path / "reader", features=features)
    rng = np.random.default_rng(0)
    for episode in range(2):
        for _ in range(4):
            built.add_frame(
                {
                    CAMERA: rng.integers(0, 256, DUMMY_CHW[1:] + (3,), dtype=np.uint8),
                    "observation.state": torch.zeros(2),
                    "action": torch.zeros(2),
                    "task": f"task {episode}",
                }
            )
        built.save_episode()
    built.finalize()
    # Delta timestamps are what make `get_item` consult the episode's bounds, which is the lookup
    # these caches are about.
    fps = built.fps
    return LeRobotDataset(
        built.repo_id,
        root=built.root,
        delta_timestamps={CAMERA: [-1 / fps, 0.0]},
        download_videos=False,
    )


def test_episode_rows_are_fetched_once_per_episode(dataset, monkeypatch):
    reader = dataset.reader
    dataset[0]
    fetched = []
    episodes = dataset.meta.episodes

    class CountingEpisodes:
        def __getitem__(self, index):
            fetched.append(index)
            return episodes[index]

        def __len__(self):
            return len(episodes)

    dataset.meta.episodes = CountingEpisodes()
    reader._episode_rows = {}
    for index in range(len(dataset)):
        dataset[index]
    assert sorted(set(fetched)) == [0, 1]
    assert len(fetched) == 2, f"one fetch per episode, got {fetched}"


def test_key_groups_match_the_metadata(dataset):
    reader = dataset.reader
    dataset[0]
    assert reader._keys("camera") == list(dataset.meta.camera_keys)
    assert reader._keys("video") == list(dataset.meta.video_keys)
    assert reader._keys("depth") == list(dataset.meta.depth_keys)


def test_task_names_match_the_metadata(dataset):
    reader = dataset.reader
    dataset[0]
    for index in range(len(dataset.meta.tasks)):
        assert reader._task_name(index) == dataset.meta.tasks.iloc[index].name


def test_every_sample_still_carries_its_own_task(dataset):
    tasks = [dataset[index]["task"] for index in range(len(dataset))]
    assert tasks[:4] == ["task 0"] * 4
    assert tasks[4:] == ["task 1"] * 4


def test_caches_are_dropped_when_the_dataset_is_reloaded(dataset):
    reader = dataset.reader
    dataset[0]
    assert reader._episode_rows, "reading a sample should have cached its episode row"
    assert reader._key_groups is not None
    reader.hf_dataset = reader._load_hf_dataset()
    assert reader._episode_rows == {}
    assert reader._key_groups is None
    assert reader._task_names is None
    assert reader._video_paths == {}
    assert dataset[0]["task"] == "task 0"  # and it refills correctly
