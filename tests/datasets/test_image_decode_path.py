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

"""The reader decodes embedded images itself; everything else still sees decoded PIL images."""

import io

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from lerobot.datasets.io_utils import encoded_to_chw_tensor, hf_transform_to_torch
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
    built = empty_lerobot_dataset_factory(root=tmp_path / "decode", features=features)
    rng = np.random.default_rng(0)
    for _ in range(4):
        built.add_frame(
            {
                CAMERA: rng.integers(0, 256, DUMMY_CHW[1:] + (3,), dtype=np.uint8),
                "observation.state": torch.zeros(2),
                "action": torch.zeros(2),
                "task": "dummy",
            }
        )
    built.save_episode()
    built.finalize()
    return LeRobotDataset(built.repo_id, root=built.root, download_videos=False)


def _encoded(array):
    buffer = io.BytesIO()
    PILImage.fromarray(array).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def test_encoded_decode_matches_pil():
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
    decoded = encoded_to_chw_tensor(_encoded(array))
    assert decoded.dtype == torch.uint8 and decoded.shape == (3, 12, 9)
    torch.testing.assert_close(decoded, torch.from_numpy(array).permute(2, 0, 1))


def test_batch_transform_handles_encoded_columns():
    rng = np.random.default_rng(2)
    array = rng.integers(0, 256, (6, 5, 3), dtype=np.uint8)
    out = hf_transform_to_torch({CAMERA: [_encoded(array), _encoded(array)]})
    assert all(frame.dtype == torch.float32 for frame in out[CAMERA])
    torch.testing.assert_close(out[CAMERA][0], torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)


def test_the_shared_dataset_still_decodes_to_pil(dataset):
    """Four tools read the dataset with its transform stripped and expect PIL images."""
    dataset[0]
    raw = dataset.hf_dataset.with_format(None)[0][CAMERA]
    assert isinstance(raw, PILImage.Image)


def test_the_readers_own_view_leaves_images_encoded(dataset):
    dataset[0]
    view = dataset.reader._row_view().with_format(None)[0][CAMERA]
    assert isinstance(view, dict) and set(view) == {"bytes", "path"}


def test_frames_are_unchanged_by_the_decode_path(dataset):
    """What get_item returns must not depend on which decoder produced it."""
    item = dataset[1]
    expected = np.array(dataset.hf_dataset.with_format(None)[1][CAMERA])
    torch.testing.assert_close(item[CAMERA], torch.from_numpy(expected).permute(2, 0, 1).float() / 255.0)


def test_depth_columns_keep_the_pil_path(dataset):
    """Depth maps are 16-bit; PIL is what reads their native units."""
    reader = dataset.reader
    dataset[0]
    reader._undecoded_keys = None
    reader._meta.features[CAMERA]["info"] = {"is_depth_map": True}
    assert CAMERA not in reader._undecoded_image_keys()


def test_views_share_the_datasets_transform(dataset):
    """A view that installed its own transform would silently drop the reader's options."""
    reader = dataset.reader
    dataset[0]

    def marker(items_dict):
        return hf_transform_to_torch(items_dict)

    reader.hf_dataset.set_transform(marker)
    reader._column_views = {}
    assert reader._row_view()._format_kwargs["transform"] is marker
    assert reader._column_view("action")._format_kwargs["transform"] is marker
