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

"""Decoded frames are returned as the decoder stacked them, and each is checked against its own request."""

import subprocess

import pytest
import torch

from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames_torchcodec

FPS = 10


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x64:rate={FPS}:duration=3",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"ffmpeg is needed to build the fixture video: {error}")
    return path


def test_frames_are_returned_in_the_requested_order(video):
    timestamps = [0.5, 0.1, 0.9]
    frames = decode_video_frames_torchcodec(video, timestamps, 1 / FPS, return_uint8=True)
    assert frames.shape[0] == len(timestamps)
    for position, timestamp in enumerate(timestamps):
        one = decode_video_frames_torchcodec(video, [timestamp], 1 / FPS, return_uint8=True)
        torch.testing.assert_close(frames[position], one[0])


def test_repeated_timestamps_are_both_returned(video):
    frames = decode_video_frames_torchcodec(video, [0.4, 0.4], 1 / FPS, return_uint8=True)
    assert frames.shape[0] == 2
    torch.testing.assert_close(frames[0], frames[1])


def test_frames_survive_a_later_decode_on_the_same_decoder(video):
    """The batch must own its storage; the decoder is cached and reused across samples."""
    first = decode_video_frames_torchcodec(video, [0.1, 0.2], 1 / FPS, return_uint8=True)
    reference = first.clone()
    decode_video_frames_torchcodec(video, [1.5, 1.6], 1 / FPS, return_uint8=True)
    torch.testing.assert_close(first, reference)


@pytest.mark.parametrize("return_uint8", [False, True])
def test_dtype_contract(video, return_uint8):
    frames = decode_video_frames_torchcodec(video, [0.3], 1 / FPS, return_uint8=return_uint8)
    if return_uint8:
        assert frames.dtype == torch.uint8 and frames.max() > 1
    else:
        assert frames.dtype == torch.float32 and frames.min() >= 0.0 and frames.max() <= 1.0


def test_a_timestamp_outside_tolerance_is_rejected(video):
    with pytest.raises(FrameTimestampError, match="tolerance"):
        decode_video_frames_torchcodec(video, [0.55], 1e-6)


def test_tolerance_is_checked_per_request(video):
    """A frame near one requested timestamp must not satisfy a different one."""
    with pytest.raises(FrameTimestampError, match="tolerance"):
        decode_video_frames_torchcodec(video, [0.1, 0.15], 1e-6)
