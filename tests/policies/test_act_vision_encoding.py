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

"""ACT encodes its cameras in one backbone pass; the tokens must match the per-camera loop."""

import einops
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTSinusoidalPositionEmbedding2d

CAMERAS = ["observation.images.a", "observation.images.b", "observation.images.c"]


@pytest.fixture
def model():
    config = ACTConfig(
        input_features={
            **{camera: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)) for camera in CAMERAS},
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        use_vae=False,
        device="cpu",
    )
    return ACT(config).eval()


def _per_camera_reference(model, images):
    """What the forward pass did before: one backbone call per camera, tokens appended in order."""
    tokens, pos_embeds = [], []
    for image in images:
        feature_map = model.backbone(image)["feature_map"]
        pos_embed = model.encoder_cam_feat_pos_embed(feature_map).to(dtype=feature_map.dtype)
        feature_map = model.encoder_img_feat_input_proj(feature_map)
        tokens.extend(list(einops.rearrange(feature_map, "b c h w -> (h w) b c")))
        pos_embeds.extend(list(einops.rearrange(pos_embed, "b c h w -> (h w) b c")))
    return torch.stack(tokens, 0), torch.stack(pos_embeds, 0)


@pytest.mark.parametrize(
    "shapes",
    [
        [(96, 96)],
        [(96, 96), (96, 96), (96, 96)],
        # The forward pass documents that H and W may differ between cameras while H*W does not.
        [(96, 96), (64, 144), (96, 96)],
    ],
)
def test_batched_encoding_matches_the_per_camera_loop(model, shapes):
    images = [torch.rand(2, 3, h, w) for h, w in shapes]
    with torch.no_grad():
        expected_tokens, expected_pos = _per_camera_reference(model, images)
        tokens, pos_embed = model._encode_images(images)
    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(pos_embed, expected_pos)


def test_tokens_stay_in_camera_order(model):
    """Camera n's tokens must be the n-th contiguous block, as the encoder's position embeddings assume."""
    images = [torch.rand(2, 3, 96, 96) for _ in CAMERAS]
    with torch.no_grad():
        tokens, _ = model._encode_images(images)
        first_only, _ = model._encode_images(images[:1])
    per_camera = tokens.shape[0] // len(images)
    torch.testing.assert_close(tokens[:per_camera], first_only)


def test_one_backbone_call_per_distinct_resolution(model):
    calls = []
    original = model.backbone.forward

    def counting_forward(x):
        calls.append(x.shape[0])
        return original(x)

    model.backbone.forward = counting_forward
    with torch.no_grad():
        model._encode_images([torch.rand(2, 3, 96, 96) for _ in range(3)])
        assert calls == [6]
        calls.clear()
        model._encode_images([torch.rand(2, 3, 96, 96), torch.rand(2, 3, 64, 144)])
        assert calls == [2, 2]


class TestPositionEmbeddingCache:
    def test_same_size_returns_the_same_table(self):
        embedding = ACTSinusoidalPositionEmbedding2d(32)
        first = embedding(torch.rand(2, 8, 6, 7))
        assert embedding(torch.rand(5, 8, 6, 7)) is first

    def test_distinct_sizes_get_distinct_tables(self):
        embedding = ACTSinusoidalPositionEmbedding2d(32)
        first = embedding(torch.rand(2, 8, 6, 7))
        second = embedding(torch.rand(2, 8, 3, 3))
        assert second is not first
        assert len(embedding._cache) == 2

    def test_value_is_unchanged_by_caching(self):
        embedding = ACTSinusoidalPositionEmbedding2d(32)
        x = torch.rand(2, 8, 6, 7)
        torch.testing.assert_close(embedding(x), embedding._make_pos_embed(x))


def test_latent_sample_is_allocated_on_the_batch_device(model):
    """The zero latent used without the VAE was built on the host and copied every step."""
    from lerobot.utils.constants import OBS_IMAGES

    images = [torch.rand(2, 3, 96, 96) for _ in CAMERAS]
    batch = {
        **{camera: images[i] for i, camera in enumerate(CAMERAS)},
        OBS_IMAGES: images,
        "observation.state": torch.rand(2, 7),
        "action": torch.rand(2, 10, 7),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool),
    }
    with torch.no_grad():
        actions, (mu, log_sigma_x2) = model(batch)
    assert actions.shape == (2, 10, 7)
    assert mu is None and log_sigma_x2 is None
