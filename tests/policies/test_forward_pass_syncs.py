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

"""A policy's forward pass reports its sub-losses without reading them back from the accelerator."""

from pathlib import Path

import pytest
import torch

import lerobot.policies
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VqVae
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

CAMERA = "observation.images.cam"

POLICIES_ROOT = Path(lerobot.policies.__file__).parent
POLICY_DIRS = [d for d in POLICIES_ROOT.iterdir() if d.is_dir() and (d / f"modeling_{d.name}.py").exists()]


def _act_policy(use_vae):
    config = ACTConfig(
        input_features={
            CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
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
        use_vae=use_vae,
        device="cpu",
    )
    stats = {
        CAMERA: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    return ACTPolicy(config, dataset_stats=stats)


@pytest.mark.parametrize("use_vae", [False, True])
def test_act_reports_sub_losses_as_tensors(use_vae):
    policy = _act_policy(use_vae)
    batch = {
        CAMERA: torch.rand(2, 3, 96, 96),
        "observation.state": torch.rand(2, 7),
        "action": torch.rand(2, 10, 7),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool),
    }
    loss, loss_dict = policy(batch)
    assert loss.requires_grad
    expected = {"l1_loss", "kld_loss"} if use_vae else {"l1_loss"}
    assert set(loss_dict) == expected
    for name, value in loss_dict.items():
        assert isinstance(value, torch.Tensor), name
        assert value.ndim == 0 and not value.requires_grad, name


def test_sub_loss_tensors_reach_the_metrics_tracker():
    """The tracker is what makes reporting tensors possible; without it they would be dropped."""
    policy = _act_policy(use_vae=True)
    batch = {
        CAMERA: torch.rand(2, 3, 96, 96),
        "observation.state": torch.rand(2, 7),
        "action": torch.rand(2, 10, 7),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool),
    }
    _, loss_dict = policy(batch)
    tracker = MetricsTracker(2, 10, 2, {"loss": AverageMeter("loss", ":.3f")})
    tracker.update_metrics(loss_dict)
    assert tracker.metrics["l1_loss"].avg == pytest.approx(float(loss_dict["l1_loss"]))
    assert tracker.metrics["kld_loss"].avg == pytest.approx(float(loss_dict["kld_loss"]))


class TestVqVaeDiscretizedMirror:
    """VQ-BeT branches on this flag on the first line of its forward, so it must not read a buffer."""

    @pytest.fixture
    def vqvae(self):
        config = VQBeTConfig(
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (7,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        )
        return VqVae(config)

    def test_starts_untrained(self, vqvae):
        assert vqvae.is_discretized is False
        assert bool(vqvae.discretized) is False

    def test_set_updates_both_the_mirror_and_the_buffer(self, vqvae):
        vqvae.set_discretized(True)
        assert vqvae.is_discretized is True
        assert bool(vqvae.discretized) is True

    def test_mirror_is_restored_from_a_checkpoint(self, vqvae):
        vqvae.set_discretized(True)
        reloaded = VqVae(vqvae.config)
        assert reloaded.is_discretized is False
        reloaded.load_state_dict(vqvae.state_dict())
        assert reloaded.is_discretized is True
        assert bool(reloaded.discretized) is True

    def test_mirror_stays_false_for_an_untrained_checkpoint(self, vqvae):
        reloaded = VqVae(vqvae.config)
        reloaded.set_discretized(True)
        reloaded.load_state_dict(vqvae.state_dict())
        assert reloaded.is_discretized is False


class TestNoPolicyForwardReadsTheAccelerator:
    """A structural check, so a new policy cannot quietly reintroduce a per-step stall.

    Instantiating every policy here is not an option - most VLAs pull a pretrained backbone - so
    this reads the source instead. `.item()` and `.tolist()` in a training `forward` copy a value
    from the accelerator to the host, which drains the queue before the backward is even issued.
    """

    # (policy, attribute): a device read that is not a metric and has no tensor-valued equivalent.
    ALLOWED = {
        ("eo1", "item"),  # shape consistency check that raises; a bool is genuinely needed
        ("pi0", "tolist"),  # per-dimension loss vector; the tracker averages scalars only
        ("pi05", "tolist"),  # likewise
    }

    def _offenders(self, path):
        import ast

        tree = ast.parse(path.read_text())
        found = []
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"):
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"item", "tolist"}
                    ):
                        found.append((node.func.attr, node.lineno))
        return found

    @pytest.mark.parametrize("policy_dir", sorted(POLICY_DIRS), ids=lambda p: p.name)
    def test_forward_does_not_copy_to_the_host(self, policy_dir):
        path = policy_dir / f"modeling_{policy_dir.name}.py"
        if not path.exists():
            pytest.skip(f"{policy_dir.name} has no modeling module")
        offenders = [
            (attr, line)
            for attr, line in self._offenders(path)
            if (policy_dir.name, attr) not in self.ALLOWED
        ]
        assert not offenders, (
            f"{path}: forward() copies to the host at "
            + ", ".join(f"line {line} (.{attr}())" for attr, line in offenders)
            + ". Report the value as a detached tensor instead; the metrics tracker accumulates it "
            "on the accelerator."
        )
