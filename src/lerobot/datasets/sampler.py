#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _as_lookup_arrays(mapping: dict[int, int] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Turn an index mapping into sorted key/value arrays for vectorized lookup.

    Args:
        mapping (`dict[int, int] | None`):
            Dataset frame index to row position, or `None` when the dataset holds every episode.

    Returns:
        `tuple[np.ndarray | None, np.ndarray | None]`: The sorted keys and their values, or
        `(None, None)` when there is nothing to map.
    """
    if not mapping:
        return None, None
    keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    values = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    order = np.argsort(keys)
    return keys[order], values[order]


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
        continuous: bool = False,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
            absolute_to_relative_idx: Mapping from dataset frame index to row position, when the
                dataset holds a subset of the episodes.
            continuous: Yield epoch after epoch from a single iterator instead of ending at each
                epoch boundary. Keeps the DataLoader's prefetch queue full across the boundary.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self.continuous = continuous
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx
        self._relative_keys, self._relative_values = _as_lookup_arrays(absolute_to_relative_idx)

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def _frame_indices(self, positions: np.ndarray) -> np.ndarray:
        """Translate sampler positions into dataset indices, all of them at once.

        The per-position form costs a `searchsorted` and three Python casts each, in the main
        process, for every index the loop consumes. One vectorized call per epoch does the same work.
        """
        episodes = np.searchsorted(self._cum_lengths, positions, side="right")
        offsets = np.where(episodes > 0, self._cum_lengths[np.maximum(episodes - 1, 0)], 0)
        absolute = self._starts[episodes] + (positions - offsets)
        if self._relative_keys is None:
            return absolute
        return self._relative_values[np.searchsorted(self._relative_keys, absolute)]

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        if not self.continuous:
            return self._iter_epoch(epoch, start)
        return self._iter_continuous(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        yield from (int(index) for index in self._epoch_indices(epoch, start))

    def _iter_continuous(self, epoch: int, start: int) -> Iterator[int]:
        """Yield one epoch after another without ever ending.

        A finite sampler ends the DataLoader's iterator once per epoch. Restarting it drains the
        workers' prefetch queue, and the accelerator waits for a full fetch round before the next
        step can run. Iterating without a break keeps the queue full across the boundary; the frames
        of an epoch's tail simply share a batch with the next epoch's head, and no frame is skipped
        or repeated within an epoch.
        """
        while True:
            yield from self._iter_epoch(epoch, start)
            epoch += 1
            start = 0
            self._epoch = epoch + 1

    def _epoch_indices(self, epoch: int, start: int) -> np.ndarray:
        positions = (
            torch.randperm(self._num_frames, generator=self._epoch_generator(epoch)).numpy()
            if self.shuffle
            else np.arange(self._num_frames)
        )
        return self._frame_indices(positions[start:])

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(
    step: int, num_frames: int, batch_size: int, num_processes: int, continuous: bool = False
) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off. A `continuous` sampler has neither a short batch nor padding, so its
          position is exact.
    """
    if continuous:
        # Nothing is padded and no batch is short, so the position is simply what has been consumed.
        position = step * batch_size * num_processes
        epoch, start_index = divmod(position, num_frames)
        return {"epoch": epoch, "start_index": start_index}
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
