"""Goal relabeling on the FourRoom HDF5 schema (raw environment timesteps)."""

from pathlib import Path
from typing import cast

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GoalClips(Dataset[dict[str, Tensor]]):
    pixels: h5py.Dataset

    def __init__(self, path: Path, horizon: int, goal_max: int, *, use_proprio: bool, training: bool):
        self.path = path
        self.horizon = horizon
        self.goal_max = goal_max
        self.use_proprio = use_proprio
        self.training = training
        self.file: h5py.File | None = None
        with h5py.File(path, 'r') as file:
            self.offsets = cast(h5py.Dataset, file['ep_offset'])[:]
            self.lengths = cast(h5py.Dataset, file['ep_len'])[:]
            self.action_dim = cast(h5py.Dataset, file['action']).shape[-1]
            self.state_dim = cast(h5py.Dataset, file['proprio']).shape[-1] if use_proprio else 0
        # A length-L episode has L-H windows, including those ending at its last observation.
        counts = np.maximum(self.lengths - horizon, 0)
        self.boundaries = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    def __len__(self) -> int:
        return int(self.boundaries[-1])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode = int(np.searchsorted(self.boundaries, index, side='right')) - 1
        start = int(self.offsets[episode] + index - self.boundaries[episode])
        future = start + self.horizon
        end = int(self.offsets[episode] + self.lengths[episode]) - 1
        max_goal = min(start + self.goal_max, end)
        goal = int(torch.randint(future, max_goal + 1, ()).item()) if self.training else max_goal
        if self.file is None:
            # FourRoom packs 100 frames in a ~14 MiB raw chunk. Nearby rows can
            # share decompression only while their dataset handle and cache stay open.
            self.file = h5py.File(self.path, 'r', rdcc_nbytes=32 * 1024**2)
            self.pixels = cast(h5py.Dataset, self.file['pixels'])
        actions = cast(h5py.Dataset, self.file['action'])
        frames = [start, future, goal] if goal > future else [start, future]
        # Read directly into the output; sparse HDF5 selection over image rows
        # is much slower than copying each row via a contiguous slice.
        pixels = np.empty((len(frames), *self.pixels.shape[1:]), dtype=self.pixels.dtype)
        for i, frame in enumerate(frames):
            self.pixels.read_direct(pixels, np.s_[frame:frame + 1], np.s_[i:i + 1])
        images = torch.from_numpy(pixels).permute(0, 3, 1, 2)
        goal_index = len(frames) - 1
        sample = {'current': images[0], 'future': images[1], 'goal': images[goal_index],
                  'actions': torch.from_numpy(actions[start:future])}
        if self.use_proprio:
            states = torch.from_numpy(cast(h5py.Dataset, self.file['proprio'])[frames]).float()
            sample.update(current_state=states[0], future_state=states[1], goal_state=states[goal_index])
        return sample

    def proprio_stats(self) -> tuple[Tensor, Tensor]:
        with h5py.File(self.path, 'r') as file:
            states = torch.from_numpy(cast(h5py.Dataset, file['proprio'])[:]).float()
        return states.mean(0), states.std(0, correction=0).clamp_min(1e-6)
