"""Append original state rows to an unchanged current/goal visual pair."""
from pathlib import Path

import numpy as np
import torch
from ..common import sha_file


class CubeStateVisualDataset(torch.utils.data.Dataset):
    def __init__(self, visual_dataset, observations_path, expected_sha256=None):
        self.visual_dataset = visual_dataset
        self.observations_path = str(Path(observations_path).resolve())
        self.split, self.rows, self.transitions = visual_dataset.split, visual_dataset.rows, visual_dataset.transitions
        if Path(self.observations_path).parent.name != self.split or self.rows != 1001 or self.transitions != 1000:
            raise ValueError('Original matching train/val raw1001/compact1000 state source required')
        if expected_sha256 is not None and sha_file(self.observations_path) != expected_sha256:
            raise ValueError('State observation source SHA differs')
        self._observations = None
        self._open_observations()

    def _open_observations(self):
        if self._observations is None:
            states = np.load(self.observations_path, mmap_mode='r', allow_pickle=False)
            if states.dtype != np.float32 or states.shape != (len(self.visual_dataset.entries) * self.rows, 28):
                raise ValueError('Original float32 raw rows x official28 observations required')
            self._observations = states
        return self._observations

    def __getstate__(self):
        state = self.__dict__.copy(); state['_observations'] = None
        return state

    def __len__(self):
        return len(self.visual_dataset)

    def __getitem__(self, pair):
        result = self.visual_dataset[pair]
        states = self._open_observations()
        for key, row_key in (('current_state', 'source_row'), ('goal_state', 'goal_source_row')):
            row = int(result[row_key])
            if not 0 <= row < len(states) or row % self.rows == self.transitions:
                raise ValueError('Raw terminal or out-of-split state may not enter compact sample')
            state = torch.from_numpy(np.array(states[row], copy=True))
            if not torch.isfinite(state).all():
                raise ValueError('Nonfinite original state row')
            result[key] = state
        return result
