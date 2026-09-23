"""Training frames for the reconstruction decoder: what the frozen encoder reads, and the point cloud of the
same simulator state as the target.

Point-cloud checkpoints read both from one dataset. A pixel checkpoint needs two: its pixel dataset and a
point-cloud dataset holding the SAME rows (same episodes, same steps) -- the 2D and 3D datasets of one paired
collection (scripts/data/collect_cube_pcd.py). Separately collected datasets are different episodes and are refused."""

import os

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch


class PairedFrames(torch.utils.data.Dataset):
    """Row i of ``inputs`` + row i of ``targets``; only ``obs_key`` and ``target`` are kept."""

    def __init__(self, inputs, targets, obs_key, align_on=('qpos',)):
        assert np.array_equal(inputs.lengths, targets.lengths) and np.array_equal(inputs.offsets, targets.offsets), \
            'input and target datasets do not hold the same episodes (ep_len / ep_offset differ)'
        for col in align_on:
            a, b = inputs.get_col_data(col), targets.get_col_data(col)
            assert a.shape == b.shape and np.array_equal(a, b), \
                f"input and target datasets are not the same rows: '{col}' differs (max |diff| {np.abs(a - b).max() if a.shape == b.shape else 'shape'})"
        self.inputs, self.targets, self.obs_key = inputs, targets, obs_key
        self.lengths, self.offsets = targets.lengths, targets.offsets

    def get_col_data(self, col):  # per-row columns (object poses, actions) are the same rows in both
        return self.targets.get_col_data(col)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {self.obs_key: self.inputs[idx][self.obs_key], 'target': self.targets[idx]['target']}


def build_dataset(cfg, obs_key, input_keys, input_preprocessor):
    """cfg: ``config/recon/data/*.yaml``. One sample = one frame: ``{obs_key: (1, ...), 'target': (1, N, C)}``."""
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    target_preprocessor = hydra.utils.instantiate(cfg.target_preprocessor)

    def load(name, keys, cache=()):
        return swm.data.load_dataset(name, transform=None, cache_dir=cache_dir, num_steps=1, frameskip=1,
                                     keys_to_load=list(keys), keys_to_cache=list(cache))

    if cfg.target == cfg.input:
        dataset = load(cfg.input, dict.fromkeys([*input_keys, *cfg.target_keys]))
        # target first: the encoder's own preprocessor may drop the columns it consumes
        dataset.transform = spt.data.transforms.Compose(target_preprocessor, input_preprocessor)
        return dataset

    inputs, targets = load(cfg.input, input_keys, cfg.align_on), load(cfg.target, cfg.target_keys, cfg.align_on)
    inputs.transform, targets.transform = input_preprocessor, target_preprocessor
    return PairedFrames(inputs, targets, obs_key, cfg.align_on)


class Episodes:
    """Whole trajectories out of a frame dataset built by ``build_dataset`` (one dataset, or a row-aligned pair)."""

    def __init__(self, frames):
        self.frames = frames
        self.lengths = [int(n) for n in frames.lengths]
        self.first = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])   # index of step 0 of every episode
        self._cols = {}

    def __len__(self):
        return len(self.lengths)

    def rows(self, ep, stride=1):
        return self.first[ep] + np.arange(0, self.lengths[ep], stride)

    def load(self, ep, stride=1):
        """every ``stride``-th frame of episode ``ep``: ``{obs_key: (K, ...), 'target': (K, N, C)}``."""
        items = [self.frames[int(i)] for i in self.rows(ep, stride)]
        return {k: torch.cat([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}

    def column(self, name, ep=None, stride=1):
        """a per-row column of the dataset (e.g. ``action``, ``privileged_block_0_pos``), raw: the rows of one
        episode, or all rows. ``KeyError`` if the dataset has no such column."""
        if name not in self._cols:
            self._cols[name] = np.asarray(self.frames.get_col_data(name))
            for d in (self.frames, getattr(self.frames, 'inputs', None), getattr(self.frames, 'targets', None)):
                if getattr(d, 'h5_file', None) is not None:          # reopened lazily; an open handle must not cross a DataLoader fork
                    d.h5_file.close()
                    d.h5_file = None
        return self._cols[name] if ep is None else self._cols[name][self.rows(ep, stride)]
