import subprocess
import sys

import h5py
import numpy as np
import pytest

from scripts.data.summarize_episode_outcomes import classify_episode, summarize_dataset


def test_nonfinite_success_is_not_success():
    with pytest.raises(ValueError):
        classify_episode([False, np.nan], [[0, 0, .02]] * 2, [[0, 0, .02]] * 2)


def test_fractional_lengths_are_rejected(tmp_path):
    path = tmp_path / 'fraction.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [2.5, 2.5]
        f['ep_offset'] = [0, 2.5]
        f['success'] = [False] * 5
        f['privileged/block_0_pos'] = [[0, 0, .02]] * 5
        f['privileged/target_block_pos'] = [[0, 0, .02]] * 5
        f['qpos'] = np.zeros((5, 16))
        f['qvel'] = np.zeros((5, 15))
    with pytest.raises(ValueError):
        summarize_dataset(path)


def test_cli_cannot_overwrite_source_dataset(tmp_path):
    from scripts.data import summarize_episode_outcomes as module
    path = tmp_path / 'source.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [2]
        f['ep_offset'] = [0]
        f['success'] = [False, True]
        f['privileged/block_0_pos'] = [[0, 0, .02]] * 2
        f['privileged/target_block_pos'] = [[0, 0, .02]] * 2
        f['qpos'] = np.zeros((2, 16))
        f['qvel'] = np.zeros((2, 15))
    before = path.read_bytes()
    result = subprocess.run([sys.executable, module.__file__, str(path), '--out', str(path)], capture_output=True)
    assert result.returncode != 0
    assert path.read_bytes() == before
