from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.summarize_episode_outcomes import classify_episode, summarize_dataset


def test_classifies_success_after_reset():
    result = classify_episode([False, False, True], [[0, 0, .02], [0, 0, .06], [0, 0, .02]], [[0, 0, .02]] * 3)
    assert result['reached_goal'] and result['lifted']
    assert result['failure_reason'] == 'none'
    assert result['final_goal_error'] == 0


def test_reset_success_is_not_a_successful_rollout():
    result = classify_episode([True, False, False], [[0, 0, .02]] * 3, [[.2, 0, .02]] * 3)
    assert not result['reached_goal']
    assert result['failure_reason'] == 'never_lifted'
    assert result['final_goal_error'] == pytest.approx(.2)


def test_lifted_not_placed_is_separate():
    result = classify_episode([False] * 3, [[0, 0, .02], [0, 0, .08], [0, 0, .02]], [[.2, 0, .02]] * 3)
    assert result['lifted'] and not result['reached_goal']
    assert result['failure_reason'] == 'lifted_not_placed'


def test_rejects_nonfinite_or_inconsistent_episode():
    with pytest.raises(ValueError):
        classify_episode([False] * 3, [[np.nan, 0, .02]] * 3, [[0, 0, .02]] * 3)
    with pytest.raises(ValueError):
        classify_episode([False] * 2, [[0, 0, .02]] * 3, [[0, 0, .02]] * 3)


def test_hdf5_report_counts_duplicates_and_does_not_write(tmp_path: Path):
    path = tmp_path / 'data.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [3, 3]
        f['ep_offset'] = [0, 3]
        f['success'] = [False, False, True, False, False, False]
        f['privileged/block_0_pos'] = [[0, 0, .02]] * 6
        f['privileged/target_block_pos'] = [[0, 0, .02]] * 6
        f['qpos'] = np.zeros((6, 16))
        f['qvel'] = np.zeros((6, 15))
    before = path.read_bytes()
    report = summarize_dataset(path)
    assert report['episodes'] == 2
    assert report['successes'] == 1 and report['failures'] == 1
    assert report['success_rate'] == .5
    assert report['duplicate_initial_states'] == 1
    assert report['failure_reasons'] == {'never_lifted': 1}
    assert len(report['records']) == 2
    assert path.read_bytes() == before
