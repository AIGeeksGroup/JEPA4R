from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.accepted_episode_h5 import (
    classify_episodes,
    copy_accepted_episodes,
    validate_labeled_dataset,
)


def write_toy(path: Path):
    lengths = np.array([3, 2, 4], dtype=np.int64)
    offsets = np.array([0, 3, 5], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f["ep_len"] = lengths
        f["ep_offset"] = offsets
        f["success"] = np.array(
            [1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8
        )
        f["action"] = np.arange(45, dtype=np.float32).reshape(9, 5)
        f["qpos"] = np.zeros((9, 28), dtype=np.float32)
        f["qvel"] = np.zeros((9, 26), dtype=np.float32)
        f["control"] = np.zeros((9, 7), dtype=np.float32)
        f["observation"] = np.zeros((9, 37), dtype=np.float32)
        f["pcd_xyz"] = np.zeros((9, 1200, 3), dtype=np.float32)
        f["pcd_rgb"] = np.zeros((9, 1200, 3), dtype=np.uint8)
        privileged = f.create_group("privileged")
        privileged["block_0_pos"] = np.arange(27, dtype=np.float32).reshape(9, 3)


def test_reset_frame_success_does_not_make_episode_successful(tmp_path):
    src = tmp_path / "toy.h5"
    write_toy(src)
    with h5py.File(src, "r") as f:
        assert classify_episodes(f) == [False, False, True]


def test_copy_exact_successful_episode_and_add_provenance(tmp_path):
    src = tmp_path / "toy.h5"
    dst = tmp_path / "accepted.h5"
    write_toy(src)
    copied = copy_accepted_episodes(
        src,
        dst,
        outcome="success",
        limit=1,
        family="double",
        task_id=4,
        policy_kind="expert",
    )
    assert copied == 1
    with h5py.File(dst, "r") as f:
        np.testing.assert_array_equal(f["ep_len"][:], [4])
        np.testing.assert_array_equal(f["ep_offset"][:], [0])
        np.testing.assert_array_equal(f["action"][:], np.arange(45).reshape(9, 5)[5:9])
        np.testing.assert_array_equal(f["privileged/block_0_pos"][:], np.arange(27).reshape(9, 3)[5:9])
        assert np.all(f["source_policy"][:] == 1)
        assert np.all(f["accepted_success"][:] == 1)
        assert np.all(f["fixed_task_id"][:] == 4)
        assert np.all(f["cube_count"][:] == 2)


def test_copy_failure_excludes_reset_only_success(tmp_path):
    src = tmp_path / "toy.h5"
    dst = tmp_path / "failures.h5"
    write_toy(src)
    copied = copy_accepted_episodes(
        src,
        dst,
        outcome="failure",
        limit=2,
        family="double",
        task_id=1,
        policy_kind="random",
    )
    assert copied == 2
    with h5py.File(dst, "r") as f:
        np.testing.assert_array_equal(f["ep_len"][:], [3, 2])
        assert not f["accepted_success"][:].any()
        assert not f["source_policy"][:].any()


def test_copy_refuses_existing_destination(tmp_path):
    src = tmp_path / "toy.h5"
    dst = tmp_path / "exists.h5"
    write_toy(src)
    dst.touch()
    with pytest.raises(FileExistsError):
        copy_accepted_episodes(
            src, dst, outcome="success", limit=1,
            family="double", task_id=1, policy_kind="expert"
        )


def test_validator_checks_task_dependent_ur5e_widths_and_outcome(tmp_path):
    src = tmp_path / "toy.h5"
    dst = tmp_path / "accepted.h5"
    write_toy(src)
    copy_accepted_episodes(
        src, dst, outcome="success", limit=1,
        family="double", task_id=4, policy_kind="expert"
    )
    report = validate_labeled_dataset(
        dst,
        family="double",
        task_id=4,
        expected_episodes=1,
        expected_policy_counts={"expert": 1, "random": 0},
    )
    assert report["episodes"] == 1
    assert report["rows"] == 4
    assert report["successes"] == 1

    with h5py.File(dst, "r+") as f:
        del f["qpos"]
        f["qpos"] = np.zeros((4, 21), dtype=np.float32)
    with pytest.raises(ValueError, match="qpos"):
        validate_labeled_dataset(
            dst,
            family="double",
            task_id=4,
            expected_episodes=1,
            expected_policy_counts={"expert": 1, "random": 0},
        )


def test_validator_allows_only_full_nan_action_on_episode_terminal_row(tmp_path):
    src = tmp_path / "toy.h5"
    dst = tmp_path / "accepted.h5"
    write_toy(src)
    copy_accepted_episodes(
        src, dst, outcome="success", limit=1,
        family="double", task_id=4, policy_kind="expert"
    )
    with h5py.File(dst, "r+") as f:
        f["action"][-1] = np.nan
    validate_labeled_dataset(
        dst,
        family="double",
        task_id=4,
        expected_episodes=1,
        expected_policy_counts={"expert": 1, "random": 0},
    )

    with h5py.File(dst, "r+") as f:
        f["action"][1, 0] = np.nan
    with pytest.raises(ValueError, match="action"):
        validate_labeled_dataset(
            dst,
            family="double",
            task_id=4,
            expected_episodes=1,
            expected_policy_counts={"expert": 1, "random": 0},
        )
