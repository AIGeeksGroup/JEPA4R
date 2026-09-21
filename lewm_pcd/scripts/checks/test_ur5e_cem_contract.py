"""Contract tests for replaying the fixed UR5e Cube holdout sets."""

import json

import h5py
import numpy as np
import pytest

from envs.ur5e_cube_tasks import get_task


def write_holdout(path, *, task_id=1, offsets=(0, 3), successful=True):
    with h5py.File(path, "w") as h5:
        h5["ep_len"] = np.array([3, 3], dtype=np.int64)
        h5["ep_offset"] = np.array(offsets, dtype=np.int64)
        h5["fixed_task_id"] = np.full(6, task_id, dtype=np.uint8)
        h5["source_policy"] = np.ones(6, dtype=np.uint8)
        h5["accepted_success"] = np.full(6, successful, dtype=np.bool_)
        h5["qpos"] = np.zeros((6, 28), dtype=np.float64)
        h5["qvel"] = np.zeros((6, 26), dtype=np.float64)
        h5["pcd_xyz"] = np.zeros((6, 1200, 3), dtype=np.float32)
        h5["pcd_rgb"] = np.zeros((6, 1200, 3), dtype=np.uint8)
        h5["action"] = np.zeros((6, 5), dtype=np.float32)


def test_episode_specs_use_ep_len_offsets_and_final_goal(tmp_path):
    from scripts.eval.ur5e_cem_contract import inspect_holdout

    path = tmp_path / "holdout.h5"
    write_holdout(path)
    specs = inspect_holdout(path, get_task("double", 1), require_full=False)
    assert [(s.episode_index, s.offset, s.length, s.goal_step) for s in specs] == [
        (0, 0, 3, 2),
        (1, 3, 3, 2),
    ]


def test_mismatched_task_id_is_rejected(tmp_path):
    from scripts.eval.ur5e_cem_contract import inspect_holdout

    path = tmp_path / "holdout.h5"
    write_holdout(path, task_id=2)
    with pytest.raises(ValueError, match="task ID"):
        inspect_holdout(path, get_task("double", 1), require_full=False)


def test_overlapping_episode_offsets_are_rejected(tmp_path):
    from scripts.eval.ur5e_cem_contract import inspect_holdout

    path = tmp_path / "holdout.h5"
    write_holdout(path, offsets=(0, 2))
    with pytest.raises(ValueError, match="offset"):
        inspect_holdout(path, get_task("double", 1), require_full=False)


def test_unsuccessful_expert_holdout_is_rejected(tmp_path):
    from scripts.eval.ur5e_cem_contract import inspect_holdout

    path = tmp_path / "holdout.h5"
    write_holdout(path, successful=False)
    with pytest.raises(ValueError, match="expert success"):
        inspect_holdout(path, get_task("double", 1), require_full=False)


def test_atomic_json_writes_complete_record(tmp_path):
    from scripts.eval.ur5e_cem_contract import atomic_json

    path = tmp_path / "results" / "episode_000.json"
    atomic_json(path, {"success": False, "episode_index": 0})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "success": False,
        "episode_index": 0,
    }
    assert list(path.parent.iterdir()) == [path]


def write_local_goal_dataset(path):
    lengths = np.array([31, 31], dtype=np.int64)
    total = int(lengths.sum())
    with h5py.File(path, "w") as h5:
        h5["ep_len"] = lengths
        h5["ep_offset"] = np.array([0, 31], dtype=np.int64)
        h5["fixed_task_id"] = np.ones(total, dtype=np.uint8)
        privileged = h5.create_group("privileged")
        for cube_id in range(2):
            pos = np.zeros((total, 3), dtype=np.float64)
            quat = np.zeros((total, 4), dtype=np.float64)
            quat[:, 0] = 1.0
            if cube_id == 1:
                pos[31:, 0] = np.arange(31) * 0.01
            privileged[f"block_{cube_id}_pos"] = pos
            privileged[f"block_{cube_id}_quat"] = quat


def test_local_goal_sampling_is_seeded_and_stays_inside_episodes(tmp_path):
    from scripts.eval.ur5e_cem_contract import sample_local_goal_specs

    path = tmp_path / "local.h5"
    write_local_goal_dataset(path)
    first = sample_local_goal_specs(path, get_task("double", 1), count=4, seed=42)
    again = sample_local_goal_specs(path, get_task("double", 1), count=4, seed=42)
    other = sample_local_goal_specs(path, get_task("double", 1), count=4, seed=43)

    assert first == again
    assert first != other
    assert all(spec.goal_step - spec.start_step == 25 for spec in first)
    assert all(spec.goal_step < spec.length for spec in first)
    assert [spec.start_row for spec in first] == sorted(spec.start_row for spec in first)


def test_nontrivial_local_goal_sampling_requires_cube_displacement(tmp_path):
    from scripts.eval.ur5e_cem_contract import sample_local_goal_specs

    path = tmp_path / "local.h5"
    write_local_goal_dataset(path)
    specs = sample_local_goal_specs(
        path,
        get_task("double", 1),
        count=4,
        seed=42,
        nontrivial_only=True,
    )

    assert all(spec.episode_index == 1 for spec in specs)
    assert all(spec.max_cube_displacement > 0.04 for spec in specs)
    assert all(not spec.start_already_success for spec in specs)


def test_local_goal_sampling_rejects_impossible_requests(tmp_path):
    from scripts.eval.ur5e_cem_contract import sample_local_goal_specs

    path = tmp_path / "local.h5"
    write_local_goal_dataset(path)
    with pytest.raises(ValueError, match="goal offset"):
        sample_local_goal_specs(path, get_task("double", 1), count=1, goal_offset=31)
    with pytest.raises(ValueError, match="requested"):
        sample_local_goal_specs(path, get_task("double", 1), count=13)
