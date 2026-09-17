from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import h5py
import mujoco
import numpy as np
import pytest

import envs  # noqa: F401
from scripts.checks.verify_franka import (
    assert_rollout_equal,
    gripper_closure,
    joint_limits_ok,
    robot_geom_mask,
    verify_hash_manifest,
)
from scripts.checks.verify_franka_replay import (
    point_cloud_frame_stats,
    validate_action_rows,
    validate_dataset,
    validate_episode_layout,
)


REPO = Path(__file__).resolve().parents[2]
ENV_ID = "swm/OGBCubeFranka-v0"


def _make():
    env = gym.make(
        ENV_ID,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
    )
    env.reset(seed=31)
    return env


def test_hash_manifest_and_robot_geom_mask() -> None:
    asset_dir = REPO / "envs" / "descriptions" / "franka_emika_panda"
    assert verify_hash_manifest(asset_dir)
    env = _make()
    try:
        mask = robot_geom_mask(env.unwrapped._model)
        assert mask.shape == (env.unwrapped._model.ngeom,)
        assert mask.any()
        assert not mask[0]  # world/floor geom is not a robot geom
    finally:
        env.close()


def test_joint_limit_and_gripper_closure_helpers() -> None:
    env = _make()
    try:
        robot = env.unwrapped
        values = robot._data.qpos[robot._arm_joint_ids]
        ranges = robot._model.jnt_range[robot._arm_joint_ids]
        assert joint_limits_ok(values, ranges)
        robot._data.qpos[robot._gripper_qpos_ids] = [0.0, 0.0]
        assert gripper_closure(robot) == pytest.approx(1.0)
        robot._data.qpos[robot._gripper_qpos_ids] = [0.04, 0.04]
        assert gripper_closure(robot) == pytest.approx(0.0)
    finally:
        env.close()


def test_replay_comparison_accepts_identical_states_and_rejects_1e8_mismatch() -> None:
    reference = [np.zeros(4), np.ones(4)]
    actual = [item.copy() for item in reference]
    assert_rollout_equal(reference, actual, atol=1e-10)
    actual[1][0] += 1e-8
    with pytest.raises(AssertionError, match="1.0e-10"):
        assert_rollout_equal(reference, actual, atol=1e-10)


def test_short_deterministic_franka_replay_has_no_state_drift() -> None:
    env = _make()
    try:
        robot = env.unwrapped
        model = robot._model
        state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial_state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(
            model,
            robot._data,
            initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        initial_ctrl = robot._data.ctrl.copy()
        initial_mocap_pos = robot._data.mocap_pos.copy()
        initial_mocap_quat = robot._data.mocap_quat.copy()
        actions = np.random.default_rng(5).uniform(-0.4, 0.4, size=(4, 5))

        def run():
            values = []
            for action in actions:
                _, _, _, _, info = env.step(action)
                values.append(
                    np.concatenate(
                        [
                            robot._data.qpos.copy(),
                            robot._data.qvel.copy(),
                            info["privileged/block_0_pos"],
                        ]
                    )
                )
            return values

        first = run()
        mujoco.mj_setState(
            model,
            robot._data,
            initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        robot._data.ctrl[:] = initial_ctrl
        robot._data.mocap_pos[:] = initial_mocap_pos
        robot._data.mocap_quat[:] = initial_mocap_quat
        robot._prev_qpos = robot._data.qpos.copy()
        robot._prev_qvel = robot._data.qvel.copy()
        mujoco.mj_forward(model, robot._data)
        second = run()
        assert_rollout_equal(first, second, atol=1e-10)
    finally:
        env.close()


def test_episode_layout_requires_contiguous_offsets() -> None:
    validate_episode_layout(
        ep_len=np.array([3, 2], dtype=np.int32),
        ep_offset=np.array([0, 3], dtype=np.int64),
        total_rows=5,
        expected_episodes=2,
    )
    with pytest.raises(AssertionError, match="contiguous"):
        validate_episode_layout(
            ep_len=np.array([3, 2], dtype=np.int32),
            ep_offset=np.array([0, 4], dtype=np.int64),
            total_rows=5,
            expected_episodes=2,
        )


def test_only_terminal_action_rows_may_be_nan() -> None:
    actions = np.zeros((5, 5), dtype=np.float32)
    actions[[2, 4]] = np.nan
    validate_action_rows(
        actions,
        ep_len=np.array([3, 2]),
        ep_offset=np.array([0, 3]),
    )
    actions[1] = np.nan
    with pytest.raises(AssertionError, match="non-terminal"):
        validate_action_rows(
            actions,
            ep_len=np.array([3, 2]),
            ep_offset=np.array([0, 3]),
        )


def test_point_cloud_stats_count_cube_and_duplicate_points() -> None:
    cube_center = np.array([0.4, 0.0, 0.02])
    cube = cube_center + np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
    )
    robot = np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]])
    stats = point_cloud_frame_stats(np.concatenate([cube, robot]), cube_center)
    assert stats["cube_points"] == 3
    assert stats["robot_points"] == 2
    assert stats["repeated_points"] == 1
    assert stats["unique_cube_points"] == 3


def test_dataset_validation_rejects_non_franka_state_widths(tmp_path: Path) -> None:
    path = tmp_path / "wrong_robot.h5"
    pixels = np.arange(2 * 224 * 224 * 3, dtype=np.uint32)
    pixels = (pixels % 251).astype(np.uint8).reshape(2, 224, 224, 3)
    with h5py.File(path, "w") as handle:
        handle["ep_len"] = np.array([2], dtype=np.int32)
        handle["ep_offset"] = np.array([0], dtype=np.int64)
        actions = np.zeros((2, 5), dtype=np.float32)
        actions[-1] = np.nan
        handle["action"] = actions
        handle["observation"] = np.zeros((2, 28), dtype=np.float64)
        handle["target"] = np.zeros((2, 28), dtype=np.float64)
        handle["qpos"] = np.zeros((2, 21), dtype=np.float64)
        handle["qvel"] = np.zeros((2, 20), dtype=np.float64)
        handle["prev_qpos"] = np.zeros((2, 21), dtype=np.float64)
        handle["prev_qvel"] = np.zeros((2, 20), dtype=np.float64)
        handle["control"] = np.zeros((2, 7), dtype=np.float64)
        handle["pixels"] = pixels

    with pytest.raises(AssertionError, match="does not match Franka"):
        validate_dataset(path, expected_episodes=1)
