from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

import envs  # noqa: F401  # registers local environments


STATE_ENV_ID = "swm/OGBCubeFranka-v0"
PCD_ENV_ID = "swm/OGBCubeFrankaPCD-v0"


def _make(env_id: str):
    env = gym.make(
        env_id,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
    )
    np.random.seed(7)
    env.reset(seed=7)
    return env


def _names(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, kind, index)) is not None
    }


@pytest.mark.parametrize("env_id", [STATE_ENV_ID, PCD_ENV_ID])
def test_franka_variants_keep_the_ogbench_5d_action(env_id: str) -> None:
    env = _make(env_id)
    try:
        assert env.action_space.shape == (5,)
        np.testing.assert_array_equal(env.action_space.low, -np.ones(5))
        np.testing.assert_array_equal(env.action_space.high, np.ones(5))
        assert env.unwrapped._model.opt.timestep == pytest.approx(0.002)
        assert env.unwrapped._control_timestep == pytest.approx(0.05)
    finally:
        env.close()


def test_state_variant_is_the_real_seven_dof_franka_panda() -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        model = robot._model
        info = robot.compute_ob_info()

        assert info["proprio/joint_pos"].shape == (7,)
        assert info["proprio/joint_vel"].shape == (7,)

        joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        actuator_names = _names(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu
        )
        assert {f"franka/joint{index}" for index in range(1, 8)} <= joint_names
        assert {"franka/finger_joint1", "franka/finger_joint2"} <= joint_names
        assert {"franka/link0", "franka/hand", "franka/right_finger"} <= body_names
        assert {f"franka/actuator{index}" for index in range(1, 9)} <= actuator_names

        every_robot_name = joint_names | body_names | actuator_names
        assert not any("ur5e" in name.lower() for name in every_robot_name)
        assert not any("robotiq" in name.lower() for name in every_robot_name)
    finally:
        env.close()


def test_pcd_variant_adds_the_existing_multiview_camera_contract() -> None:
    env = _make(PCD_ENV_ID)
    try:
        model = env.unwrapped._model
        camera_names = _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        assert {
            "front_pixels",
            "pcd_front_left",
            "pcd_back_left",
            "pcd_back_right",
            "pcd_front_right",
            "pcd_top",
        } <= camera_names
    finally:
        env.close()


def test_task_mode_reset_and_step_are_finite() -> None:
    env = gym.make(
        STATE_ENV_ID,
        env_type="single",
        ob_type="states",
        mode="task",
        reward_task_id=2,
        terminate_at_goal=False,
        visualize_info=False,
    )
    try:
        observation, info = env.reset(seed=11)
        assert np.isfinite(observation).all()
        assert np.isfinite(info["qpos"]).all()
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(5)
        )
        assert np.isfinite(observation).all()
        assert np.isfinite(info["qpos"]).all()
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_scene_and_ik_models_use_the_same_pinch_frame() -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        robot._data.qpos[robot._arm_joint_ids] = robot._home_qpos
        mujoco.mj_forward(robot._model, robot._data)

        robot._ik._data.qpos[:] = robot._home_qpos
        robot._ik._forward_kinematics()
        ik_site_id = robot._ik._site_ids[0]
        np.testing.assert_allclose(
            robot._data.site_xpos[robot._pinch_site_id],
            robot._ik._data.site_xpos[ik_site_id],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            robot._data.site_xmat[robot._pinch_site_id],
            robot._ik._data.site_xmat[ik_site_id],
            atol=1e-8,
        )
    finally:
        env.close()


def test_control_preserves_xyz_yaw_and_ogbench_gripper_signs() -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        model = robot._model
        data = robot._data

        current_pos = data.site_xpos[robot._pinch_site_id].copy()
        current_yaw = robot._pinch_yaw()
        robot.set_control(np.array([1.0, 0.0, 0.0, 1.0, 0.0]))
        target_pos = robot._target_effector_pose.translation()
        target_yaw = robot._target_effector_pose.rotation().compute_yaw_radians()
        assert target_pos[0] - current_pos[0] == pytest.approx(0.05, abs=1e-6)
        expected_yaw = np.clip(current_yaw + 0.3, -np.pi, np.pi)
        assert target_yaw == pytest.approx(expected_yaw, abs=1e-6)

        for joint_name in ["franka/finger_joint1", "franka/finger_joint2"]:
            qpos_address = model.joint(joint_name).qposadr[0]
            data.qpos[qpos_address] = 0.03
        mujoco.mj_forward(model, data)

        robot.set_control(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        assert data.ctrl[robot._gripper_actuator_ids[0]] == pytest.approx(12.75)
        robot.set_control(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))
        assert data.ctrl[robot._gripper_actuator_ids[0]] == pytest.approx(255.0)
    finally:
        env.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_signed_x_pulse_moves_the_physical_panda_in_that_direction(
    sign: float,
) -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        before = robot._data.site_xpos[robot._pinch_site_id, 0].copy()
        env.step(np.array([sign, 0.0, 0.0, 0.0, 0.0]))
        after = robot._data.site_xpos[robot._pinch_site_id, 0]
        assert sign * (after - before) > 1e-5
    finally:
        env.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_positive_gripper_action_closes_and_negative_action_opens(
    sign: float,
) -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        robot._data.qpos[robot._gripper_qpos_ids] = 0.03
        robot._data.qvel[robot._gripper_dof_ids] = 0.0
        mujoco.mj_forward(robot._model, robot._data)
        before = np.mean(robot._data.qpos[robot._gripper_qpos_ids])
        env.step(np.array([0.0, 0.0, 0.0, 0.0, sign]))
        after = np.mean(robot._data.qpos[robot._gripper_qpos_ids])
        assert sign * (before - after) > 1e-5
    finally:
        env.close()


def test_reset_starts_with_open_panda_gripper_and_normalized_closure() -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        np.testing.assert_allclose(robot._data.qpos[robot._gripper_qpos_ids], 0.04)
        assert robot.compute_ob_info()["proprio/gripper_opening"][0] == pytest.approx(0.0)
    finally:
        env.close()


def test_reset_clips_franka_ik_to_joint_ranges() -> None:
    env = gym.make(
        STATE_ENV_ID,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
    )
    try:
        np.random.seed(7000)
        env.reset(seed=7000)
        robot = env.unwrapped
        values = robot._data.qpos[robot._arm_joint_ids]
        ranges = robot._model.jnt_range[robot._arm_joint_ids]
        assert np.all(values >= ranges[:, 0] - 1e-10)
        assert np.all(values <= ranges[:, 1] + 1e-10)
        oracle = CubePlanOracle(env=env, noise=0.0, noise_smoothing=0.5)
        observation, info = env.reset(seed=7005)
        oracle.reset(observation, info)
        env.step(oracle.select_action(observation, info))
        values = env.unwrapped._data.qpos[env.unwrapped._arm_joint_ids]
        assert np.all(values >= ranges[:, 0] - 1e-3)
        assert np.all(values <= ranges[:, 1] + 1e-3)
    finally:
        env.close()


def test_franka_finger_collision_surface_uses_gripping_friction() -> None:
    env = _make(STATE_ENV_ID)
    try:
        robot = env.unwrapped
        pad_ids = []
        other_collision_ids = []
        for geom_id in range(robot._model.ngeom):
            body_name = robot._model.body(
                int(robot._model.geom_bodyid[geom_id])
            ).name or ""
            if body_name in {"franka/left_finger", "franka/right_finger"}:
                if robot._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                    pad_ids.append(geom_id)
                elif robot._model.geom_contype[geom_id] != 0:
                    other_collision_ids.append(geom_id)
        assert len(pad_ids) == 10
        np.testing.assert_allclose(robot._model.geom_friction[pad_ids, 0], 2.0)
        np.testing.assert_array_equal(robot._model.geom_priority[pad_ids], 1)
        np.testing.assert_allclose(
            robot._model.geom_solref[pad_ids],
            np.tile([0.004, 1.0], (len(pad_ids), 1)),
        )
        np.testing.assert_allclose(robot._model.geom_friction[other_collision_ids, 0], 1.0)
        force_ranges = robot._model.actuator_forcerange[
            robot._gripper_actuator_ids
        ]
        np.testing.assert_allclose(force_ranges, [[-70.0, 70.0]])
    finally:
        env.close()


def test_first_oracle_targets_stay_within_panda_joint_ranges() -> None:
    for seed in (7000, 7002, 7003):
        np.random.seed(seed)
        env = _make(STATE_ENV_ID)
        try:
            observation, info = env.reset(seed=seed)
            oracle = CubePlanOracle(env=env, noise=0.0, noise_smoothing=0.5)
            oracle.reset(observation, info)
            env.step(oracle.select_action(observation, info))
            robot = env.unwrapped
            values = robot._data.qpos[robot._arm_joint_ids]
            ranges = robot._model.jnt_range[robot._arm_joint_ids]
            assert np.all(values >= ranges[:, 0] - 1e-3)
            assert np.all(values <= ranges[:, 1] + 1e-3)
        finally:
            env.close()
