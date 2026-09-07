"""Retargeting must keep the native reset workspace while the cube is lifted."""
from types import SimpleNamespace

import numpy as np
import pytest
from sapien.core import Pose
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv

from ms2_lewm.environment import ManiSkillPointCloudEnv


def make_adapter(height, seed):
    site = SimpleNamespace()
    site.set_pose = lambda pose: setattr(site, "pose", pose)
    simulator = SimpleNamespace(
        obj=SimpleNamespace(pose=Pose([0.01, -0.03, height])),
        goal_site=site, cube_half_size=np.asarray([0.02] * 3, dtype=np.float32),
        min_goal_dist=PickCubeEnv.min_goal_dist, _episode_rng=np.random.RandomState(seed),
    )
    adapter = ManiSkillPointCloudEnv.__new__(ManiSkillPointCloudEnv)
    adapter.rl_env = SimpleNamespace(unwrapped=simulator)
    adapter._last_obs = {"pointcloud": object()}
    adapter.oracle_info = lambda: {"privileged/target_block_pos": simulator.goal_pos.copy()}
    return adapter, simulator


def test_table_cube_matches_native_goal_sampling_and_rng_exactly():
    for seed in range(100):
        adapted, simulator = make_adapter(0.02, seed)
        _, reference = make_adapter(0.02, seed)
        PickCubeEnv._initialize_task(reference)
        adapted.set_new_target()
        np.testing.assert_array_equal(simulator.goal_pos, reference.goal_pos)
        actual_rng, expected_rng = simulator._episode_rng.get_state(), reference._episode_rng.get_state()
        np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
        assert actual_rng[2:] == expected_rng[2:]


@pytest.mark.parametrize("height", [0.1, 0.3, 0.5, 0.7])
def test_lifted_cube_retargets_without_height_accumulation_or_object_motion(height):
    adapter, simulator = make_adapter(height, 3072)
    cube_pose = simulator.obj.pose.to_transformation_matrix().copy()
    table_z = float(simulator.cube_half_size[2])
    goals = []
    for _ in range(200):
        observation, info = adapter.set_new_target()
        goal = simulator.goal_pos
        assert np.all(np.abs(goal[:2]) <= 0.1)
        assert table_z <= goal[2] <= table_z + 0.5
        assert np.linalg.norm(goal - simulator.obj.pose.p) > simulator.min_goal_dist
        np.testing.assert_array_equal(simulator.obj.pose.to_transformation_matrix(), cube_pose)
        np.testing.assert_allclose(simulator.goal_site.pose.p, goal, atol=3e-8, rtol=0)
        np.testing.assert_array_equal(info["privileged/target_block_pos"], goal)
        assert observation is adapter._last_obs
        goals.append(goal.copy())
    assert np.ptp(np.asarray(goals)[:, 2]) > 0.4


def test_native_reset_hook_reproduces_the_lifted_cube_bug():
    _, simulator = make_adapter(0.3, 3072)
    heights = []
    for _ in range(100):
        PickCubeEnv._initialize_task(simulator)
        heights.append(simulator.goal_pos[2])
    assert max(heights) > 0.7


def test_retarget_preserves_native_minimum_distance_rejection():
    adapter, simulator = make_adapter(0.3, 3072)
    draws = iter([simulator.obj.pose.p[:2], 0.28, np.array([-0.1, 0.1]), 0.1])
    simulator._episode_rng = SimpleNamespace(uniform=lambda *args: next(draws))
    adapter.set_new_target()
    np.testing.assert_allclose(simulator.goal_pos, [-0.1, 0.1, 0.12], atol=1e-8, rtol=0)
