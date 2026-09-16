from pathlib import Path
import subprocess
import sys

from gymnasium.spaces import Box
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import eval_maniskill  # noqa: E402
from eval_maniskill import (  # noqa: E402
    build_eval_env_kwargs,
    build_policy_info,
    clip_action,
    mask_goal_to_cube,
)


def test_build_policy_info_preserves_env_and_time_dimensions():
    xyz = np.zeros((1200, 3), dtype=np.float32)
    rgb = np.zeros((1200, 3), dtype=np.uint8)
    action = np.zeros((8,), dtype=np.float32)

    info = build_policy_info(xyz, rgb, xyz, rgb, action)

    assert info["pcd_xyz"].shape == (1, 1, 1200, 3)
    assert info["pcd_rgb"].shape == (1, 1, 1200, 3)
    assert info["goal_pcd_xyz"].shape == (1, 1, 1200, 3)
    assert info["goal_pcd_rgb"].shape == (1, 1, 1200, 3)
    assert info["action"].shape == (1, 1, 8)
    assert info["terminated"].shape == (1,)


def test_clip_action_applies_box_bounds_without_changing_shape():
    action = np.array([[2.0, -2.0, 0.5]], dtype=np.float32)
    low = np.full((1, 3), -1.0, dtype=np.float32)
    high = np.full((1, 3), 1.0, dtype=np.float32)

    clipped = clip_action(action, low, high)

    np.testing.assert_array_equal(clipped, [[1.0, -1.0, 0.5]])
    assert clipped.dtype == np.float32


def test_policy_environment_batches_maniskill_single_action_space():
    class VectorEnv:
        num_envs = 1
        single_action_space = Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        action_space = single_action_space

    class TimeLimitLikeWrapper:
        def __init__(self, env):
            self.unwrapped = env

    vector_env = VectorEnv()
    wrapped_env = TimeLimitLikeWrapper(vector_env)
    policy_env = eval_maniskill.policy_environment(wrapped_env)

    assert policy_env.num_envs == 1
    assert policy_env.single_action_space.shape == (8,)
    assert policy_env.action_space.shape == (1, 8)


def test_maniskill_eval_cli_exposes_cem_protocol():
    script = ROOT / "eval_maniskill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "--num-samples" in result.stdout
    assert "--cem-steps" in result.stdout
    assert "--eval-budget" in result.stdout


def test_maniskill_eval_cli_exposes_history_len():
    script = ROOT / "eval_maniskill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "--history-len" in result.stdout


def test_eval_env_kwargs_honor_requested_budget():
    kwargs = build_eval_env_kwargs({"max_episode_steps": 50, "control_mode": "pd_joint_pos"}, 100)

    assert kwargs["max_episode_steps"] == 100
    assert kwargs["control_mode"] == "pd_joint_pos"
    assert kwargs["sim_backend"] == "physx_cpu"
    assert kwargs["render_backend"] == "sapien_cuda"


def test_mask_goal_to_cube_preserves_current_non_cube_points():
    current_xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    current_rgb = np.arange(18, dtype=np.uint8).reshape(6, 3)
    current_groups = np.array([1, 1, 2, 3, 4, 2], dtype=np.uint8)
    goal_xyz = current_xyz + 100
    goal_rgb = current_rgb + 100
    goal_groups = current_groups.copy()

    masked_xyz, masked_rgb = mask_goal_to_cube(
        current_xyz,
        current_rgb,
        current_groups,
        goal_xyz,
        goal_rgb,
        goal_groups,
    )

    np.testing.assert_array_equal(masked_xyz[current_groups == 1], goal_xyz[goal_groups == 1])
    np.testing.assert_array_equal(masked_xyz[current_groups != 1], current_xyz[current_groups != 1])
    np.testing.assert_array_equal(masked_rgb[current_groups != 1], current_rgb[current_groups != 1])
