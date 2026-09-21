"""Tests for the fixed-task UR5e random-policy control."""

from envs.ur5e_cube_tasks import get_task


def test_random_control_uses_the_same_fixed_task_world():
    from scripts.eval.evaluate_ur5e_random_fixed_pcd import make_random_world_kwargs

    kwargs = make_random_world_kwargs(get_task("double", 1))
    assert kwargs["env_name"] == "swm/OGBCubePCD-v0"
    assert kwargs["env_type"] == "double"
    assert kwargs["reward_task_id"] == 1
    assert kwargs["mode"] == "task"
    assert kwargs["permute_blocks"] is False
    assert kwargs["terminate_at_goal"] is True
    assert kwargs["num_envs"] == 1


def test_random_control_does_not_load_a_model():
    from scripts.eval.evaluate_ur5e_random_fixed_pcd import make_random_policy

    policy = make_random_policy(123)
    assert policy.type == "random"
    assert policy.seed == 123
