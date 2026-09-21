"""Protocol tests for the mainline-aligned UR5e local-goal diagnostic."""

from envs.ur5e_cube_tasks import get_task


def test_local_goal_dataset_keys_include_each_cube_pose():
    from scripts.eval.evaluate_ur5e_local_goal_pcd import local_eval_keys

    assert local_eval_keys(get_task("double", 1)) == [
        "qpos",
        "qvel",
        "pcd_xyz",
        "pcd_rgb",
        "action",
        "privileged/block_0_pos",
        "privileged/block_0_quat",
        "privileged/block_1_pos",
        "privileged/block_1_quat",
    ]


def test_local_goal_callables_restore_state_and_relabel_both_cube_targets():
    from scripts.eval.evaluate_ur5e_local_goal_pcd import make_local_goal_callables

    callables = make_local_goal_callables(get_task("double", 1))
    assert callables[0] == {
        "method": "set_state",
        "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
    }
    assert callables[1:] == [
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged/block_0_pos"},
                "target_quat": {"value": "goal_privileged/block_0_quat"},
            },
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 1, "in_dataset": False},
                "target_pos": {"value": "goal_privileged/block_1_pos"},
                "target_quat": {"value": "goal_privileged/block_1_quat"},
            },
        },
    ]


def test_local_summary_reports_immediate_and_nontrivial_subsets():
    from scripts.eval.evaluate_ur5e_local_goal_pcd import summarize_local_records

    records = [
        {"sample_index": 0, "success": True, "start_already_success": True, "nontrivial": False},
        {"sample_index": 1, "success": False, "start_already_success": False, "nontrivial": True},
        {"sample_index": 2, "success": True, "start_already_success": False, "nontrivial": True},
    ]
    assert summarize_local_records(records) == {
        "completed": 3,
        "successes": 2,
        "success_rate": 2 / 3,
        "success_percent": 200 / 3,
        "start_already_success": {"count": 1, "successes": 1, "success_rate": 1.0},
        "nontrivial": {"count": 2, "successes": 1, "success_rate": 0.5},
    }


def test_local_cem_iterations_are_explicit_and_mainline_default_is_30():
    from scripts.eval.evaluate_ur5e_local_goal_pcd import make_local_cem_kwargs

    kwargs = make_local_cem_kwargs(seed=42, iterations=30)
    assert kwargs == {
        "batch_size": 1,
        "num_samples": 300,
        "var_scale": 1.0,
        "n_steps": 30,
        "topk": 30,
        "device": "cuda",
        "seed": 42,
    }


def test_local_random_policy_is_seeded_without_a_model():
    from scripts.eval.evaluate_ur5e_local_goal_pcd import make_local_policy

    policy = make_local_policy("random", seed=123)

    assert policy.type == "random"
    assert policy.seed == 123
