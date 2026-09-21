"""Configuration and persistence tests for fixed-task UR5e CEM."""

import json

import pytest

from envs.pcd_wrapper import DEFAULT_CAMERAS
from envs.ur5e_cube_tasks import TASKS, get_task


def test_world_matches_fixed_task_and_collection_camera_contract():
    from scripts.eval.evaluate_ur5e_fixed_pcd import make_world_kwargs

    kwargs = make_world_kwargs(get_task("double", 1))
    assert kwargs["env_name"] == "swm/OGBCubePCD-v0"
    assert kwargs["env_type"] == "double"
    assert kwargs["reward_task_id"] == 1
    assert kwargs["mode"] == "task"
    assert kwargs["permute_blocks"] is False
    assert kwargs["terminate_at_goal"] is True
    assert kwargs["num_envs"] == 1
    assert kwargs["image_shape"] == (224, 224)
    wrapper = kwargs["pre_wrappers"][0]
    assert wrapper.keywords["num_points"] == 1200
    assert tuple(wrapper.keywords["cameras"]) == DEFAULT_CAMERAS
    assert wrapper.keywords["cube_fraction"] == 0.25


def test_cem_matches_original_cube_protocol():
    from scripts.eval.evaluate_ur5e_fixed_pcd import make_cem_kwargs

    assert make_cem_kwargs(42) == {
        "batch_size": 1,
        "num_samples": 300,
        "var_scale": 1.0,
        "n_steps": 10,
        "topk": 30,
        "device": "cuda",
        "seed": 42,
    }


def test_state_callables_do_not_override_multi_cube_goal():
    from scripts.eval.evaluate_ur5e_fixed_pcd import make_state_callables

    callables = make_state_callables()
    assert [item["method"] for item in callables] == ["set_state"]
    assert callables[0]["args"] == {
        "qpos": {"value": "qpos"},
        "qvel": {"value": "qvel"},
    }


def test_existing_matching_episode_is_skipped_and_mismatch_rejected(tmp_path):
    from scripts.eval.evaluate_ur5e_fixed_pcd import read_completed_record

    path = tmp_path / "episode_000.json"
    identity = {"task": "cube_double_task1_single_pnp", "checkpoint_sha256": "abc", "protocol": "cem10-v1"}
    record = {**identity, "episode_index": 0, "success": False, "elapsed_seconds": 2.0}
    path.write_text(json.dumps(record), encoding="utf-8")
    assert read_completed_record(path, identity) == record
    with pytest.raises(ValueError, match="mismatch"):
        read_completed_record(path, {**identity, "checkpoint_sha256": "different"})


def test_success_summary_is_based_only_on_completed_episode_records():
    from scripts.eval.evaluate_ur5e_fixed_pcd import summarize_records

    result = summarize_records([
        {"episode_index": 0, "success": False},
        {"episode_index": 1, "success": True},
    ])
    assert result == {"completed": 2, "successes": 1, "success_rate": 0.5, "success_percent": 50.0}


def test_queue_contains_exactly_nine_canonical_tasks():
    from scripts.eval.evaluate_ur5e_fixed_pcd import TASK_SLUGS

    assert TASK_SLUGS == tuple(task.slug for task in TASKS)
    assert len(TASK_SLUGS) == 9
