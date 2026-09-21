import numpy as np
import pytest

from envs.ur5e_cube_tasks import TASKS, build_subtask_plan, get_task


EXPECTED = [
    ("double", 1, "single_pnp"),
    ("double", 2, "double_pnp1"),
    ("double", 3, "double_pnp2"),
    ("double", 4, "swap"),
    ("double", 5, "stack"),
    ("triple", 1, "single_pnp"),
    ("triple", 2, "triple_pnp"),
    ("triple", 3, "pnp_from_stack"),
    ("triple", 4, "cycle"),
]


def test_registry_contains_exact_requested_tasks():
    assert [(task.family, task.task_id, task.name) for task in TASKS] == EXPECTED
    assert len({task.slug for task in TASKS}) == 9


@pytest.mark.parametrize("family,task_id,name", EXPECTED)
def test_lookup_is_exact(family, task_id, name):
    task = get_task(family, task_id)
    assert task.name == name
    assert task.env_type == family


def test_lookup_rejects_unsupported_triple_stack():
    with pytest.raises(KeyError, match="unsupported"):
        get_task("triple", 5)


@pytest.mark.parametrize(
    "family,task_id,expected_blocks,expected_labels",
    [
        ("double", 1, [1], ["goal"]),
        ("double", 2, [0, 1], ["goal", "goal"]),
        ("double", 3, [0, 1], ["goal", "goal"]),
        ("double", 4, [0, 1, 0], ["buffer", "goal", "goal"]),
        ("double", 5, [0, 1], ["goal", "goal"]),
        ("triple", 1, [2], ["goal"]),
        ("triple", 2, [0, 1, 2], ["goal", "goal", "goal"]),
        ("triple", 3, [2, 1, 0], ["goal", "goal", "goal"]),
        ("triple", 4, [0, 2, 1, 0], ["buffer", "goal", "goal", "goal"]),
    ],
)
def test_subtask_order_avoids_occupied_goals(
    family, task_id, expected_blocks, expected_labels
):
    task = get_task(family, task_id)
    plan = build_subtask_plan(task)
    assert [step.block_id for step in plan] == expected_blocks
    assert [step.kind for step in plan] == expected_labels
    for step in plan:
        assert step.target_xyz.shape == (3,)
        assert np.isfinite(step.target_xyz).all()


def test_swap_and_cycle_finish_at_canonical_goals():
    for family, task_id in [("double", 4), ("triple", 4)]:
        task = get_task(family, task_id)
        plan = build_subtask_plan(task)
        final_by_block = {}
        for step in plan:
            final_by_block[step.block_id] = step.target_xyz
        assert set(final_by_block) == set(range(len(task.goal_xyzs)))
        for block_id, goal in enumerate(task.goal_xyzs):
            np.testing.assert_allclose(final_by_block[block_id], goal)


def test_registry_arrays_are_not_mutable_through_callers():
    task = get_task("double", 4)
    original = task.goal_xyzs.copy()
    plan = build_subtask_plan(task)
    plan[-1].target_xyz[:] = -123
    np.testing.assert_array_equal(task.goal_xyzs, original)


def test_episode_budget_scales_with_number_of_oracle_moves():
    for task in TASKS:
        assert task.max_episode_steps == 20 + 100 * len(build_subtask_plan(task))


@pytest.mark.parametrize(
    "family,qpos_dim,qvel_dim,observation_dim",
    [("double", 28, 26, 37), ("triple", 35, 32, 46)],
)
def test_multicube_ur5e_widths_include_each_free_cube(
    family, qpos_dim, qvel_dim, observation_dim
):
    task = next(task for task in TASKS if task.family == family)
    assert task.qpos_dim == qpos_dim
    assert task.qvel_dim == qvel_dim
    assert task.control_dim == 7
    assert task.action_dim == 5
    assert task.observation_dim == observation_dim
