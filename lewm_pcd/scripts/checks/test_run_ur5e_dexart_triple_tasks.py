from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_ur5e_dexart_triple_tasks.sh"


def text():
    return SCRIPT.read_text(encoding="utf-8")


def test_runner_has_all_four_triple_tasks_in_order():
    value = text()
    tasks = (
        "cube_triple_task1_single_pnp",
        "cube_triple_task2_triple_pnp",
        "cube_triple_task3_pnp_from_stack",
        "cube_triple_task4_cycle",
    )
    assert all(task in value for task in tasks)
    assert [value.index(task) for task in tasks] == sorted(value.index(task) for task in tasks)
    assert "for slug in \"${TASKS[@]}\"" in value


def test_runner_waits_for_prior_queue_without_competing_for_gpu():
    value = text()
    assert "DEXART_FOUR_TASKS_ACCEPTED" in value
    assert "DEXART_TRIPLE_TASKS_WAITING" in value
    assert "ur5e_dexart_triple_tasks.lock" in value
    assert "ur5e_multitask_pipeline.lock" in value
    assert value.index("flock -n 8") < value.index("trap fail EXIT")
    assert value.index("DEXART_FOUR_TASKS_ACCEPTED") < value.index("flock -w 120 9")
    assert "run_ur5e_dexart_four_tasks.sh" in value
    assert "^FAILED" in value


def test_runner_uses_existing_dexart_training_and_holdout_gate():
    value = text()
    assert "model/encoder=dexart" in value
    assert "trainer.max_epochs=10" in value
    assert "data.validation_name=\"$holdout_data\"" in value
    assert "check_ur5e_holdout_checkpoint.py" in value
    assert 'grep -q \'"status": "PASS"\'' in value
    assert "test ! -e \"$RUNTIME/checkpoints/$run\"" in value
    assert "test ! -e \"$report\"" in value
    assert "model/encoder=ms2" not in value
    assert "CEM" not in value


def test_runner_marks_failure_or_nine_task_acceptance():
    value = text()
    assert "set -euo pipefail" in value
    assert "trap fail EXIT" in value
    assert "FAILED exit_code=" in value
    assert "DEXART_NINE_TASKS_ACCEPTED" in value
    assert "mv \"$tmp\" \"$ACCEPTED\"" in value
