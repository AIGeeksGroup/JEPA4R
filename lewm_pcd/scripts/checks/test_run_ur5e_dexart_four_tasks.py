from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_ur5e_dexart_four_tasks.sh"


def text():
    return SCRIPT.read_text(encoding="utf-8")


def test_runner_has_four_tasks_and_dexart_training():
    value = text()
    for slug in (
        "cube_double_task2_double_pnp1",
        "cube_double_task3_double_pnp2",
        "cube_double_task4_swap",
        "cube_double_task5_stack",
    ):
        assert slug in value
    assert "model/encoder=dexart" in value
    assert "trainer.max_epochs=10" in value


def test_runner_is_sequential_and_uses_holdout_gate():
    value = text()
    assert "flock" in value
    assert "FULL_DATA_ACCEPTED" in value
    assert "DEXART_TASK1_ACCEPTED" in value
    assert "check_ur5e_holdout_checkpoint.py" in value
    assert "for slug in" in value
    assert "model/encoder=ms2" not in value


def test_runner_refuses_overwrite_and_has_fail_stop_status():
    value = text()
    assert "set -euo pipefail" in value
    assert "test ! -e" in value
    assert "DEXART_FOUR_TASKS_ACCEPTED" in value
    assert "CEM" not in value
