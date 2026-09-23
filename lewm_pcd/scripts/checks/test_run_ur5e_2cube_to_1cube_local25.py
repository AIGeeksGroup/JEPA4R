"""Static contract tests for the persistent transfer-evaluation runner."""

from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "run_ur5e_2cube_to_1cube_local25.sh"
)


def test_runner_executes_the_four_approved_cells_in_order():
    script = RUNNER.read_text(encoding="utf-8")
    stages = [
        'run_stage "random_all" "random" "all"',
        'run_stage "random_nontrivial" "random" "nontrivial"',
        'run_stage "cem_all" "cem" "all"',
        'run_stage "cem_nontrivial" "cem" "nontrivial"',
    ]

    positions = [script.index(stage) for stage in stages]
    assert positions == sorted(positions)


def test_runner_pins_protocol_and_uses_only_the_corrected_output_root():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'COUNT="${COUNT:-50}"' in script
    assert 'GOAL_OFFSET="${GOAL_OFFSET:-25}"' in script
    assert 'EVAL_BUDGET="${EVAL_BUDGET:-50}"' in script
    assert 'SAMPLE_SEED="${SAMPLE_SEED:-42}"' in script
    assert 'POLICY_SEED="${POLICY_SEED:-42}"' in script
    assert "transfer_2cube_to_1cube_local25_corrected" in script
    assert "INVALID_OUTPUT_ROOT" in script
    assert 'if [[ "$(readlink -m "$OUTPUT_ROOT")" ==' in script
    assert "export MUJOCO_GL=egl" in script


def test_runner_persists_stage_logs_status_and_numeric_exit_codes():
    script = RUNNER.read_text(encoding="utf-8")

    assert '"$stage_dir/run.log"' in script
    assert '"$stage_dir/exit_code"' in script
    assert '"$stage_dir/status"' in script
    assert 'write_runner_state "$stage" "RUNNING"' in script
    assert 'write_runner_state "$stage" "FAILED" "$rc"' in script
    assert 'write_runner_state "$stage" "PASS" 0' in script
    assert 'write_runner_state "complete" "PASS" 0' in script


def test_runner_forwards_every_fixed_evaluator_argument():
    script = RUNNER.read_text(encoding="utf-8")

    for argument in (
        '--count "$COUNT"',
        '--goal-offset "$GOAL_OFFSET"',
        '--eval-budget "$EVAL_BUDGET"',
        '--sample-seed "$SAMPLE_SEED"',
        '--policy-seed "$POLICY_SEED"',
        '--runtime "$RUNTIME"',
        '--target-dataset "$TARGET_DATASET"',
        '--output-root "$OUTPUT_ROOT"',
    ):
        assert argument in script
