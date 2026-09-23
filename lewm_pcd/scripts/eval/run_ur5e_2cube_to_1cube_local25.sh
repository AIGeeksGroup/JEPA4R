#!/usr/bin/env bash
set -uo pipefail

export MUJOCO_GL=egl

REPO="${REPO:-/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-task1-2cube-pnp-local25/lewm_pcd}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python}"
EVALUATOR="$REPO/scripts/eval/evaluate_ur5e_2cube_to_1cube_local25.py"
RUNTIME="${RUNTIME:-/autodl-fs/data/lewm/ogbench-ur5e-task1-10000}"
TARGET_DATASET="${TARGET_DATASET:-/root/autodl-tmp/workspace/users/lewm/hf-upload-jepa4r-robotics-data-archive/ur5e/cube_single_expert_pcd_raw.h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/autodl-fs/data/lewm/ogbench-ur5e-task1-10000/transfer_2cube_to_1cube_local25_corrected}"
INVALID_OUTPUT_ROOT="/autodl-fs/data/lewm/ogbench-ur5e-task1-10000/transfer_2cube_to_1cube_local25"
COUNT="${COUNT:-50}"
GOAL_OFFSET="${GOAL_OFFSET:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
POLICY_SEED="${POLICY_SEED:-42}"

if [[ "$(readlink -m "$OUTPUT_ROOT")" == "$(readlink -m "$INVALID_OUTPUT_ROOT")" ]]; then
    echo "refusing historical invalid output root: $OUTPUT_ROOT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "python is not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -f "$EVALUATOR" ]]; then
    echo "evaluator is missing: $EVALUATOR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO" || exit 2

atomic_line() {
    local path="$1"
    local value="$2"
    local temporary="${path}.tmp.$$"
    printf '%s\n' "$value" > "$temporary"
    mv -f "$temporary" "$path"
}

write_runner_state() {
    local stage="$1"
    local status="$2"
    local exit_code="${3:-null}"
    local path="$OUTPUT_ROOT/runner_state.json"
    local temporary="${path}.tmp.$$"
    printf '{"stage":"%s","status":"%s","exit_code":%s,"updated_at":"%s"}\n' \
        "$stage" "$status" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$temporary"
    mv -f "$temporary" "$path"
}

run_stage() {
    local stage="$1"
    local policy="$2"
    local selection="$3"
    local stage_dir="$OUTPUT_ROOT/$selection/$policy"
    local log="$stage_dir/run.log"
    local rc
    mkdir -p "$stage_dir"
    atomic_line "$stage_dir/status" "RUNNING"
    write_runner_state "$stage" "RUNNING"

    local command=(
        "$PYTHON" "$EVALUATOR"
        --policy "$policy"
        --selection "$selection"
        --count "$COUNT"
        --goal-offset "$GOAL_OFFSET"
        --eval-budget "$EVAL_BUDGET"
        --sample-seed "$SAMPLE_SEED"
        --policy-seed "$POLICY_SEED"
        --runtime "$RUNTIME"
        --target-dataset "$TARGET_DATASET"
        --output-root "$OUTPUT_ROOT"
    )
    {
        printf '\n[%s] stage=%s command=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage"
        printf ' %q' "${command[@]}"
        printf '\n'
    } >> "$log"
    "${command[@]}" >> "$log" 2>&1
    rc=$?
    atomic_line "$stage_dir/exit_code" "$rc"
    if [[ "$rc" -ne 0 ]]; then
        atomic_line "$stage_dir/status" "FAILED"
        write_runner_state "$stage" "FAILED" "$rc"
        return "$rc"
    fi
    atomic_line "$stage_dir/status" "PASS"
    write_runner_state "$stage" "PASS" 0
}

run_stage "random_all" "random" "all" || exit $?
run_stage "random_nontrivial" "random" "nontrivial" || exit $?
run_stage "cem_all" "cem" "all" || exit $?
run_stage "cem_nontrivial" "cem" "nontrivial" || exit $?

write_runner_state "complete" "PASS" 0
atomic_line "$OUTPUT_ROOT/runner_status" "PASS"
