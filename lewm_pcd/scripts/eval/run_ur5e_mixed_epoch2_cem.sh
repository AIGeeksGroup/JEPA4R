#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-task1-2cube-pnp-local25/lewm_pcd
python=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
runtime=/autodl-fs/data/lewm/ogbench-ur5e-mixed-1cube-2cube-pcd-v2
run=ur5e_cube_mixed_1cube9900_2cube_task1_10000_pcd_dexart_10ep
checkpoint="$runtime/checkpoints/$run/weights_epoch_2.pt"
one_dataset=/root/autodl-tmp/workspace/users/lewm/hf-upload-jepa4r-robotics-data-archive/ur5e/cube_single_expert_pcd_raw.h5
two_dataset=/autodl-fs/data/lewm/ogbench-ur5e-task1-10000/full/holdout.h5
output="$runtime/cem_epoch2_local25"
current_stage=PREFLIGHT

mkdir -p "$output/logs"
exec 9>"$output/runner.lock"
flock -n 9 || { echo "epoch-2 CEM runner is already active" >&2; exit 9; }

export STABLEWM_HOME="$runtime"
export SPT_CACHE_DIR="$runtime/checkpoints/_spt_cache"
export MUJOCO_GL=egl
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

write_state() {
    "$python" - "$output/state.json" "$1" "$2" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "stage": sys.argv[2],
    "detail": sys.argv[3],
    "time": datetime.now(timezone.utc).isoformat(),
    "pid": os.getppid(),
}
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

on_exit() {
    local code=$?
    printf '%s\n' "$code" > "$output/exit_code"
    if [[ "$code" -ne 0 ]]; then
        write_state FAILED "stage=$current_stage exit_code=$code"
    fi
}
trap on_exit EXIT

run_stage() {
    local stage=$1 summary=$2 expected=$3
    shift 3
    current_stage=$stage
    write_state RUNNING "$stage"
    local code=0
    "$@" >"$output/logs/$stage.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$output/logs/$stage.exit_code"
    if [[ "$code" -ne 0 ]]; then
        return "$code"
    fi
    if [[ "$expected" -eq 0 ]]; then
        return 0
    fi
    if "$python" - "$summary" "$expected" >>"$output/logs/$stage.log" 2>&1 <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert summary["requested"] == expected, summary
assert summary["completed"] == expected, summary
print(f"summary=PASS completed={expected} successes={summary['successes']}")
PY
    then
        return 0
    else
        code=$?
        printf '%s\n' "$code" > "$output/logs/$stage.exit_code"
        return "$code"
    fi
}

cd "$repo"
test -s "$checkpoint"
test -s "$runtime/train.h5"
test -s "$runtime/validation.h5"
test -s "$one_dataset"
test -s "$two_dataset"
"$python" - "$runtime/state.json" "$runtime/stop_after_epoch_2.json" <<'PY'
import json
import sys

state, stop = (json.load(open(path)) for path in sys.argv[1:])
assert state["stage"] == stop["stage"] == "STOPPED_AFTER_EPOCH_2"
assert state["checkpoint_sha256"] == stop["checkpoint_sha256"]
PY
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
[[ -z "$gpu_pids" ]] || { echo "GPU in use by: $gpu_pids" >&2; exit 3; }

run_stage CHECKPOINT_RELOAD "$runtime/reports/${run}_epoch2_reload.json" 0 \
    "$python" "$repo/scripts/checks/check_ur5e_holdout_checkpoint.py" \
    --train-dataset "$runtime/train.h5" \
    --holdout-dataset "$runtime/validation.h5" \
    --checkpoint-run "$run" --epoch 2 \
    --report "$runtime/reports/${run}_epoch2_reload.json" \
    --batch-size 128 --num-workers 1 --max-batches 5

"$python" - "$runtime/reports/${run}_epoch2_reload.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS", report
PY

one_eval="$repo/scripts/eval/evaluate_ur5e_2cube_to_1cube_local25.py"
two_eval="$repo/scripts/eval/evaluate_ur5e_local_goal_pcd.py"
one_common=(--runtime "$runtime" --target-dataset "$one_dataset" \
    --count 50 --sample-seed 42 --policy-seed 42 --goal-offset 25 \
    --episode-start 9900 --episode-stop 10000 --eval-budget 50 \
    --output-root "$output/one_cube")
two_common=(--task cube_double_task1_single_pnp --source holdout \
    --dataset-path "$two_dataset" --count 50 --sample-seed 42 \
    --cem-seed 42 --random-seed 42 --cem-iterations 10 \
    --goal-offset 25 --eval-budget 50)
model_source=(--checkpoint "$checkpoint" --scaler-train "$runtime/train.h5")

run_stage SMOKE_ONE_CEM "$output/smoke/one_cube/all/cem/summary.json" 1 \
    "$python" "$one_eval" --policy cem --selection all \
    "${one_common[@]}" --count 1 \
    --output-root "$output/smoke/one_cube" "${model_source[@]}"
run_stage SMOKE_TWO_CEM "$output/smoke/two_cube/cem_all/summary.json" 1 \
    "$python" "$two_eval" --policy cem "${two_common[@]}" --count 1 \
    --output "$output/smoke/two_cube/cem_all" "${model_source[@]}"

for selection in all nontrivial; do
    one_summary="$output/one_cube/$selection"
    two_summary="$output/two_cube/$selection"
    two_flags=()
    if [[ "$selection" == nontrivial ]]; then
        two_flags=(--nontrivial-only)
    fi
    run_stage "ONE_RANDOM_$selection" "$one_summary/random/summary.json" 50 \
        "$python" "$one_eval" --policy random --selection "$selection" \
        "${one_common[@]}"
    run_stage "TWO_RANDOM_$selection" "$two_summary/random/summary.json" 50 \
        "$python" "$two_eval" --policy random "${two_common[@]}" \
        "${two_flags[@]}" --output "$two_summary/random"
    run_stage "ONE_CEM_$selection" "$one_summary/cem/summary.json" 50 \
        "$python" "$one_eval" --policy cem --selection "$selection" \
        "${one_common[@]}" "${model_source[@]}"
    run_stage "TWO_CEM_$selection" "$two_summary/cem/summary.json" 50 \
        "$python" "$two_eval" --policy cem "${two_common[@]}" \
        "${two_flags[@]}" --output "$two_summary/cem" "${model_source[@]}"
done

current_stage=FINAL_AUDIT
write_state RUNNING "$current_stage"
"$python" - "$output" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, checkpoint = map(Path, sys.argv[1:])
results = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
for task in ("one_cube", "two_cube"):
    for selection in ("all", "nontrivial"):
        records = {}
        for policy in ("random", "cem"):
            path = root / task / selection / policy / "summary.json"
            summary = json.loads(path.read_text())
            assert summary["requested"] == summary["completed"] == 50
            records[policy] = summary
        if task == "one_cube":
            assert records["random"]["selection_sha256"] == records["cem"]["selection_sha256"]
        else:
            random_samples = json.loads((root / task / selection / "random" / "selection.json").read_text())["samples"]
            cem_samples = json.loads((root / task / selection / "cem" / "selection.json").read_text())["samples"]
            assert random_samples == cem_samples
        results[f"{task}_{selection}"] = {
            policy: {"completed": records[policy]["completed"], "successes": records[policy]["successes"]}
            for policy in ("random", "cem")
        }
(root / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps(results, sort_keys=True))
PY

current_stage=COMPLETE
write_state COMPLETE "checkpoint=$checkpoint results=$output/results.json"
