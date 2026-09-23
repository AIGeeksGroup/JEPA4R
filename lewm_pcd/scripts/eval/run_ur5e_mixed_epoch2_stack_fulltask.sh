#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-task1-2cube-pnp-local25/lewm_pcd
python=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
mixed=/autodl-fs/data/lewm/ogbench-ur5e-mixed-1cube-2cube-pcd-v2
source_runtime=/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask
run=ur5e_cube_mixed_1cube9900_2cube_task1_10000_pcd_dexart_10ep
checkpoint="$mixed/checkpoints/$run/weights_epoch_2.pt"
output="$mixed/stack_task5_epoch2_fulltask_v1"
task=cube_double_task5_stack
current_stage=PREFLIGHT

mkdir -p "$output/logs" "$output/formal"
exec 9>"$output/runner.lock"
flock -n 9 || { echo "stack CEM runner is already active" >&2; exit 9; }

export STABLEWM_HOME="$mixed"
export SPT_CACHE_DIR="$mixed/checkpoints/_spt_cache"
export MUJOCO_GL=egl
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

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
    local stage=$1
    shift
    current_stage=$stage
    write_state RUNNING "$stage"
    local code=0
    "$@" >"$output/logs/$stage.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$output/logs/$stage.exit_code"
    if [[ "$code" -ne 0 ]]; then
        return "$code"
    fi
    if "$python" - "$output/formal/${stage,,}/summary.json" >>"$output/logs/$stage.log" 2>&1 <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1]))
assert summary["requested"] == summary["completed"] == 50, summary
print(f"summary=PASS completed=50 successes={summary['successes']}")
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
test -s "$mixed/train.h5"
test -s "$source_runtime/full/$task/holdout.h5"
test "$(cat "$output/smoke/status")" = PASS
test "$(cat "$output/smoke/exit_code")" = 0
"$python" - "$mixed/reports/${run}_epoch2_reload.json" "$output/smoke/cem/episode_000.json" "$checkpoint" <<'PY'
import hashlib
import json
import sys

reload, smoke = (json.load(open(path)) for path in sys.argv[1:3])
checksum = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
assert reload["status"] == "PASS"
assert reload["checkpoint_sha256"] == smoke["checkpoint_sha256"] == checksum
assert smoke["preflight"] == {
    "qpos_max_abs_error": 0.0,
    "qvel_max_abs_error": 0.0,
    "live_pcd_points": 1200,
}
PY
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
[[ -z "$gpu_pids" ]] || { echo "GPU in use by: $gpu_pids" >&2; exit 3; }

run_stage RANDOM \
    "$python" scripts/eval/evaluate_ur5e_random_fixed_pcd.py \
    --task "$task" --episodes 50 --runtime "$source_runtime" \
    --output "$output/formal/random" --seed 42000

run_stage CEM \
    "$python" scripts/eval/evaluate_ur5e_fixed_pcd.py \
    --task "$task" --episodes 50 --runtime "$source_runtime" \
    --checkpoint "$checkpoint" --scaler-train "$mixed/train.h5" \
    --output "$output/formal/cem" --seed 42000

current_stage=FINAL_AUDIT
write_state RUNNING "$current_stage"
"$python" - "$output" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, checkpoint = map(Path, sys.argv[1:])
results = {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "task": "cube_double_task5_stack",
    "evaluation": "native_full_task",
    "holdout_episodes": 50,
}
groups = {}
for policy in ("random", "cem"):
    folder = root / "formal" / policy
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob("episode_*.json"))]
    assert summary["requested"] == summary["completed"] == len(rows) == 50
    assert [row["episode_index"] for row in rows] == list(range(50))
    assert [row["seed"] for row in rows] == list(range(42000, 42050))
    assert sum(row["success"] for row in rows) == summary["successes"]
    if policy == "cem":
        assert all(row["checkpoint_sha256"] == results["checkpoint_sha256"] for row in rows)
        assert all(row["preflight"]["qpos_max_abs_error"] == 0.0 for row in rows)
        assert all(row["preflight"]["qvel_max_abs_error"] == 0.0 for row in rows)
        assert all(row["preflight"]["live_pcd_points"] == 1200 for row in rows)
    groups[policy] = {"completed": summary["completed"], "successes": summary["successes"]}
results["groups"] = groups
(root / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps(results, sort_keys=True))
PY

current_stage=COMPLETE
write_state COMPLETE "results=$output/results.json"
