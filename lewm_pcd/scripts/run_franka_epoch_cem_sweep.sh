#!/usr/bin/env bash
set -euo pipefail

P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$P/env_franka.sh"

PY=${PYTHON:-python}
RUN=franka_pcd_ms2_10ep_seed3072
DATA_NAME=ogbench/franka_full/cube_franka_single_expert_pcd.rechunk20
STATUS="$STABLEWM_HOME/logs/franka_epoch_cem_sweep.status"
LOCK="$STABLEWM_HOME/logs/franka_epoch_cem_sweep.lock"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="$STABLEWM_HOME/logs/eval/${RUN}_epochs01-09_cem100_$STAMP"
mkdir -p "$LOG_ROOT" "$(dirname "$STATUS")"

exec 9> "$LOCK"
flock -n 9 || {
    echo "Another Franka epoch CEM sweep is active" >&2
    exit 6
}

write_status() {
    local stage=$1 epoch=${2:-none} tmp
    tmp=$(mktemp "${STATUS}.tmp.XXXXXX")
    printf 'stage=%s time=%s pid=%s epoch=%s log_root=%s\n' \
        "$stage" "$(date --iso-8601=seconds)" "$$" "$epoch" "$LOG_ROOT" > "$tmp"
    mv "$tmp" "$STATUS"
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        write_status "FAILED_exit_code_$rc" "${epoch:-none}"
    fi
}
trap on_exit EXIT

cd "$P"
for epoch in $(seq 1 9); do
    checkpoint="$STABLEWM_HOME/checkpoints/$RUN/weights_epoch_${epoch}.pt"
    test -s "$checkpoint"

    epoch_dir=$(printf '%s/epoch_%02d' "$LOG_ROOT" "$epoch")
    mkdir -p "$epoch_dir"
    result_json="$epoch_dir/cem100.json"
    result_name=$(printf 'franka_cem100_epoch_%02d_%s_results.txt' "$epoch" "$STAMP")

    write_status EVALUATING "$epoch"
    started=$(date +%s)
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
        "$PY" eval.py \
        --config-name=cube_franka_pcd \
        policy="$RUN/weights_epoch_${epoch}.pt" \
        eval.dataset_name="$DATA_NAME" eval.num_eval=100 \
        output.filename="$result_name" video=false \
        hydra.run.dir="$epoch_dir/hydra" \
        2>&1 | tee "$epoch_dir/cem100.log"
    elapsed=$(( $(date +%s) - started ))

    "$PY" scripts/eval/summarize_cem_result.py "$epoch_dir/cem100.log" \
        --episodes 100 --checkpoint "$RUN/weights_epoch_${epoch}.pt" \
        --evaluation-seconds "$elapsed" --out "$result_json"
    "$PY" - "$result_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
assert record["completed"] == 100
assert record["episodes"] == 100
assert 0 <= record["successes"] <= 100
assert 0.0 <= record["success_rate"] <= 100.0
PY
    write_status EPOCH_COMPLETE "$epoch"
done

"$PY" - "$LOG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("epoch_*/cem100.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    record["epoch"] = int(path.parent.name.split("_")[-1])
    records.append(record)
assert [record["epoch"] for record in records] == list(range(1, 10))
(root / "sweep_summary.json").write_text(
    json.dumps(records, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

write_status COMPLETE 9
