#!/usr/bin/env bash
set -euo pipefail

P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$P/env_franka.sh"
PY=${PYTHON:-python}
MODE=${1:-run}
RUN=franka_pcd_ms2_10ep_seed3072
DATA_NAME=ogbench/franka_full/cube_franka_single_expert_pcd.rechunk20
DATA_REL="${DATA_NAME}.h5"
DATASET="$STABLEWM_HOME/datasets/$DATA_REL"
CHECKPOINT="$STABLEWM_HOME/checkpoints/$RUN/weights_epoch_10.pt"
POLICY="$RUN/weights_epoch_10.pt"
RUN_DIR="$STABLEWM_HOME/checkpoints/$RUN"
COLLECTION_STATUS="$STABLEWM_HOME/logs/franka_collection.status"
STATUS="$STABLEWM_HOME/logs/franka_autotrain_cem.status"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$STABLEWM_HOME/logs/autotrain_cem/$STAMP"
RESULT_TXT="$RUN_DIR/franka_cem100_results.txt"
RESULT_JSON="$STABLEWM_HOME/logs/eval/${RUN}_cem100.json"
LOCK="$STABLEWM_HOME/logs/franka_autotrain_cem.lock"
mkdir -p "$(dirname "$STATUS")" "$LOG_DIR"
exec 9> "$LOCK"
flock -n 9 || {
    echo "Another Franka autotrain/CEM supervisor is active" >&2
    exit 6
}

write_status() {
    local stage=$1 tmp
    tmp=$(mktemp "${STATUS}.tmp.XXXXXX")
    printf 'stage=%s time=%s pid=%s run=%s log_dir=%s\n' \
        "$stage" "$(date --iso-8601=seconds)" "$$" "$RUN" "$LOG_DIR" > "$tmp"
    mv "$tmp" "$STATUS"
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        write_status "FAILED_exit_code_$rc"
    fi
}
trap on_exit EXIT

case "$MODE" in
    run|resume) ;;
    *)
        echo "usage: $0 {run|resume}" >&2
        exit 2
        ;;
esac

while true; do
    stage=$(sed -n 's/^stage=\([^ ]*\).*/\1/p' "$COLLECTION_STATUS" 2>/dev/null || true)
    case "$stage" in
        FULL_ACCEPTED) break ;;
        FAILED*)
            echo "Collection failed: $stage" >&2
            exit 3
            ;;
        *)
            write_status "WAITING_FOR_DATA"
            sleep "${POLL_SECONDS:-60}"
            ;;
    esac
done

write_status "VALIDATING_DATA"
test -s "$DATASET"
cd "$P"
"$PY" scripts/checks/verify_franka_replay.py "$DATASET" \
    --expected-episodes 10000 --validate-only \
    --report "$LOG_DIR/data_validation.json" \
    2>&1 | tee "$LOG_DIR/data_validation.log"

if [ "$MODE" = run ] && [ -e "$RUN_DIR" ]; then
    echo "Refusing existing run: $RUN" >&2
    exit 4
fi

if [ ! -s "$CHECKPOINT" ]; then
    if [ -e "$RUN_DIR" ]; then
        echo "Refusing incomplete existing run: $RUN" >&2
        exit 4
    fi
    write_status "WAITING_FOR_GPU"
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] || \
          [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)" -gt 1024 ]; do
        sleep "${GPU_POLL_SECONDS:-60}"
    done
    write_status "TRAINING"
    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
        data=ogb_franka_pcd model=lewm_pcd model/encoder=ms2 \
        data.dataset.name="$DATA_REL" \
        subdir="$RUN" output_model_name="$RUN" \
        trainer.max_epochs=10 \
        loader.batch_size=128 loader.num_workers=6 \
        wandb.enabled=false hydra.run.dir="$LOG_DIR/hydra_train" \
        2>&1 | tee "$LOG_DIR/train.log"
fi

test -s "$CHECKPOINT"
write_status "VALIDATING_CHECKPOINT"
"$PY" scripts/checks/check_franka_checkpoint.py "$DATASET" "$RUN" \
    --epoch 10 --report "$LOG_DIR/checkpoint_reload.txt" \
    2>&1 | tee "$LOG_DIR/checkpoint_reload.log"
grep -q '^status=PASS$' "$LOG_DIR/checkpoint_reload.txt"

if [ "$MODE" = run ] && { [ -e "$RESULT_TXT" ] || [ -e "$RESULT_JSON" ]; }; then
    echo "Refusing existing CEM output" >&2
    exit 5
fi

if [ ! -s "$RESULT_JSON" ] && [ -e "$RESULT_TXT" ]; then
    echo "Refusing incomplete existing CEM result: $RESULT_TXT" >&2
    exit 5
fi

if [ ! -s "$RESULT_JSON" ]; then
    write_status "EVALUATING_CEM100"
    started=$(date +%s)
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 "$PY" eval.py \
        --config-name=cube_franka_pcd \
        policy="$POLICY" \
        eval.dataset_name="$DATA_NAME" eval.num_eval=100 \
        output.filename=franka_cem100_results.txt video=false \
        hydra.run.dir="$LOG_DIR/hydra_cem100" \
        2>&1 | tee "$LOG_DIR/cem100.log"
    elapsed=$(( $(date +%s) - started ))
    "$PY" scripts/eval/summarize_cem_result.py "$LOG_DIR/cem100.log" \
        --episodes 100 --checkpoint "$POLICY" \
        --evaluation-seconds "$elapsed" --out "$RESULT_JSON" \
        2>&1 | tee "$LOG_DIR/cem100_summary.log"
fi

"$PY" - "$RESULT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
assert record["completed"] == 100
assert record["episodes"] == 100
assert 0 <= record["successes"] <= 100
assert 0.0 <= record["success_rate"] <= 100.0
PY

write_status "COMPLETE"
