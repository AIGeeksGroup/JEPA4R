#!/usr/bin/env bash
set -euo pipefail

P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$P/env_franka.sh"
PY=${PYTHON:-python}
STAGE="${1:-verify}"
STATUS="$STABLEWM_HOME/logs/franka_pipeline.status"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$STABLEWM_HOME/logs/pipeline/$STAMP"
mkdir -p "$(dirname "$STATUS")" "$STABLEWM_HOME/logs/checks" "$LOG_DIR"

write_status() {
    local value=$1 tmp
    tmp=$(mktemp "${STATUS}.tmp.XXXXXX")
    printf '%s time=%s pid=%s log_dir=%s\n' \
        "$value" "$(date --iso-8601=seconds)" "$$" "$LOG_DIR" > "$tmp"
    mv "$tmp" "$STATUS"
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        write_status "FAILED exit_code=$rc"
    fi
}
trap on_exit EXIT

refuse_existing() {
    if [ -e "$1" ]; then
        echo "Refusing to overwrite existing output: $1" >&2
        exit 2
    fi
}

collect_pcd() {
    local episodes=${NUM_TRAJ:-10}
    local out="$STABLEWM_HOME/datasets/ogbench/smoke_franka_pcd_${episodes}.h5"
    refuse_existing "$out"
    write_status "COLLECTING_PCD"
    cd "$P"
    "$PY" scripts/data/collect_cube_pcd.py \
        --config-name=ogb_franka_pcd \
        num_traj="$episodes" \
        dataset_name="ogbench/smoke_franka_pcd_${episodes}.h5" \
        hydra.run.dir="$LOG_DIR/hydra_collect_pcd" \
        2>&1 | tee "$LOG_DIR/collect_pcd.log"
    test -s "$out"
    write_status "PCD_COLLECTED"
    printf '%s\n' "$out"
}

collect_pix() {
    local episodes=${NUM_TRAJ:-10}
    local out="$STABLEWM_HOME/datasets/ogbench/smoke_franka_pix_${episodes}.h5"
    refuse_existing "$out"
    write_status "COLLECTING_PIX"
    cd "$P"
    "$PY" scripts/data/collect_cube_pix.py \
        --config-name=ogb_franka_pix \
        num_traj="$episodes" \
        dataset_name="ogbench/smoke_franka_pix_${episodes}.h5" \
        hydra.run.dir="$LOG_DIR/hydra_collect_pix" \
        2>&1 | tee "$LOG_DIR/collect_pix.log"
    test -s "$out"
    write_status "PIX_COLLECTED"
    printf '%s\n' "$out"
}

check_data() {
    local kind=${2:-pcd} dataset=${3:?usage: $0 check-data pcd|pix DATASET}
    local report="$LOG_DIR/$(basename "$dataset").json"
    write_status "CHECKING_DATA"
    cd "$P"
    if [ "$kind" = pcd ]; then
        "$PY" scripts/checks/verify_franka_replay.py "$dataset" --report "$report" \
            2>&1 | tee "$LOG_DIR/check_data.log"
    elif [ "$kind" = pix ]; then
        "$PY" scripts/checks/verify_franka_replay.py "$dataset" --report "$report" --validate-only \
            2>&1 | tee "$LOG_DIR/check_data.log"
    else
        echo "Unknown data kind: $kind" >&2
        exit 2
    fi
    write_status "DATA_CHECKED"
}

train_smoke() {
    local dataset=${2:?usage: $0 train-smoke DATASET}
    local stamp run_name run_dir checkpoint
    stamp=$(date +%Y%m%d_%H%M%S)
    run_name=${RUN_NAME:-franka_pcd_smoke_${stamp}}
    run_dir="$STABLEWM_HOME/checkpoints/$run_name"
    checkpoint="$run_dir/weights_epoch_1.pt"
    refuse_existing "$run_dir"
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
        echo "GPU has active compute processes; refusing training smoke" >&2
        exit 3
    fi
    write_status "TRAINING_SMOKE"
    mkdir -p "$STABLEWM_HOME/logs/train"
    cd "$P"
    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
        data=ogb_franka_pcd \
        model=lewm_pcd \
        model/encoder=ms2 \
        data.dataset.name="$dataset" \
        subdir="$run_name" \
        output_model_name="$run_name" \
        trainer.max_epochs=1 \
        +trainer.limit_train_batches=2 \
        +trainer.limit_val_batches=1 \
        loader.batch_size=8 \
        loader.num_workers=1 \
        loader.persistent_workers=false \
        wandb.enabled=false \
        hydra.run.dir="$LOG_DIR/hydra_train" \
        2>&1 | tee "$LOG_DIR/train.log"
    test -s "$checkpoint"
    "$PY" scripts/checks/check_franka_checkpoint.py \
        "$dataset" "$run_name" \
        --report "$LOG_DIR/checkpoint_reload.txt"
    write_status "TRAINING_SMOKE_COMPLETE"
}

case "$STAGE" in
    verify)
        write_status "VERIFYING"
        cd "$P"
        CUDA_VISIBLE_DEVICES= "$PY" scripts/checks/verify_franka.py \
            --episodes "${VERIFY_EPISODES:-10}" 2>&1 | tee "$LOG_DIR/verify.log"
        write_status "VERIFIED"
        ;;
    collect-pcd)
        collect_pcd
        ;;
    collect-pix)
        collect_pix
        ;;
    check-data)
        check_data "$@"
        ;;
    train-smoke)
        train_smoke "$@"
        ;;
    *)
        echo "usage: $0 {verify|collect-pcd|collect-pix|check-data|train-smoke} [args...]" >&2
        exit 2
        ;;
esac
