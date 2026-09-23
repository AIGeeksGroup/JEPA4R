#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
RUNTIME=/autodl-fs/data/lewm/ogbench-ur5e-mixed-1cube-2cube-pcd-v2
TRAIN=$RUNTIME/train.h5
VALIDATION=$RUNTIME/validation.h5
RUN=ur5e_cube_mixed_1cube9900_2cube_task1_10000_pcd_dexart_10ep
SMOKE_RUN=ur5e_cube_mixed_1cube_2cube_pcd_dexart_smoke_1ep
LOG_DIR=$RUNTIME/logs
REPORT_DIR=$RUNTIME/reports
STATE=$RUNTIME/state.json
LOCK=$RUNTIME/training.lock
EXIT_CODE=$RUNTIME/exit_code

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$RUNTIME/checkpoints"
exec 9>"$LOCK"
flock -n 9 || { echo "mixed training supervisor is already running" >&2; exit 9; }

export STABLEWM_HOME=$RUNTIME
export SPT_CACHE_DIR=$RUNTIME/checkpoints/_spt_cache
export UR5E_MIXED_TRAIN_DATA=$TRAIN
export UR5E_MIXED_VALIDATION_DATA=$VALIDATION
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH=$ROOT:${PYTHONPATH:-}

CURRENT_STAGE=PREFLIGHT

write_state() {
  local stage=$1 detail=${2:-} tmp now
  now=$(date --iso-8601=seconds)
  tmp=$(mktemp "$STATE.tmp.XXXXXX")
  STATE_STAGE="$stage" STATE_DETAIL="$detail" STATE_TIME="$now" STATE_PID="$$" \
    "$PY" -c 'import json,os; print(json.dumps({"stage":os.environ["STATE_STAGE"],"detail":os.environ["STATE_DETAIL"],"time":os.environ["STATE_TIME"],"pid":int(os.environ["STATE_PID"])},indent=2,sort_keys=True))' \
    > "$tmp"
  mv "$tmp" "$STATE"
}

on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [ "$rc" -ne 0 ]; then
    write_state FAILED "stage=$CURRENT_STAGE exit_code=$rc"
  fi
}
trap on_exit EXIT

require_idle_gpu() {
  local gpu_pids
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  [ -z "$gpu_pids" ] || {
    echo "GPU is already in use by: $gpu_pids" >&2
    return 3
  }
}

run_training() {
  local run=$1 epochs=$2 batch_size=$3 smoke=$4
  local checkpoint=$RUNTIME/checkpoints/$run/weights_epoch_${epochs}.pt
  local run_log=$LOG_DIR/$run.log
  local reload_report=$REPORT_DIR/${run}_reload.json
  local extras=()
  local workers=6 persistent=true

  if [ ! -s "$checkpoint" ]; then
    [ ! -e "$RUNTIME/checkpoints/$run" ] || {
      echo "refusing to overwrite incomplete checkpoint directory: $run" >&2
      return 4
    }
    if [ "$smoke" = yes ]; then
      extras=(+trainer.limit_train_batches=2 +trainer.limit_val_batches=1)
      workers=2
      persistent=false
    fi
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
        data=ogb_ur5e_mixed_pcd model=lewm_pcd model/encoder=dexart \
        data.dataset.name="$TRAIN" data.validation_name="$VALIDATION" \
        subdir="$run" output_model_name="$run" \
        trainer.max_epochs="$epochs" trainer.devices=1 \
        loader.batch_size="$batch_size" loader.num_workers="$workers" \
        loader.persistent_workers="$persistent" wandb.enabled=false \
        hydra.run.dir="$LOG_DIR/hydra_$run" "${extras[@]}"
    ) 2>&1 | tee "$run_log"
  fi
  [ -s "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; return 5; }

  if [ ! -s "$reload_report" ]; then
    local max_batches=5
    [ "$smoke" = yes ] && max_batches=1
    CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/checks/check_ur5e_holdout_checkpoint.py" \
      --train-dataset "$TRAIN" --holdout-dataset "$VALIDATION" \
      --checkpoint-run "$run" --epoch "$epochs" --report "$reload_report" \
      --batch-size "$batch_size" --num-workers 1 --max-batches "$max_batches"
  fi
  grep -q '"status": "PASS"' "$reload_report"
}

test -s "$TRAIN"
test -s "$VALIDATION"
grep -q '"status": "PASS"' "$RUNTIME/acceptance.json"
grep -q '"robot": "UR5e + Robotiq 2F-85"' "$RUNTIME/manifest.json"
require_idle_gpu

CURRENT_STAGE=SMOKE_TRAINING
write_state SMOKE_TRAINING "run=$SMOKE_RUN train=$TRAIN validation=$VALIDATION"
run_training "$SMOKE_RUN" 1 8 yes

CURRENT_STAGE=FORMAL_TRAINING
write_state FORMAL_TRAINING "run=$RUN epochs=10 batch_size=128"
run_training "$RUN" 10 128 no

CURRENT_STAGE=COMPLETE
write_state COMPLETE "run=$RUN checkpoint=$RUNTIME/checkpoints/$RUN/weights_epoch_10.pt"
