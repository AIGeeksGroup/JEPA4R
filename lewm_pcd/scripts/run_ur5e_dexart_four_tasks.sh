#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-multitask/lewm_pcd
RUNTIME=/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask
PY=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
STATUS=$RUNTIME/logs/ur5e_dexart_four_tasks.status
ACCEPTED=$RUNTIME/full/DEXART_FOUR_TASKS_ACCEPTED
LOCK=$RUNTIME/logs/ur5e_multitask_pipeline.lock

TASKS=(
  cube_double_task2_double_pnp1
  cube_double_task3_double_pnp2
  cube_double_task4_swap
  cube_double_task5_stack
)

status() {
  local value=$1 task=${2:-none} tmp
  tmp=$(mktemp "$STATUS.tmp.XXXXXX")
  printf '%s time=%s pid=%s task=%s\n' "$value" "$(date --iso-8601=seconds)" "$$" "$task" > "$tmp"
  mv "$tmp" "$STATUS"
}

fail() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    status "FAILED exit_code=$rc" "${CURRENT_TASK:-none}"
  fi
}
trap fail EXIT

exec 9>"$LOCK"
flock -n 9
test -s "$RUNTIME/full/FULL_DATA_ACCEPTED"
test -s "$RUNTIME/logs/ur5e_dexart_task1.status"
grep -q '^DEXART_TASK1_ACCEPTED ' "$RUNTIME/logs/ur5e_dexart_task1.status"

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
[ -z "$gpu_pids" ] || { echo "GPU busy before four-task run: $gpu_pids" >&2; exit 3; }

export LEWM_PCD_ROOT=$ROOT STABLEWM_HOME=$RUNTIME MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel
cd "$ROOT"
status DEXART_FOUR_TASKS_RUNNING

for slug in "${TASKS[@]}"; do
  CURRENT_TASK=$slug
  train_data=$RUNTIME/full/$slug/train.h5
  holdout_data=$RUNTIME/full/$slug/holdout.h5
  run=ur5e_${slug}_pcd_dexart_10ep
  checkpoint=$RUNTIME/checkpoints/$run/weights_epoch_10.pt
  report=$RUNTIME/data/$slug/reports/checkpoint_dexart_epoch_10.json
  log=$RUNTIME/logs/${run}.log
  cache=$RUNTIME/checkpoints/_spt_cache/$run

  test -s "$train_data"
  test -s "$holdout_data"
  test ! -e "$RUNTIME/checkpoints/$run"
  test ! -e "$report"
  mkdir -p "$cache" "$(dirname "$report")"
  export UR5E_TRAIN_DATA=$train_data UR5E_HOLDOUT_DATA=$holdout_data
  status DEXART_TASK_TRAINING_RUNNING "$slug"
  SPT_CACHE_DIR="$cache" CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
    data=ogb_ur5e_task_pcd model=lewm_pcd model/encoder=dexart \
    data.dataset.name="$train_data" data.validation_name="$holdout_data" \
    subdir="$run" output_model_name="$run" trainer.max_epochs=10 \
    loader.batch_size=128 loader.num_workers=1 loader.persistent_workers=false \
    wandb.enabled=false hydra.run.dir="$RUNTIME/logs/hydra_$run" 2>&1 | tee "$log"
  test -s "$checkpoint"
  status DEXART_TASK_VALIDATING "$slug"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/checks/check_ur5e_holdout_checkpoint.py \
    --train-dataset "$train_data" --holdout-dataset "$holdout_data" \
    --checkpoint-run "$run" --epoch 10 --report "$report" \
    --batch-size 128 --num-workers 1
  test -s "$report"
  grep -q '"status": "PASS"' "$report"
  rm -rf -- "$cache"
  status DEXART_TASK_ACCEPTED "$slug"
done

tmp=$(mktemp "$ACCEPTED.tmp.XXXXXX")
printf 'DEXART_FOUR_TASKS_ACCEPTED time=%s tasks=4\n' "$(date --iso-8601=seconds)" > "$tmp"
mv "$tmp" "$ACCEPTED"
status DEXART_FOUR_TASKS_ACCEPTED all
