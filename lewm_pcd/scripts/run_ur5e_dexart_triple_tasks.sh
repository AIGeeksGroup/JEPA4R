#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-multitask/lewm_pcd
RUNTIME=/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask
PY=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
STATUS=$RUNTIME/logs/ur5e_dexart_triple_tasks.status
ACCEPTED=$RUNTIME/full/DEXART_NINE_TASKS_ACCEPTED
PREVIOUS=$RUNTIME/full/DEXART_FOUR_TASKS_ACCEPTED
PREVIOUS_STATUS=$RUNTIME/logs/ur5e_dexart_four_tasks.status
LOCK=$RUNTIME/logs/ur5e_multitask_pipeline.lock
QUEUE_LOCK=$RUNTIME/logs/ur5e_dexart_triple_tasks.lock

TASKS=(
  cube_triple_task1_single_pnp
  cube_triple_task2_triple_pnp
  cube_triple_task3_pnp_from_stack
  cube_triple_task4_cycle
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

exec 8>"$QUEUE_LOCK"
flock -n 8
test ! -e "$ACCEPTED"
trap fail EXIT
status DEXART_TRIPLE_TASKS_WAITING

while ! test -s "$PREVIOUS"; do
  test -s "$PREVIOUS_STATUS" || { echo "Missing four-task status" >&2; exit 4; }
  if grep -q '^FAILED' "$PREVIOUS_STATUS"; then
    echo "Four-task runner failed; refusing to continue" >&2
    exit 4
  fi
  previous_pid=$(sed -nE 's/.* pid=([0-9]+).*/\1/p' "$PREVIOUS_STATUS")
  if [ -z "$previous_pid" ] || [ ! -r "/proc/$previous_pid/cmdline" ]; then
    echo "Four-task runner exited without acceptance" >&2
    exit 4
  fi
  if ! tr '\0' ' ' < "/proc/$previous_pid/cmdline" | grep -qF 'run_ur5e_dexart_four_tasks.sh'; then
    echo "Four-task PID no longer belongs to its runner" >&2
    exit 4
  fi
  sleep 30
done

exec 9>"$LOCK"
flock -w 120 9
grep -q '^DEXART_FOUR_TASKS_ACCEPTED ' "$PREVIOUS_STATUS"
test -s "$RUNTIME/full/FULL_DATA_ACCEPTED"
for slug in "${TASKS[@]}"; do
  run=ur5e_${slug}_pcd_dexart_10ep
  report=$RUNTIME/data/$slug/reports/checkpoint_dexart_epoch_10.json
  test -s "$RUNTIME/full/$slug/train.h5"
  test -s "$RUNTIME/full/$slug/holdout.h5"
  test ! -e "$RUNTIME/checkpoints/$run"
  test ! -e "$report"
done

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$gpu_pids" ] || { echo "GPU busy before triple-task run: $gpu_pids" >&2; exit 3; }

export LEWM_PCD_ROOT=$ROOT STABLEWM_HOME=$RUNTIME MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel
cd "$ROOT"
status DEXART_TRIPLE_TASKS_RUNNING

for slug in "${TASKS[@]}"; do
  CURRENT_TASK=$slug
  train_data=$RUNTIME/full/$slug/train.h5
  holdout_data=$RUNTIME/full/$slug/holdout.h5
  run=ur5e_${slug}_pcd_dexart_10ep
  checkpoint=$RUNTIME/checkpoints/$run/weights_epoch_10.pt
  report=$RUNTIME/data/$slug/reports/checkpoint_dexart_epoch_10.json
  log=$RUNTIME/logs/${run}.log
  cache=$RUNTIME/checkpoints/_spt_cache/$run

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
printf 'DEXART_NINE_TASKS_ACCEPTED time=%s tasks=9\n' "$(date --iso-8601=seconds)" > "$tmp"
mv "$tmp" "$ACCEPTED"
status DEXART_NINE_TASKS_ACCEPTED all
