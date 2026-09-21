#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-all}
PY=${PYTHON:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python}

RUNTIME_ROOT=${UR5E_TASK1_10K_HOME:-/autodl-fs/data/lewm/ogbench-ur5e-task1-10000}
SOURCE_RUNTIME=${UR5E_MULTITASK_HOME:-/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask}
SOURCE_TASK_ROOT=$SOURCE_RUNTIME/full/cube_double_task1_single_pnp

FAMILY=double
TASK_ID=1
TASK_SLUG=cube_double_task1_single_pnp
COMPLETED_EXTENSION_EPISODES=940
REMAINING_EXPERT_EPISODES=8060
NEW_EXPERT_EPISODES=9000
ORIGINAL_EXPERT_TRAIN_EPISODES=900
EXPANDED_EXPERT_TRAIN_EPISODES=9900
RANDOM_FAILURE_EPISODES=100
HOLDOUT_EXPERT_EPISODES=100
EXPANDED_TRAIN_EPISODES=10000
COLLECTION_SHARDS=24
MAX_PARALLEL=16
COLLECTION_NUM_ENVS=5
COLLECTION_BATCH_SIZE=20
COLLECTION_SEED=1000000
MIN_FREE_GB=80

LOG_ROOT=$RUNTIME_ROOT/logs
STATUS=$LOG_ROOT/ur5e_task1_expand10k.status
LOCK=$LOG_ROOT/ur5e_task1_expand10k.lock
RUN_LOG_DIR=$LOG_ROOT/task1_expand10k

SMOKE_ROOT=$RUNTIME_ROOT/smoke_e2e_v1
SMOKE_MARKER=$SMOKE_ROOT/SMOKE_ACCEPTED
FULL_ROOT=$RUNTIME_ROOT/full
FULL_DATA_MARKER=$FULL_ROOT/FULL_DATA_ACCEPTED
TRAINING_MARKER=$FULL_ROOT/TRAINING_ACCEPTED

PRESERVED_940=$RUNTIME_ROOT/extension/preserved_940.h5
REMAINING_8060=$RUNTIME_ROOT/extension/remaining_8060.h5
EXTENSION_9000=$RUNTIME_ROOT/extension/expert_9000.h5
ORIGINAL_EXPERT_900=$RUNTIME_ROOT/source/original_expert_900.h5
ORIGINAL_RANDOM_100=$RUNTIME_ROOT/source/original_random_failure_100.h5
EXPERT_9900=$FULL_ROOT/expert_9900.h5
FORMAL_TRAIN=$FULL_ROOT/train.h5
FORMAL_HOLDOUT=$FULL_ROOT/holdout.h5

FORMAL_RUN=ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_10ep
FORMAL_CHECKPOINT=$RUNTIME_ROOT/checkpoints/$FORMAL_RUN/weights_epoch_10.pt
FORMAL_REPORT=$RUNTIME_ROOT/reports/checkpoint_dexart_expert10k_epoch_10.json

mkdir -p "$LOG_ROOT" "$RUN_LOG_DIR" "$RUNTIME_ROOT/extension" \
  "$RUNTIME_ROOT/source" "$FULL_ROOT" "$RUNTIME_ROOT/reports"

export LEWM_PCD_ROOT=$ROOT
export STABLEWM_HOME=$RUNTIME_ROOT
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel

status() {
  local value=$1 detail=${2:-} tmp
  tmp=$(mktemp "$STATUS.tmp.XXXXXX")
  printf '%s time=%s pid=%s mode=%s %s\n' \
    "$value" "$(date --iso-8601=seconds)" "$$" "$MODE" "$detail" > "$tmp"
  mv "$tmp" "$STATUS"
}

write_marker() {
  local path=$1 value=$2 detail=$3 tmp
  tmp=$(mktemp "$path.tmp.XXXXXX")
  printf '%s time=%s %s\n' "$value" "$(date --iso-8601=seconds)" "$detail" > "$tmp"
  mv "$tmp" "$path"
}

fail() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    status "FAILED exit_code=$rc" "stage=${CURRENT_STAGE:-unknown}"
  fi
}
trap fail EXIT

validate() {
  local dataset=$1 episodes=$2 expert=$3 random=$4 report=$5
  "$PY" "$ROOT/scripts/data/accepted_episode_h5.py" "$dataset" \
    --family "$FAMILY" --task-id "$TASK_ID" --episodes "$episodes" \
    --expert "$expert" --random "$random" --report "$report"
}

require_idle_gpu() {
  local gpu_pids
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  [ -z "$gpu_pids" ] || {
    echo "GPU is already in use by: $gpu_pids" >&2
    return 3
  }
}

require_collection_resources() {
  local available required
  available=$(df -PB1 "$RUNTIME_ROOT" | awk 'NR == 2 {print $4}')
  required=$((MIN_FREE_GB * 1024 * 1024 * 1024))
  [ "$available" -ge "$required" ] || {
    echo "collection requires $MIN_FREE_GB GiB free; only $available bytes remain" >&2
    return 6
  }
  require_idle_gpu
}

collect_one() {
  local policy=$1
  shift
  [ "$1" = --episodes ]
  local episodes=$2 seed=$3 output=$4 report=$5
  local expert=0 random=0 resume=()
  if [ "$policy" = expert ]; then expert=$episodes; else random=$episodes; fi
  if [ -s "$output" ]; then
    validate "$output" "$episodes" "$expert" "$random" "$report"
    return 0
  fi
  [ ! -e "$output" ] || { echo "refusing to overwrite $output" >&2; return 2; }
  if [ -d "$output.work" ]; then resume=(--resume); fi
  "$PY" "$ROOT/scripts/data/collect_fixed_ur5e_task_pcd.py" \
    --family "$FAMILY" --task-id "$TASK_ID" --policy "$policy" \
    --episodes "$episodes" --seed "$seed" --out "$output" \
    --num-envs 1 --batch-size "$episodes" --max-attempts "$((episodes * 10))" \
    --cleanup-work "${resume[@]}"
  validate "$output" "$episodes" "$expert" "$random" "$report"
}

train_and_reload() {
  local run=$1 train_data=$2 holdout_data=$3 epochs=$4 limited=$5 batch_size=$6 report=$7
  local epoch_override=$8
  local checkpoint=$RUNTIME_ROOT/checkpoints/$run/weights_epoch_$epochs.pt
  local cache=$RUNTIME_ROOT/checkpoints/_spt_cache/$run
  local extra=()
  if [ ! -s "$checkpoint" ]; then
    test ! -e "$RUNTIME_ROOT/checkpoints/$run"
    require_idle_gpu
    mkdir -p "$cache" "$(dirname "$report")"
    export UR5E_TRAIN_DATA=$train_data UR5E_HOLDOUT_DATA=$holdout_data
    if [ "$limited" = yes ]; then
      extra=(+trainer.limit_train_batches=2 +trainer.limit_val_batches=1)
    fi
    (
      cd "$ROOT"
      SPT_CACHE_DIR="$cache" CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
        data=ogb_ur5e_task_pcd model=lewm_pcd model/encoder=dexart \
        data.dataset.name="$train_data" data.validation_name="$holdout_data" \
        subdir="$run" output_model_name="$run" "$epoch_override" \
        loader.batch_size="$batch_size" loader.num_workers=1 \
        loader.persistent_workers=false wandb.enabled=false \
        "${extra[@]}" hydra.run.dir="$RUN_LOG_DIR/hydra_$run"
    ) 2>&1 | tee "$RUN_LOG_DIR/$run.log"
  fi
  test -s "$checkpoint"
  if [ ! -s "$report" ]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/checks/check_ur5e_holdout_checkpoint.py" \
      --train-dataset "$train_data" --holdout-dataset "$holdout_data" \
      --checkpoint-run "$run" --epoch "$epochs" --report "$report" \
      --batch-size "$batch_size" --num-workers 1 \
      $([ "$limited" = yes ] && printf '%s' '--max-batches 1')
  fi
  grep -q '"status": "PASS"' "$report"
  rm -rf -- "$cache"
}

run_smoke() {
  local expert=$SMOKE_ROOT/expert_train.h5
  local random=$SMOKE_ROOT/random_failure.h5
  local holdout=$SMOKE_ROOT/holdout.h5
  local train=$SMOKE_ROOT/train.h5
  local report_dir=$SMOKE_ROOT/reports
  local run=ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_smoke_1ep
  local checkpoint_report=$report_dir/checkpoint_epoch_1.json
  CURRENT_STAGE=smoke
  status SMOKE_RUNNING
  mkdir -p "$SMOKE_ROOT" "$report_dir"
  if [ -s "$SMOKE_MARKER" ]; then
    validate "$train" 6 4 2 "$report_dir/train.json"
    validate "$holdout" 2 2 0 "$report_dir/holdout.json"
    grep -q '"status": "PASS"' "$checkpoint_report"
    status SMOKE_ACCEPTED
    return 0
  fi
  require_idle_gpu
  collect_one expert --episodes 4 910000 "$expert" "$report_dir/expert.json"
  collect_one random --episodes 2 920000 "$random" "$report_dir/random.json"
  collect_one expert --episodes 2 930000 "$holdout" "$report_dir/holdout_source.json"
  if [ ! -s "$train" ]; then
    "$PY" "$ROOT/scripts/data/publish_fixed_ur5e_task_dataset.py" \
      --expert-train "$expert" --random-failures "$random" --holdout "$holdout" \
      --train-out "$train" --family "$FAMILY" --task-id "$TASK_ID" \
      --expert-train-episodes 4 --random-failure-episodes 2 --holdout-episodes 2 \
      > "$report_dir/publish.json"
  fi
  validate "$train" 6 4 2 "$report_dir/train.json"
  validate "$holdout" 2 2 0 "$report_dir/holdout.json"
  train_and_reload "$run" "$train" "$holdout" 1 yes 8 "$checkpoint_report" \
    trainer.max_epochs=1
  write_marker "$SMOKE_MARKER" SMOKE_ACCEPTED \
    'expert_train=4 random_failure=2 holdout_expert=2 epochs=1'
  status SMOKE_ACCEPTED
}

merge_preserved_940() {
  local report=$RUNTIME_ROOT/reports/preserved_940.json
  local partial=$PRESERVED_940.partial
  local sources=()
  if [ -s "$PRESERVED_940" ]; then
    validate "$PRESERVED_940" "$COMPLETED_EXTENSION_EPISODES" \
      "$COMPLETED_EXTENSION_EPISODES" 0 "$report"
    return 0
  fi
  shopt -s nullglob
  sources+=("$RUNTIME_ROOT"/expert_add9000.h5.work/accepted_*.h5)
  sources+=("$RUNTIME_ROOT"/parallel_shards/shard_*.h5.work/accepted_*.h5)
  shopt -u nullglob
  [ "${#sources[@]}" -eq 47 ] || {
    echo "expected 47 preserved accepted shards, found ${#sources[@]}" >&2
    return 8
  }
  rm -f "$partial"
  cp --reflink=auto "${sources[0]}" "$partial"
  "$PY" "$ROOT/scripts/data/merge_h5_shards.py" \
    --out "$partial" "${sources[@]:1}"
  validate "$partial" "$COMPLETED_EXTENSION_EPISODES" \
    "$COMPLETED_EXTENSION_EPISODES" 0 "$report.partial"
  mv "$partial" "$PRESERVED_940"
  mv "$report.partial" "$report"
}

collect_remaining_8060() {
  local report=$RUNTIME_ROOT/reports/remaining_8060.json
  local resume=()
  if [ -s "$REMAINING_8060" ]; then
    validate "$REMAINING_8060" "$REMAINING_EXPERT_EPISODES" \
      "$REMAINING_EXPERT_EPISODES" 0 "$report"
    return 0
  fi
  [ ! -e "$REMAINING_8060" ] || {
    echo "refusing to overwrite $REMAINING_8060" >&2
    return 2
  }
  if [ -s "$REMAINING_8060.shards/collection_manifest.json" ]; then resume=(--resume); fi
  require_collection_resources
  "$PY" "$ROOT/scripts/data/collect_fixed_ur5e_task_pcd_sharded.py" \
    --family "$FAMILY" --task-id "$TASK_ID" --policy expert \
    --episodes "$REMAINING_EXPERT_EPISODES" --seed "$COLLECTION_SEED" \
    --out "$REMAINING_8060" --shards "$COLLECTION_SHARDS" \
    --max-parallel "$MAX_PARALLEL" --num-envs "$COLLECTION_NUM_ENVS" \
    --batch-size "$COLLECTION_BATCH_SIZE" --attempts-factor 10 --threads 1 \
    "${resume[@]}" 2>&1 | tee "$RUN_LOG_DIR/remaining_8060_collection.log"
  validate "$REMAINING_8060" "$REMAINING_EXPERT_EPISODES" \
    "$REMAINING_EXPERT_EPISODES" 0 "$report"
}

merge_extension_9000() {
  local partial=$EXTENSION_9000.partial
  local report=$RUNTIME_ROOT/reports/expert_9000.json
  if [ -s "$EXTENSION_9000" ]; then
    validate "$EXTENSION_9000" "$NEW_EXPERT_EPISODES" \
      "$NEW_EXPERT_EPISODES" 0 "$report"
    return 0
  fi
  rm -f "$partial"
  cp --reflink=auto "$PRESERVED_940" "$partial"
  "$PY" "$ROOT/scripts/data/merge_h5_shards.py" \
    --out "$partial" "$REMAINING_8060"
  validate "$partial" "$NEW_EXPERT_EPISODES" "$NEW_EXPERT_EPISODES" 0 \
    "$report.partial"
  mv "$partial" "$EXTENSION_9000"
  mv "$report.partial" "$report"
}

select_original_data() {
  if [ ! -s "$ORIGINAL_EXPERT_900" ]; then
    "$PY" "$ROOT/scripts/data/select_fixed_ur5e_episodes.py" \
      --source "$SOURCE_TASK_ROOT/train.h5" --output "$ORIGINAL_EXPERT_900" \
      --family "$FAMILY" --task-id "$TASK_ID" --outcome success \
      --policy expert --episodes "$ORIGINAL_EXPERT_TRAIN_EPISODES"
  fi
  if [ ! -s "$ORIGINAL_RANDOM_100" ]; then
    "$PY" "$ROOT/scripts/data/select_fixed_ur5e_episodes.py" \
      --source "$SOURCE_TASK_ROOT/train.h5" --output "$ORIGINAL_RANDOM_100" \
      --family "$FAMILY" --task-id "$TASK_ID" --outcome failure \
      --policy random --episodes "$RANDOM_FAILURE_EPISODES"
  fi
  validate "$ORIGINAL_EXPERT_900" "$ORIGINAL_EXPERT_TRAIN_EPISODES" \
    "$ORIGINAL_EXPERT_TRAIN_EPISODES" 0 "$RUNTIME_ROOT/reports/original_expert_900.json"
  validate "$ORIGINAL_RANDOM_100" "$RANDOM_FAILURE_EPISODES" 0 \
    "$RANDOM_FAILURE_EPISODES" "$RUNTIME_ROOT/reports/original_random_100.json"
  if [ ! -s "$FORMAL_HOLDOUT" ]; then
    cp --reflink=auto "$SOURCE_TASK_ROOT/holdout.h5" "$FORMAL_HOLDOUT.partial"
    validate "$FORMAL_HOLDOUT.partial" "$HOLDOUT_EXPERT_EPISODES" \
      "$HOLDOUT_EXPERT_EPISODES" 0 "$RUNTIME_ROOT/reports/holdout.partial.json"
    mv "$FORMAL_HOLDOUT.partial" "$FORMAL_HOLDOUT"
    mv "$RUNTIME_ROOT/reports/holdout.partial.json" "$RUNTIME_ROOT/reports/holdout.json"
  fi
  validate "$FORMAL_HOLDOUT" "$HOLDOUT_EXPERT_EPISODES" \
    "$HOLDOUT_EXPERT_EPISODES" 0 "$RUNTIME_ROOT/reports/holdout.json"
}

publish_expanded_data() {
  local partial=$EXPERT_9900.partial
  if [ ! -s "$EXPERT_9900" ]; then
    rm -f "$partial"
    cp --reflink=auto "$ORIGINAL_EXPERT_900" "$partial"
    "$PY" "$ROOT/scripts/data/merge_h5_shards.py" --out "$partial" "$EXTENSION_9000"
    validate "$partial" "$EXPANDED_EXPERT_TRAIN_EPISODES" \
      "$EXPANDED_EXPERT_TRAIN_EPISODES" 0 \
      "$RUNTIME_ROOT/reports/expert_9900.partial.json"
    mv "$partial" "$EXPERT_9900"
    mv "$RUNTIME_ROOT/reports/expert_9900.partial.json" \
      "$RUNTIME_ROOT/reports/expert_9900.json"
  fi
  validate "$EXPERT_9900" "$EXPANDED_EXPERT_TRAIN_EPISODES" \
    "$EXPANDED_EXPERT_TRAIN_EPISODES" 0 "$RUNTIME_ROOT/reports/expert_9900.json"
  if [ ! -s "$FORMAL_TRAIN" ]; then
    "$PY" "$ROOT/scripts/data/publish_fixed_ur5e_task_dataset.py" \
      --expert-train "$EXPERT_9900" --random-failures "$ORIGINAL_RANDOM_100" \
      --holdout "$FORMAL_HOLDOUT" --train-out "$FORMAL_TRAIN" \
      --family "$FAMILY" --task-id "$TASK_ID" \
      --expert-train-episodes "$EXPANDED_EXPERT_TRAIN_EPISODES" \
      --random-failure-episodes "$RANDOM_FAILURE_EPISODES" \
      --holdout-episodes "$HOLDOUT_EXPERT_EPISODES" \
      > "$RUNTIME_ROOT/reports/publish_expanded.json"
  fi
  validate "$FORMAL_TRAIN" "$EXPANDED_TRAIN_EPISODES" \
    "$EXPANDED_EXPERT_TRAIN_EPISODES" "$RANDOM_FAILURE_EPISODES" \
    "$RUNTIME_ROOT/reports/train_10000.json"
  validate "$FORMAL_HOLDOUT" "$HOLDOUT_EXPERT_EPISODES" \
    "$HOLDOUT_EXPERT_EPISODES" 0 "$RUNTIME_ROOT/reports/holdout.json"
  write_marker "$FULL_DATA_MARKER" FULL_DATA_ACCEPTED \
    'train_total=10000 train_expert=9900 train_random_failure=100 holdout_expert=100'
}

run_full_collection() {
  CURRENT_STAGE=full_collection
  status FULL_COLLECTION_RUNNING \
    "completed=$COMPLETED_EXTENSION_EPISODES remaining=$REMAINING_EXPERT_EPISODES"
  test -s "$SMOKE_MARKER"
  merge_preserved_940
  collect_remaining_8060
  merge_extension_9000
  select_original_data
  publish_expanded_data
  status FULL_DATA_ACCEPTED
}

run_formal_training() {
  CURRENT_STAGE=formal_training
  test -s "$FULL_DATA_MARKER"
  validate "$FORMAL_TRAIN" "$EXPANDED_TRAIN_EPISODES" \
    "$EXPANDED_EXPERT_TRAIN_EPISODES" "$RANDOM_FAILURE_EPISODES" \
    "$RUNTIME_ROOT/reports/train_10000.json"
  validate "$FORMAL_HOLDOUT" "$HOLDOUT_EXPERT_EPISODES" \
    "$HOLDOUT_EXPERT_EPISODES" 0 "$RUNTIME_ROOT/reports/holdout.json"
  status FORMAL_TRAINING_RUNNING "run=$FORMAL_RUN epochs=10"
  train_and_reload "$FORMAL_RUN" "$FORMAL_TRAIN" "$FORMAL_HOLDOUT" \
    10 no 128 "$FORMAL_REPORT" trainer.max_epochs=10
  test -s "$FORMAL_CHECKPOINT"
  grep -q '"status": "PASS"' "$FORMAL_REPORT"
  write_marker "$TRAINING_MARKER" TRAINING_ACCEPTED \
    "run=$FORMAL_RUN checkpoint=weights_epoch_10.pt"
  status TRAINING_ACCEPTED "run=$FORMAL_RUN"
}

main() {
  if [ "$MODE" = status ]; then
    cat "$STATUS"
    return 0
  fi
  case "$MODE" in
    smoke|full|all) ;;
    *) echo "usage: $0 {smoke|full|all|status}" >&2; return 2 ;;
  esac
  test -x "$PY"
  exec 9>"$LOCK"
  flock -n 9 || { echo "another task1 expansion pipeline is running" >&2; return 3; }
  case "$MODE" in
    smoke) run_smoke ;;
    full)
      test -s "$SMOKE_MARKER"
      run_full_collection
      run_formal_training
      ;;
    all)
      run_smoke
      run_full_collection
      run_formal_training
      ;;
  esac
}

main "$@"
