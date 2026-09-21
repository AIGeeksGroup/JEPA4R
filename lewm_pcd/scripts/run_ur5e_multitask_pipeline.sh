#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=smoke; if [ "$#" -gt 0 ]; then MODE=$1; fi
PY=$(printenv PYTHON || true); if [ -z "$PY" ]; then PY=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python; fi
FULL_COLLECT_ENVS=1
FULL_COLLECT_BATCH=20
FULL_COLLECT_SHARDS=8
FULL_COLLECT_MAX_PARALLEL=8
PARALLEL_SMOKE_SHARDS=2
PARALLEL_SMOKE_MAX_PARALLEL=2
MIN_COLLECT_FREE_GB=20
FULL_TRAIN_BATCH=128
RUNTIME_ROOT=$(printenv UR5E_MULTITASK_HOME || true); if [ -z "$RUNTIME_ROOT" ]; then RUNTIME_ROOT=/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask; fi
STATUS=$RUNTIME_ROOT/logs/ur5e_multitask_pipeline.status; LOCK=$RUNTIME_ROOT/logs/ur5e_multitask_pipeline.lock
PARALLEL_SMOKE_MARKER=$RUNTIME_ROOT/smoke/PARALLEL_COLLECTION_SMOKE_ACCEPTED
FULL_DATA_MARKER=$RUNTIME_ROOT/full/FULL_DATA_ACCEPTED
LOG_DIR=$RUNTIME_ROOT/logs/multitask/$MODE/$(date +%Y%m%d_%H%M%S)
export LEWM_PCD_ROOT=$ROOT STABLEWM_HOME=$RUNTIME_ROOT MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel
TASKS='
cube_double_task1_single_pnp double 1
cube_double_task2_double_pnp1 double 2
cube_double_task3_double_pnp2 double 3
cube_double_task4_swap double 4
cube_double_task5_stack double 5
cube_triple_task1_single_pnp triple 1
cube_triple_task2_triple_pnp triple 2
cube_triple_task3_pnp_from_stack triple 3
cube_triple_task4_cycle triple 4
'
mkdir -p "$RUNTIME_ROOT/logs" "$LOG_DIR"
status() { local tmp; tmp=$(mktemp "$STATUS.tmp.XXXXXX"); printf '%s time=%s pid=%s mode=%s log_dir=%s\n' "$1" "$(date --iso-8601=seconds)" "$$" "$MODE" "$LOG_DIR" > "$tmp"; mv "$tmp" "$STATUS"; }
write_marker() { local path=$1 value=$2 tmp; tmp=$(mktemp "$path.tmp.XXXXXX"); printf '%s time=%s tasks=9\n' "$value" "$(date --iso-8601=seconds)" > "$tmp"; mv "$tmp" "$path"; }
fail() { local rc=$?; if [ "$rc" -ne 0 ]; then status "FAILED exit_code=$rc"; fi; }; trap fail EXIT
validate() { "$PY" "$ROOT/scripts/data/accepted_episode_h5.py" "$1" --family "$2" --task-id "$3" --episodes "$4" --expert "$5" --random "$6" --report "$7"; }
collection_resource_gate() {
  local available required gpu_pids
  available=$(df -PB1 "$RUNTIME_ROOT" | awk 'NR == 2 {print $4}')
  required=$((MIN_COLLECT_FREE_GB * 1024 * 1024 * 1024))
  [ "$available" -ge "$required" ] || { echo "insufficient collection disk: $available bytes available" >&2; return 6; }
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  [ -z "$gpu_pids" ] || { echo "GPU busy before collection: $gpu_pids" >&2; return 3; }
}
collect() {
  local family=$1 id=$2 policy=$3 n=$4 seed=$5 out=$6 COLLECT_ENVS=$7 COLLECT_BATCH=$8; mkdir -p "$(dirname "$out")"; [ -s "$out" ] && return 0
  if [ -e "$out" ]; then echo "refusing to overwrite existing output: $out" >&2; return 2; fi
  local resume=; [ -d "$out.work" ] && resume=--resume
  "$PY" "$ROOT/scripts/data/collect_fixed_ur5e_task_pcd.py" --family "$family" --task-id "$id" --policy "$policy" --episodes "$n" --seed "$seed" --out "$out" --num-envs "$COLLECT_ENVS" --batch-size "$COLLECT_BATCH" --max-attempts "$((n * 10))" $resume 2>&1 | tee "$LOG_DIR/"$(basename "$out")".log"
  test -s "$out"
}
collect_parallel() {
  local family=$1 id=$2 policy=$3 n=$4 seed=$5 out=$6 shards=$7 max_parallel=$8 batch=$9
  local expert=0 random=0 resume=
  [ "$policy" = expert ] && expert=$n || random=$n
  if [ -s "$out" ]; then
    validate "$out" "$family" "$id" "$n" "$expert" "$random" "$out.validation.json"
    return 0
  fi
  if [ -e "$out" ]; then echo "refusing to overwrite existing output: $out" >&2; return 2; fi
  collection_resource_gate
  [ -s "$out.shards/collection_manifest.json" ] && resume=--resume
  "$PY" "$ROOT/scripts/data/collect_fixed_ur5e_task_pcd_sharded.py" \
    --family "$family" --task-id "$id" --policy "$policy" --episodes "$n" \
    --seed "$seed" --out "$out" --shards "$shards" \
    --max-parallel "$max_parallel" --num-envs "$FULL_COLLECT_ENVS" \
    --batch-size "$batch" --attempts-factor 10 --threads 2 $resume \
    2>&1 | tee "$LOG_DIR/"$(basename "$out")"_sharded.log"
  validate "$out" "$family" "$id" "$n" "$expert" "$random" "$out.validation.json"
}
rechunk_if_single_row() {
  local train=$1 family=$2 id=$3 episodes=$4 expert=$5 random=$6 report=$7
  local chunk_rows rechunked replacement_report
  chunk_rows=$("$PY" -c 'import h5py, sys; f=h5py.File(sys.argv[1], "r"); c=f["pcd_xyz"].chunks; print(c[0] if c else 0); f.close()' "$train")
  [ "$chunk_rows" != 1 ] && return 0
  rechunked="$train.rechunking"; replacement_report="$report.rechunking"
  rm -f "$rechunked" "$replacement_report"
  "$PY" "$ROOT/scripts/data/rechunk_h5.py" "$train" "$rechunked" --rows "$FULL_COLLECT_BATCH"
  validate "$rechunked" "$family" "$id" "$episodes" "$expert" "$random" "$replacement_report"
  mv "$rechunked" "$train"
  mv "$replacement_report" "$report"
}
publish() {
  local slug=$1 family=$2 id=$3 expert=$4 random=$5 holdout=$6 train=$7 ec=$8 rc=$9
  shift 9; local hc=$1 report=$RUNTIME_ROOT/data/$slug/reports; mkdir -p "$report"
  if [ ! -s "$train" ] || [ ! -s "$holdout" ]; then
    "$PY" "$ROOT/scripts/data/publish_fixed_ur5e_task_dataset.py" --expert-train "$expert" --random-failures "$random" --holdout "$holdout" --train-out "$train" --family "$family" --task-id "$id" --expert-train-episodes "$ec" --random-failure-episodes "$rc" --holdout-episodes "$hc" | tee "$LOG_DIR/"$slug"_publish.log"
  fi
  validate "$train" "$family" "$id" "$((ec + rc))" "$ec" "$rc" "$report/train.json"
  validate "$holdout" "$family" "$id" "$hc" "$hc" 0 "$report/holdout.json"
  rechunk_if_single_row "$train" "$family" "$id" "$((ec + rc))" "$ec" "$rc" "$report/train.json"
  rechunk_if_single_row "$holdout" "$family" "$id" "$hc" "$hc" 0 "$report/holdout.json"
  rm -f "$expert" "$random"
  rm -rf -- "$expert.work" "$random.work" "$holdout.work" \
    "$expert.shards" "$random.shards" "$holdout.shards"
}
train() {
  local slug=$1 train_data=$2 holdout_data=$3 epochs=$4 limited=$5 train_batch=$6
  local run=ur5e_"$slug"_pcd_dexart_"$epochs"ep; local ckpt=$RUNTIME_ROOT/checkpoints/$run/weights_epoch_$epochs.pt; local report=$RUNTIME_ROOT/data/$slug/reports/checkpoint_dexart_epoch_"$epochs".json
  local task_cache=$RUNTIME_ROOT/checkpoints/_spt_cache/$run
  [ -s "$ckpt" ] && [ -s "$report" ] && return 0
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && { echo "GPU busy" >&2; return 3; } || true
  export UR5E_TRAIN_DATA=$train_data UR5E_HOLDOUT_DATA=$holdout_data; mkdir -p "$(dirname "$ckpt")" "$(dirname "$report")" "$task_cache"
  cd "$ROOT"; local extra=; [ "$limited" = yes ] && extra="+trainer.limit_train_batches=2 +trainer.limit_val_batches=1"
  SPT_CACHE_DIR="$task_cache" CUDA_VISIBLE_DEVICES=0 "$PY" train.py data=ogb_ur5e_task_pcd model=lewm_pcd model/encoder=dexart data.dataset.name="$train_data" data.validation_name="$holdout_data" subdir="$run" output_model_name="$run" trainer.max_epochs="$epochs" loader.batch_size="$train_batch" loader.num_workers=1 loader.persistent_workers=false wandb.enabled=false $extra hydra.run.dir="$LOG_DIR/hydra_$slug" 2>&1 | tee "$RUNTIME_ROOT/logs/"$run".log"
  test -s "$ckpt"; "$PY" "$ROOT/scripts/checks/check_ur5e_holdout_checkpoint.py" --train-dataset "$train_data" --holdout-dataset "$holdout_data" --checkpoint-run "$run" --epoch "$epochs" --report "$report"
  rm -rf -- "$task_cache"
}
run_smoke() {
  local root=$RUNTIME_ROOT/smoke index=0; status SMOKE_RUNNING; mkdir -p "$root"
  while read -r slug family id; do
    [ -z "$slug" ] && continue; local d=$root/$slug; local s=$((50000 + index * 1000)); mkdir -p "$d"
    collect "$family" "$id" expert 3 "$s" "$d/expert_train.h5" 1 3; collect "$family" "$id" expert 1 "$((s + 100))" "$d/holdout.h5" 1 1; collect "$family" "$id" random 1 "$((s + 200))" "$d/random_fail.h5" 1 1
    publish "$slug" "$family" "$id" "$d/expert_train.h5" "$d/random_fail.h5" "$d/holdout.h5" "$d/train.h5" 3 1 1; train "$slug" "$d/train.h5" "$d/holdout.h5" 1 yes 8; index=$((index + 1))
  done <<< "$TASKS"; printf 'SMOKE_ACCEPTED time=%s tasks=9\n' "$(date --iso-8601=seconds)" > "$root/SMOKE_ACCEPTED"; status SMOKE_ACCEPTED
}
run_parallel_smoke() {
  local root=$RUNTIME_ROOT/smoke/parallel_collection_v1 family=double id=1
  local expert=$root/expert.h5 random=$root/random_fail.h5 marker_tmp
  status PARALLEL_SMOKE_RUNNING; mkdir -p "$root"
  collect_parallel "$family" "$id" expert 4 950000 "$expert" "$PARALLEL_SMOKE_SHARDS" "$PARALLEL_SMOKE_MAX_PARALLEL" 2
  collect_parallel "$family" "$id" random 4 960000 "$random" "$PARALLEL_SMOKE_SHARDS" "$PARALLEL_SMOKE_MAX_PARALLEL" 2
  validate "$expert" "$family" "$id" 4 4 0 "$root/expert.validation.json"
  validate "$random" "$family" "$id" 4 0 4 "$root/random.validation.json"
  marker_tmp=$(mktemp "$PARALLEL_SMOKE_MARKER.tmp.XXXXXX")
  printf 'PARALLEL_COLLECTION_SMOKE_ACCEPTED time=%s shards=2 max_parallel=2\n' "$(date --iso-8601=seconds)" > "$marker_tmp"
  mv "$marker_tmp" "$PARALLEL_SMOKE_MARKER"; status PARALLEL_SMOKE_ACCEPTED
}
run_full_collection() {
  local root=$RUNTIME_ROOT/full index=0; status FULL_COLLECTION_RUNNING; mkdir -p "$root"
  while read -r slug family id; do
    [ -z "$slug" ] && continue; local d=$root/$slug; local s=$((150000 + index * 10000)); mkdir -p "$d"
    if [ ! -s "$d/train.h5" ] || [ ! -s "$d/holdout.h5" ]; then
      test -s "$PARALLEL_SMOKE_MARKER" || { echo "full collection requires PARALLEL_COLLECTION_SMOKE_ACCEPTED" >&2; return 5; }
      collect_parallel "$family" "$id" expert 900 "$s" "$d/expert_train.h5" "$FULL_COLLECT_SHARDS" "$FULL_COLLECT_MAX_PARALLEL" "$FULL_COLLECT_BATCH"
      collect_parallel "$family" "$id" expert 100 "$((s + 100000))" "$d/holdout.h5" "$FULL_COLLECT_SHARDS" "$FULL_COLLECT_MAX_PARALLEL" "$FULL_COLLECT_BATCH"
      collect_parallel "$family" "$id" random 100 "$((s + 200000))" "$d/random_fail.h5" "$FULL_COLLECT_SHARDS" "$FULL_COLLECT_MAX_PARALLEL" "$FULL_COLLECT_BATCH"
    fi
    publish "$slug" "$family" "$id" "$d/expert_train.h5" "$d/random_fail.h5" "$d/holdout.h5" "$d/train.h5" 900 100 100; index=$((index + 1))
  done <<< "$TASKS"
}
validate_all_full_data() {
  local root=$RUNTIME_ROOT/full
  while read -r slug family id; do
    [ -z "$slug" ] && continue; local d=$root/$slug
    validate "$d/train.h5" "$family" "$id" 1000 900 100 "$RUNTIME_ROOT/data/$slug/reports/train.json"
    validate "$d/holdout.h5" "$family" "$id" 100 100 0 "$RUNTIME_ROOT/data/$slug/reports/holdout.json"
  done <<< "$TASKS"
  write_marker "$FULL_DATA_MARKER" FULL_DATA_ACCEPTED; status FULL_DATA_ACCEPTED
}
run_full_training() {
  test -s "$FULL_DATA_MARKER" || { echo "full training requires FULL_DATA_ACCEPTED" >&2; return 7; }
  local root=$RUNTIME_ROOT/full; status FULL_TRAINING_RUNNING
  while read -r slug family id; do
    [ -z "$slug" ] && continue; local d=$root/$slug
    train "$slug" "$d/train.h5" "$d/holdout.h5" 10 no "$FULL_TRAIN_BATCH"
  done <<< "$TASKS"
}
run_full() {
  test -s "$RUNTIME_ROOT/smoke/SMOKE_ACCEPTED" || { echo "full run requires SMOKE_ACCEPTED" >&2; return 4; }
  run_full_collection
  validate_all_full_data
  run_full_training
  write_marker "$RUNTIME_ROOT/full/FULL_ACCEPTED" FULL_ACCEPTED; status FULL_ACCEPTED
}
main() {
  [ "$MODE" = status ] && { cat "$STATUS"; return 0; }; case "$MODE" in smoke|parallel-smoke|full) ;; *) echo "usage: $0 {smoke|parallel-smoke|full|status}" >&2; return 2;; esac
  test -x "$PY"; exec 9>"$LOCK"; flock -n 9 || { echo "another pipeline is running" >&2; return 3; }
  case "$MODE" in smoke) run_smoke;; parallel-smoke) run_parallel_smoke;; full) run_full;; esac
}
main "$@"
