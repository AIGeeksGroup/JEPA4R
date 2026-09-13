#!/usr/bin/env bash
# Train and evaluate the original PCD encoder against the Sonata replacement.
# The script is intentionally self-contained so the old hard-coded launcher
# paths cannot redirect the run to a stale checkout.
#
# Usage (run detached on the 5090):
#   setsid nohup bash scripts/run_pcd_encoder_comparison.sh \
#     > /root/autodl-tmp/workspace/users/lewm/jepa4r-full-data/logs/encoder_compare.launch.log 2>&1 < /dev/null &
#
# Useful overrides are environment variables: BASELINE_ENCODER=ms2|ppt|dexart,
# RUN_TAG=<name>, MAX_EPOCHS=10, NUM_WORKERS=6, EVAL_MODE=all|final,
# STABILITY_SECONDS=120.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/workspace/users/lewm/JEPA4R-sonata-cls/lewm_pcd}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/workspace/users/lewm/jepa4r-full-data}"
TRAIN_PY="${TRAIN_PY:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python3.10}"
SONATA_CHECKPOINT="${SONATA_CHECKPOINT:-/workspace/users/lewm/checkpoints/sonata/sonata_small.pth}"
SONATA_SHA256="${SONATA_SHA256:-ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1}"
BASELINE_ENCODER="${BASELINE_ENCODER:-ms2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EVAL_MODE="${EVAL_MODE:-all}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
DRY_RUN="${DRY_RUN:-0}"
SONATA_ONLY="${SONATA_ONLY:-0}"

case "$BASELINE_ENCODER" in
  ms2|ppt|dexart) ;;
  *) echo "unsupported BASELINE_ENCODER=$BASELINE_ENCODER (use ms2, ppt, or dexart)" >&2; exit 2 ;;
esac
case "$EVAL_MODE" in
  all|final) ;;
  *) echo "unsupported EVAL_MODE=$EVAL_MODE (use all or final)" >&2; exit 2 ;;
esac

RUN_BASELINE="pcd_${BASELINE_ENCODER}_baseline_${RUN_TAG}"
RUN_SONATA="pcd_sonata_${RUN_TAG}"
RUN_ROOT="$DATA_ROOT/logs/encoder_compare/$RUN_TAG"
STATUS_DIR="$RUN_ROOT/status"
CSV="$RUN_ROOT/success_rates.csv"
PCD_DATA="$DATA_ROOT/datasets/ogbench/cube_single_expert_pcd.h5"

export LEWM_PCD_ROOT="$PROJECT_ROOT"
export STABLEWM_HOME="$DATA_ROOT"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/root/autodl-tmp/workspace/users/lewm/spt-cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$DATA_ROOT/logs/train"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT" "$STATUS_DIR" "$DATA_ROOT/logs/train" "$DATA_ROOT/logs/eval"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
fatal() { log "FATAL: $*"; touch "$STATUS_DIR/fatal"; exit 1; }

cgroup_gb() {
  local value
  value=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  echo $((value / 1073741824))
}

anon_gb() {
  local value
  value=$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
  echo $((value / 1073741824))
}

gpu_free_mb() {
  local total used
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo $((total - used))
}

preflight() {
  [[ -d "$PROJECT_ROOT" ]] || fatal "missing PROJECT_ROOT=$PROJECT_ROOT"
  [[ -x "$TRAIN_PY" ]] || fatal "missing TRAIN_PY=$TRAIN_PY"
  [[ -f "$PCD_DATA" ]] || fatal "missing PCD dataset=$PCD_DATA"
  [[ -f "$SONATA_CHECKPOINT" ]] || fatal "missing Sonata checkpoint=$SONATA_CHECKPOINT"
  if [[ "$SONATA_ONLY" != 1 ]]; then
    [[ ! -e "$DATA_ROOT/checkpoints/$RUN_BASELINE" ]] || fatal "run already exists: $RUN_BASELINE"
  fi
  [[ ! -e "$DATA_ROOT/checkpoints/$RUN_SONATA" ]] || fatal "run already exists: $RUN_SONATA"

  local gpu_line used hash
  gpu_line=$(nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | head -1) || fatal "nvidia-smi unavailable"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( used < 2000 )) || fatal "GPU is not idle (used ${used} MiB): $gpu_line"

  hash=$(sha256sum "$SONATA_CHECKPOINT" | awk '{print $1}')
  [[ "$hash" == "$SONATA_SHA256" ]] || fatal "Sonata SHA-256 mismatch: expected $SONATA_SHA256 got $hash"

  "$TRAIN_PY" - "$PCD_DATA" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    required = {"pcd_xyz", "pcd_rgb", "action", "ep_idx", "ep_len", "ep_offset"}
    missing = required.difference(f.keys())
    if missing:
        raise SystemExit(f"missing dataset keys: {sorted(missing)}")
    assert f["pcd_xyz"].shape == (2_010_000, 1_200, 3), f["pcd_xyz"].shape
    assert f["pcd_rgb"].shape == (2_010_000, 1_200, 3), f["pcd_rgb"].shape
    assert f["ep_len"].shape == (10_000,), f["ep_len"].shape
    assert f["ep_offset"].shape == (10_000,), f["ep_offset"].shape
    print("dataset preflight: 10000 episodes, 2010000 frames, 1200 points, eval columns present")
PY

  (cd "$PROJECT_ROOT" && "$TRAIN_PY" -m pytest -q scripts/checks/test_sonata_cls_encoder.py) \
    > "$RUN_ROOT/preflight_sonata_pytest.log" 2>&1 \
    || fatal "Sonata encoder tests failed; see $RUN_ROOT/preflight_sonata_pytest.log"

  printf 'run_tag=%s\nproject_root=%s\ndata_root=%s\nbaseline_encoder=%s\n' \
    "$RUN_TAG" "$PROJECT_ROOT" "$DATA_ROOT" "$BASELINE_ENCODER" > "$RUN_ROOT/preflight.env"
  touch "$STATUS_DIR/preflight.ok"
  log "preflight passed: $gpu_line; cgroup=$(cgroup_gb) GB anon=$(anon_gb) GB"
}

monitor_stability() {
  local run="$1" pid="$2" log_file="$3" elapsed
  for ((elapsed = 0; elapsed < STABILITY_SECONDS; elapsed++)); do
    kill -0 "$pid" 2>/dev/null || fatal "$run exited before stability gate; see $log_file"
    sleep 1
  done
  grep -Eq 'GPU available|Trainer.fit|Epoch|CUDA' "$log_file" \
    || fatal "$run stayed alive but produced no training-start evidence; see $log_file"
  touch "$STATUS_DIR/${run}.stable"
  log "$run stable after ${STABILITY_SECONDS}s (pid=$pid)"
}

train_one() {
  local encoder="$1" run="$2" log_file pid
  log_file="$RUN_ROOT/train_${run}.log"
  log "launching $run with encoder=$encoder"
  touch "$STATUS_DIR/${run}.launching"
  (
    cd "$PROJECT_ROOT"
    exec "$TRAIN_PY" train.py --config-name=lewm \
      data=ogb_pcd model=lewm_pcd model/encoder="$encoder" \
      subdir="$run" output_model_name="$run" \
      trainer.max_epochs="$MAX_EPOCHS" \
      num_workers="$NUM_WORKERS" loader.num_workers="$NUM_WORKERS" \
      hydra.run.dir="$RUN_ROOT/hydra_${run}"
  ) > "$log_file" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$STATUS_DIR/${run}.pid"
  touch "$STATUS_DIR/${run}.started"
  monitor_stability "$run" "$pid" "$log_file"
  if wait "$pid"; then
    touch "$STATUS_DIR/${run}.done"
    log "$run training complete"
  else
    fatal "$run training failed; see $log_file"
  fi
}

wait_for_eval_memory() {
  while (( $(cgroup_gb) >= 60 || $(anon_gb) >= 40 )); do
    log "waiting before eval: cgroup=$(cgroup_gb) GB anon=$(anon_gb) GB"
    sleep 60
  done
}

eval_checkpoint() {
  local run="$1" epoch="$2" backend dev log_file pid ec sr t0 elapsed
  log_file="$RUN_ROOT/eval_${run}_epoch${epoch}.log"
  wait_for_eval_memory
  if (( $(gpu_free_mb) >= 12000 )); then
    backend=egl
    dev=cuda
  else
    backend=osmesa
    dev=cpu
  fi
  log "evaluating $run epoch=$epoch using ${backend}/${dev}"
  t0=$(date +%s)
  (
    cd "$PROJECT_ROOT"
    MUJOCO_GL="$backend" PYOPENGL_PLATFORM="$backend" LP_NUM_THREADS=8 OMP_NUM_THREADS=8 \
      nice -n 10 "$TRAIN_PY" eval.py --config-name=cube_pcd.yaml \
      policy="$run/weights_epoch_${epoch}.pt" device="$dev" solver.device="$dev" \
      video=false hydra.run.dir="$RUN_ROOT/hydra_eval_${run}_epoch${epoch}"
  ) > "$log_file" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    if (( $(cgroup_gb) >= 78 || $(anon_gb) >= 60 )); then
      log "memory guard stopping eval $run epoch=$epoch"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
      elapsed=$(( $(date +%s) - t0 ))
      printf '%s,%s,ERROR(memory_guard),%s,%s/%s,%s\n' "$run" "$epoch" "$elapsed" "$backend" "$dev" "$(date '+%F %T')" >> "$CSV"
      return
    fi
  done
  if wait "$pid"; then
    ec=0
  else
    ec=$?
  fi
  elapsed=$(( $(date +%s) - t0 ))
  sr=$(grep -oE "'success_rate': [0-9.]+" "$log_file" | tail -1 | grep -oE '[0-9.]+$' || true)
  printf '%s,%s,%s,%s,%s/%s,%s\n' "$run" "$epoch" "${sr:-ERROR(exit $ec)}" "$elapsed" "$backend" "$dev" "$(date '+%F %T')" >> "$CSV"
  log "eval finished $run epoch=$epoch score=${sr:-ERROR}"
}

eval_run() {
  local run="$1" ckpts epoch
  mapfile -t ckpts < <(find "$DATA_ROOT/checkpoints/$run" -maxdepth 1 -type f -name 'weights_epoch_*.pt' -print | sort -V)
  (( ${#ckpts[@]} > 0 )) || fatal "no epoch checkpoints found for $run"
  if [[ "$EVAL_MODE" == final ]]; then
    ckpts=("${ckpts[${#ckpts[@]}-1]}")
  fi
  for ckpt in "${ckpts[@]}"; do
    epoch=$(basename "$ckpt" .pt | grep -oE '[0-9]+$')
    eval_checkpoint "$run" "$epoch"
  done
}

log "encoder comparison started: baseline=$BASELINE_ENCODER runs=$RUN_BASELINE,$RUN_SONATA"
printf 'run,epoch,success_rate,eval_seconds,backend,finished_at\n' > "$CSV"
preflight
if [[ "$DRY_RUN" == 1 ]]; then
  log "dry-run: validation passed"
  if [[ "$SONATA_ONLY" == 1 ]]; then
    log "Sonata-only mode: baseline training/evaluation skipped"
  else
    log "baseline encoder: $BASELINE_ENCODER"
  fi
  log "Sonata encoder: sonata_cls"
  log "training/evaluation not started"
  exit 0
fi
if [[ "$SONATA_ONLY" != 1 ]]; then
  train_one "$BASELINE_ENCODER" "$RUN_BASELINE"
fi
train_one sonata_cls "$RUN_SONATA"
if [[ "$SONATA_ONLY" != 1 ]]; then
  eval_run "$RUN_BASELINE"
fi
eval_run "$RUN_SONATA"
touch "$STATUS_DIR/complete"
log "encoder comparison complete; results=$CSV"
