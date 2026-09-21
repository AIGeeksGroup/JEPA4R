#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python}
RUNTIME_ROOT=${UR5E_MULTITASK_HOME:-/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask}
LOCK=$RUNTIME_ROOT/logs/ur5e_multitask_pipeline.lock
STATUS=$RUNTIME_ROOT/logs/ur5e_sigreg_protocol_v2.status
WAIT_SECONDS=${SIGREG_V2_WAIT_SECONDS:-300}
export STABLEWM_HOME=$RUNTIME_ROOT
export PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$RUNTIME_ROOT/logs"
status() {
  local value=$1 tmp
  tmp=$(mktemp "$STATUS.tmp.XXXXXX")
  printf '%s time=%s pid=%s\n' "$value" "$(date --iso-8601=seconds)" "$$" > "$tmp"
  mv "$tmp" "$STATUS"
}
fail() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    status "SIGREG_V2_FAILED exit_code=$rc"
  fi
}
trap fail EXIT

exec 9>"$LOCK"
while ! flock -n 9; do
  status SIGREG_V2_WAITING_FOR_PIPELINE
  sleep "$WAIT_SECONDS"
done

while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -q '[0-9]'; do
  status SIGREG_V2_WAITING_FOR_GPU
  flock -u 9
  sleep "$WAIT_SECONDS"
  while ! flock -n 9; do
    status SIGREG_V2_WAITING_FOR_PIPELINE
    sleep "$WAIT_SECONDS"
  done
done

status SIGREG_V2_RUNNING

TASK1_DIR=$RUNTIME_ROOT/full/cube_double_task1_single_pnp
TASK1_RUN=ur5e_cube_double_task1_single_pnp_pcd_ms2_10ep
TASK1_CKPT=$RUNTIME_ROOT/checkpoints/$TASK1_RUN/weights_epoch_10.pt
TASK1_REPORT=$RUNTIME_ROOT/data/cube_double_task1_single_pnp/reports/checkpoint_epoch_10_protocol_v2.json
TASK2_DIR=$RUNTIME_ROOT/full/cube_double_task2_double_pnp1
TASK2_RUN=ur5e_cube_double_task2_double_pnp1_pcd_ms2_10ep
TASK2_CKPT=$RUNTIME_ROOT/checkpoints/$TASK2_RUN/weights_epoch_10.pt
TASK2_REPORT=$RUNTIME_ROOT/data/cube_double_task2_double_pnp1/reports/checkpoint_epoch_10_protocol_v2.json

validate_report() {
  local report=$1 expected=$2
  "$PY" - "$report" "$expected" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
expected = int(sys.argv[2])
assert report["status"] == "PASS"
assert report["protocol_version"] == 2
assert report["examples"] == expected
assert report["sampling"]["retained_examples"] == expected
assert report["sampling"]["min_unique_episodes_per_batch"] > 1
assert report["batch_norm"]
for key in ("mean_prediction_loss", "mean_sigreg_loss"):
    assert math.isfinite(report[key])
for key in (
    "mean_abs_feature_mean",
    "mean_feature_variance",
    "covariance_effective_rank",
):
    assert math.isfinite(report["latent"][key])
assert all(item["finite"] for item in report["batch_norm"])
PY
}

run_task1() {
  local before=$TASK1_REPORT.checkpoint_sha256.before
  local after=$TASK1_REPORT.checkpoint_sha256.after
  mkdir -p "$(dirname "$TASK1_REPORT")"
  sha256sum "$TASK1_CKPT" > "$before"
  if [ ! -s "$TASK1_REPORT" ]; then
    "$PY" "$ROOT/scripts/checks/check_ur5e_holdout_checkpoint.py" \
      --train-dataset "$TASK1_DIR/train.h5" \
      --holdout-dataset "$TASK1_DIR/holdout.h5" \
      --checkpoint-run "$TASK1_RUN" --epoch 10 --seed 3072 --batch-size 128 \
      --report "$TASK1_REPORT"
  fi
  validate_report "$TASK1_REPORT" 10112
  sha256sum "$TASK1_CKPT" > "$after"
  cmp "$before" "$after"
}

run_task2() {
  local before=$TASK2_REPORT.checkpoint_sha256.before
  local after=$TASK2_REPORT.checkpoint_sha256.after
  mkdir -p "$(dirname "$TASK2_REPORT")"
  sha256sum "$TASK2_CKPT" > "$before"
  if [ ! -s "$TASK2_REPORT" ]; then
    "$PY" "$ROOT/scripts/checks/check_ur5e_holdout_checkpoint.py" \
      --train-dataset "$TASK2_DIR/train.h5" \
      --holdout-dataset "$TASK2_DIR/holdout.h5" \
      --checkpoint-run "$TASK2_RUN" --epoch 10 --seed 3072 --batch-size 128 \
      --report "$TASK2_REPORT"
  fi
  validate_report "$TASK2_REPORT" 20096
  sha256sum "$TASK2_CKPT" > "$after"
  cmp "$before" "$after"
}

run_task1
run_task2
status SIGREG_V2_ACCEPTED
