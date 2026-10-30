#!/usr/bin/env bash
# Full training run, detached from the terminal / Claude session (setsid + nohup).
#   bash scripts/train/run_train_detached.sh <model config> <run name> [extra hydra overrides]
#   e.g. bash scripts/train/run_train_detached.sh lewm_pcd cube_pcd_ms2
# Checkpoints: $STABLEWM_HOME/checkpoints/<run name>/ (config.yaml + weights_epoch_N.pt); logs: $STABLEWM_HOME/logs/train/
set -euo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
PY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python   # CUDA training env
MODEL=$1; RUN=$2; shift 2
CONFIG_NAME=lewm; ARGS=()
for a in "$@"; do case "$a" in --config-name=*) CONFIG_NAME=${a#--config-name=};; *) ARGS+=("$a");; esac; done   # e.g. --config-name=lewm_agg for the 4D model
DATA=${DATA:-ogb_pcd}                                # e.g. DATA=ogb for the pixel baseline
if [ -e "$STABLEWM_HOME/checkpoints/$RUN" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "run $RUN already exists under $STABLEWM_HOME/checkpoints (would overwrite its checkpoints); pick another name or FORCE=1"; exit 1
fi
mkdir -p "$STABLEWM_HOME/logs/train"
LOG=$STABLEWM_HOME/logs/train/launch_${RUN}_$(date +%Y%m%d_%H%M%S).log
cd "$P"
setsid nohup "$PY" train.py --config-name="$CONFIG_NAME" data="$DATA" model="$MODEL" subdir="$RUN" output_model_name="$RUN" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo "launched $RUN (model=$MODEL) pid $! log $LOG"
