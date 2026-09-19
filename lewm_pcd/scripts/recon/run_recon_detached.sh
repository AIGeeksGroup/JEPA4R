#!/usr/bin/env bash
# Reconstruction decoder on a frozen world model, detached from the terminal / Claude session (setsid + nohup).
#   bash scripts/recon/run_recon_detached.sh <world-model checkpoint> <run name> [extra hydra overrides]
#   e.g. bash scripts/recon/run_recon_detached.sh cube_pcd_dexart/weights_epoch_10.pt cube_pcd_dexart_point
#        DECODER=gaussian bash scripts/recon/run_recon_detached.sh cube_pcd_agg_dexart/weights_epoch_10.pt cube_pcd_agg_dexart_gaussian
#        DATA=ogb_pix_pcd bash scripts/recon/run_recon_detached.sh cube_pix/weights_epoch_10.pt cube_pix_point data.target=<paired point clouds>.h5
# Decoders: $STABLEWM_HOME/checkpoints/recon/<run name>/ (config.yaml + weights_epoch_N.pt); logs: $STABLEWM_HOME/logs/recon/
set -euo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
PY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python   # CUDA training env
CKPT=$1; RUN=$2; shift 2
DATA=${DATA:-ogb_pcd}                                # ogb_pix_pcd for pixel checkpoints
DECODER=${DECODER:-point}                            # or gaussian
if [ -e "$STABLEWM_HOME/checkpoints/recon/$RUN" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "run $RUN already exists under $STABLEWM_HOME/checkpoints/recon (would overwrite its checkpoints); pick another name or FORCE=1"; exit 1
fi
mkdir -p "$STABLEWM_HOME/logs/recon"
LOG=$STABLEWM_HOME/logs/recon/launch_${RUN}_$(date +%Y%m%d_%H%M%S).log
cd "$P"
setsid nohup "$PY" train_recon.py ckpt="$CKPT" data="$DATA" decoder="$DECODER" subdir="$RUN" output_model_name="$RUN" "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched $RUN (ckpt=$CKPT decoder=$DECODER) pid $! log $LOG"
