#!/usr/bin/env bash
# Companion of scripts/run_pix_pipeline.sh: once the pixel dataset is complete and the 2D run (cube_pix) has been
# launched, also launch the 4D pixel run (cube_pix_agg = lewm_pix_agg with the lewm_agg training config) and
# restart the auto-eval loop so it evaluates both runs. Detached:
#   setsid nohup bash scripts/run_pix_agg_followup.sh > $STABLEWM_HOME/logs/pix_agg_followup.log 2>&1 < /dev/null &
set -uo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)  # lewm_pcd
source "$P/env.sh"
log() { echo "$(date +%F_%T) $*"; }
cd "$P"
until [ -f "$STABLEWM_HOME/checkpoints/cube_pix/config.yaml" ]; do sleep 60; done   # written by train.py at launch
log "cube_pix training is up; launching cube_pix_agg"
DATA=ogb bash scripts/train/run_train_detached.sh lewm_pix_agg cube_pix_agg --config-name=lewm_agg || { log "launch FAILED"; exit 1; }
sleep 30
for p in $(pgrep -f "auto_eval_checkpoint[s].sh"); do kill "$p" && log "stopped eval loop $p (no eval can be running yet: no checkpoints exist)"; done
sleep 2
RUNS="cube_pix cube_pix_agg" setsid nohup bash scripts/eval/auto_eval_checkpoints.sh >> "$STABLEWM_HOME/logs/eval/auto_eval.log" 2>&1 < /dev/null &
log "eval loop restarted for cube_pix cube_pix_agg"
