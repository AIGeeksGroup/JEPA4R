#!/usr/bin/env bash
# Resume a crashed / killed training from its last Lightning checkpoint (full state: weights, optimizer,
# LR schedule, epoch counter -- saved by the Trainer at the end of every epoch under $SPT_CACHE_DIR/runs/).
#   bash scripts/train/resume_train_detached.sh <model config> <run name> [extra hydra overrides]
# Use the same arguments as the original run_train_detached.sh launch, e.g.
#   bash scripts/train/resume_train_detached.sh lewm_pcd     cube_pcd_ms2
#   bash scripts/train/resume_train_detached.sh lewm_pcd_agg cube_pcd_agg_dexart --config-name=lewm_agg model/encoder=dexart
#   bash scripts/train/resume_train_detached.sh lewm_pcd_agg cube_pcd_agg_ms2    --config-name=lewm_agg model/encoder=ms2
# It copies the newest last.ckpt whose hparams.yaml (system.working_dir) names this run to $STABLEWM_HOME/checkpoints/<run>/<run>_weights.ckpt
# (train.py hands that path to spt.Manager as ckpt_path) and relaunches through run_train_detached.sh with FORCE=1.
# Training continues at the next epoch; per-epoch weights_epoch_N.pt keep their numbering. Logging starts a new
# lightning_logs / _spt_cache run directory (metrics of a resumed run are split across two event files).
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)  # lewm_pcd
source "$P/env.sh"
MODEL=$1; RUN=$2; shift 2
CACHE=${SPT_CACHE_DIR:-$STABLEWM_HOME/checkpoints/_spt_cache}
if pgrep -f "subdir=${RUN}[ ]" > /dev/null || pgrep -f "subdir=${RUN}$" > /dev/null; then echo "run $RUN still has a live process; not resuming"; exit 1; fi
[ -d "$STABLEWM_HOME/checkpoints/$RUN" ] || { echo "no run dir $STABLEWM_HOME/checkpoints/$RUN"; exit 1; }
# the Lightning cache only records the hydra run dir, whose name ends with _<run> (or is <run>)
CKPT=$(grep -lsE "^system.working_dir: .*[/_]${RUN}$" "$CACHE"/runs/*/*/*/hparams.yaml 2>/dev/null | while read h; do d=$(dirname "$h"); [ -f "$d/checkpoints/last.ckpt" ] && printf "%s %s\n" "$(stat -c %Y "$d/checkpoints/last.ckpt")" "$d/checkpoints/last.ckpt"; done | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "no last.ckpt found for $RUN under $CACHE/runs"; exit 1; }
echo "resuming $RUN from $CKPT ($(date -r "$CKPT" +%F_%T), $(du -h "$CKPT" | cut -f1))"
cp "$CKPT" "$STABLEWM_HOME/checkpoints/$RUN/${RUN}_weights.ckpt"
FORCE=1 bash "$P/scripts/train/run_train_detached.sh" "$MODEL" "$RUN" "$@"
