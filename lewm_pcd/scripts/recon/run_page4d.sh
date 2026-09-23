#!/usr/bin/env bash
# PAGE-4D reconstruction head on a frozen world model: both stages, then the evaluation and the videos.
#   stage A  <name>_frozen  PAGE-4D frozen, only the token generator / output frame / rgb branch train
#   stage B  <name>_lora    + rank-4 LoRA on q / v of every attention block, the new parts start from stage A (LP-FT order)
#   then on stage B: eval_frames (validation frames), eval_rollout (open loop), videos of $EPISODES (recon + openloop) + gifs
#     bash scripts/recon/run_page4d.sh <world-model ckpt> <dataset> <name> [hydra overrides for both stages]
#     e.g. bash scripts/recon/run_page4d.sh cube_rm75_pcd_dexart/weights_epoch_10.pt ogbench/cube_rm75_single_expert_pcd.h5 rm75_1cam_page4d
# ckpt / dataset are relative to $STABLEWM_HOME (default: lewm_pcd_data/ next to lewm_pcd/, e.g. a worktree's own).
# Outputs: $STABLEWM_HOME/checkpoints/recon/<name>_{frozen,lora}/, logs in $STABLEWM_HOME/logs/recon/. Python: $PY (default python).
# A stage whose weights exist is skipped, so a rerun resumes. Detached:
#   setsid nohup bash scripts/recon/run_page4d.sh ... > <log> 2>&1 < /dev/null &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)                               # lewm_pcd
export STABLEWM_HOME=${STABLEWM_HOME:-$(cd "$ROOT/.." && pwd)/lewm_pcd_data}
export SPT_CACHE_DIR=${SPT_CACHE_DIR:-$STABLEWM_HOME/checkpoints/_spt_cache} PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH} \
       MUJOCO_GL=egl HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
[ $# -ge 3 ] || { sed -n '2,11p' "$0"; exit 1; }
CKPT=$1; DATA=$2; NAME=$3; shift 3
EPISODES=${EPISODES:-126,3,50}
RECON=$STABLEWM_HOME/checkpoints/recon; LOGS=$STABLEWM_HOME/logs/recon; mkdir -p "$LOGS"
log() { echo "$(date +%F_%T) $*"; }
cd "$ROOT"
log "home $STABLEWM_HOME, world model $CKPT, data $DATA, runs ${NAME}_frozen -> ${NAME}_lora"

stage() {  # <run> <exp> [overrides]
  local run=$1 exp=$2; shift 2
  if [ -e "$RECON/$run/weights_epoch_1.pt" ]; then log "$run exists, skipped"; return 0; fi
  log "training $run (exp=$exp), log $LOGS/train_$run.log"
  "$PY" train_recon.py ckpt="$CKPT" data.input="$DATA" exp="$exp" subdir="$run" output_model_name="$run" "$@" > "$LOGS/train_$run.log" 2>&1 < /dev/null
  [ -e "$RECON/$run/weights_epoch_1.pt" ] || { log "$run FAILED, see $LOGS/train_$run.log"; exit 1; }
  log "$run done"
}
stage "${NAME}_frozen" frozen "$@"
stage "${NAME}_lora" lora decoder.init_from="recon/${NAME}_frozen/weights_epoch_1.pt" "$@"

RUN=${NAME}_lora
log "eval_frames";  "$PY" scripts/recon/eval_frames.py --recon "$RUN" > "$LOGS/eval_frames_$RUN.log" 2>&1 || log "eval_frames FAILED"
log "eval_rollout"; "$PY" scripts/recon/eval_rollout.py --recon "$RUN" > "$LOGS/eval_rollout_$RUN.log" 2>&1 || log "eval_rollout FAILED"
for mode in recon openloop; do
  log "videos: $mode"
  "$PY" scripts/recon/render_episodes.py --recon "$RUN" --episodes "$EPISODES" --mode $mode > "$LOGS/render_${RUN}_$mode.log" 2>&1 || log "render $mode FAILED"
done
for f in "$RECON/$RUN/render"/*.mp4; do
  case "$f" in *_fs.mp4) ;; *) PY=$PY bash scripts/recon/to_gif.sh "$f" > /dev/null 2>&1 || log "gif of $f FAILED";; esac
done
log "done: $RECON/$RUN (eval_frames/, eval_rollout/, render/)"
