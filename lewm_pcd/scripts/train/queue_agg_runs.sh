#!/usr/bin/env bash
# Launch the 4D (aggregated) trainings one by one as GPU memory frees up, never competing with the
# running 3D trainings:  needs >= GPU_MIN_FREE_MB free VRAM and cgroup anon < ANON_MAX_GB at launch,
# then waits LAUNCH_GAP_S before considering the next one (so its memory is visible).
#   setsid nohup bash scripts/train/queue_agg_runs.sh > $STABLEWM_HOME/logs/train/queue_agg.log 2>&1 < /dev/null &
set -uo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
QUEUE=${QUEUE:-"ppt dexart ms2"}          # encoder order
GPU_MIN_FREE_MB=${GPU_MIN_FREE_MB:-12000}; ANON_MAX_GB=${ANON_MAX_GB:-40}; LAUNCH_GAP_S=${LAUNCH_GAP_S:-600}
log() { echo "$(date +%F_%T) $*"; }
gpu_free_mb() { local t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); echo $(( ${t:-0} - ${u:-0} )); }
anon_gb() { local v=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | awk '{print $2}'); echo $(( ${v:-0} / 1073741824 )); }
cd "$P"
for enc in $QUEUE; do
  run=cube_pcd_agg_$enc
  if [ -e "$STABLEWM_HOME/checkpoints/$run" ]; then log "$run exists, skipping"; continue; fi
  while [ "$(gpu_free_mb)" -lt "$GPU_MIN_FREE_MB" ] || [ "$(anon_gb)" -ge "$ANON_MAX_GB" ]; do sleep 300; done
  log "launching $run (gpu free $(gpu_free_mb) MB, anon $(anon_gb) GB)"
  bash scripts/train/run_train_detached.sh lewm_pcd_agg "$run" --config-name=lewm_agg model/encoder=$enc
  sleep "$LAUNCH_GAP_S"
done
log "queue done"
