#!/usr/bin/env bash
# OGBench scene from collection to CortexBench scores (results_ogbench.md):
#   1/4 collection   scripts/data/run_scene_collection.sh: the four object tasks (envs/ogb_scene_env.py: cube, button,
#                    drawer, window), N episodes each of 200 steps with MetaWorld's three-part plan
#                    -> datasets/ogbench/<NAME>{,_pcd}.h5
#   2/4 trainings    EPOCHS epochs each (config/train/lewm.yaml), one model for the whole scene:
#                    2D  official LeWM (ViT-tiny)            data=<DATA>     model=lewm            -> checkpoints/<RUN>_pix
#                    3D  point-cloud LeWM (DexArt PointNet)  data=<DATA>_pcd model=lewm_pcd_dexart -> checkpoints/<RUN>_pcd_dexart
#                    each starts once the GPU has room for it (3D 4.5 GB, 2D 15 GB) and, for the 2D, the container's
#                    unreclaimable memory (anon + shmem) is below 50 GiB: the 2D training holds ~25 GB of prefetched
#                    batches (8 workers x prefetch 3) against the 90 GiB cap; the gate is shared with
#                    run_puzzle_pipeline.sh (a lock file), so two trainings never take the same free memory
#   3/4 demos        bench/cortexbench/make_scene_demos.py: 25 expert demos per evaluation task (2D and 3D, independent
#                    of the world model; skipped when they exist)
#   4/4 CortexBench  bench/cortexbench/scene_grid.py with both encoders (conf/model/<ENC>_{pix,pcd}.yaml, epoch 10):
#                    scene-task1..5 x seeds 100 / 200 / 300, PROPRIO (hand6: effector xyz, cos / sin yaw, gripper),
#                    the encoders served by bench/cortexbench/enc_server.py; runs under logs/cortexbench/scene/
#                    (runs_hand6, runs_noproprio), table by summarize_scene.py
# Defaults = the full scale: 4 x 2500 episodes (2 M rows, MetaWorld's ~2 M per model as run_tf_pipeline.sh), 10 epochs.
# The small run of results_ogbench.md: N=250 SHARDS=5 NAME=scene_wm DATA=scene_small RUN=scene_small ENC=lewm_scene_small.
# Idempotent and restartable: finished steps are skipped, a running collection / training (pid files under logs/) is
# waited for instead of started again. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> CB_PY=<cortexbench python> setsid nohup \
#     bash scripts/run_scene_pipeline.sh > $STABLEWM_HOME/logs/scene_pipeline.log 2>&1 < /dev/null &
# Keep TMP/ops/cache_guard.py running alongside (the container's memory cap, see README).
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
N=${N:-2500}; SHARDS=${SHARDS:-6}; MAXPAR=${MAXPAR:-12}; NAME=${NAME:-scene_wm10000}
DATA=${DATA:-scene}; RUN=${RUN:-scene}; ENC=${ENC:-lewm_scene}; EPOCHS=${EPOCHS:-10}
SEEDS=${SEEDS:-100,200,300}; PROPRIO=${PROPRIO:-hand6}; SLOTS=${SLOTS:-10}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
DS=$STABLEWM_HOME/datasets/ogbench
LOGS=$STABLEWM_HOME/logs
mkdir -p "$LOGS/train" "$LOGS/collect" "$LOGS/cortexbench"
log() { echo "$(date +%F_%T) $*"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
fixed_gib() { awk '$1 == "anon" || $1 == "shmem" {s += $2} END {print int(s / 2^30)}' /sys/fs/cgroup/memory.stat; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
detach() {  # pidfile log cmd...: run cmd in its own session (survives this script), its pid in pidfile
  local pidf=$1 out=$2; shift 2
  rm -f "$pidf"
  # fd 9 (the GPU gate's lock, see train) closed: a detached training must not hold the gate for its whole run
  setsid nohup bash -c 'echo $$ > "$0"; exec "$@"' "$pidf" "$@" > "$out" 2>&1 < /dev/null 9>&- &
  until [ -s "$pidf" ]; do sleep 1; done
}

# ------------------------------------------------------------------------------------------------- 1 collection
if [ -f "$DS/$NAME.h5" ] && [ -f "$DS/${NAME}_pcd.h5" ] && ! alive "$LOGS/collect/$NAME.pid"; then
  log "1/4 collection: datasets exist"
else
  pidf=$LOGS/collect/$NAME.pid
  if alive "$pidf"; then
    log "1/4 collection running (pid $(cat "$pidf")), waiting"
  else
    log "1/4 collection: $N episodes x 4 tasks"
    N=$N SHARDS=$SHARDS MAXPAR=$MAXPAR NAME=$NAME detach "$pidf" "$LOGS/collect/$NAME.log" bash scripts/data/run_scene_collection.sh
  fi
  while alive "$pidf"; do sleep 60; done
  grep -q ALL_DONE "$LOGS/collect/$NAME.log" || { log "collection FAILED (see $LOGS/collect/$NAME.log)"; exit 1; }
  log "collection done"
fi

# -------------------------------------------------------------------------------------------------- 2 trainings
log "2/4 trainings: ${RUN}_pix, ${RUN}_pcd_dexart ($EPOCHS epochs)"
train() {  # run gpu_mib max_fixed_gib workers overrides...
  local r=$1 need=$2 maxfix=$3 w=$4; shift 4
  local ck=$STABLEWM_HOME/checkpoints/$r/weights_epoch_$EPOCHS.pt pidf=$LOGS/train/$r.pid
  [ -f "$ck" ] && { log "$r already trained"; return 0; }
  if alive "$pidf"; then
    log "$r: training running (pid $(cat "$pidf")), waiting"
  else
    # one GPU gate for both OGBench pipelines (run_puzzle_pipeline.sh, run_scene_pipeline.sh): the check, the start and
    # 10 minutes for the training to take its memory happen under the lock, so no two trainings count the same free memory
    until (
      flock 9
      [ "$(free_mib)" -ge "$need" ] && [ "$(fixed_gib)" -lt "$maxfix" ] || exit 1
      log "training $r ($(free_mib) MiB GPU free, $(fixed_gib) GiB anon + shmem)"
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 detach "$pidf" "$LOGS/train/$r.log" nice -n 5 $PY train.py "$@" subdir="$r" \
        output_model_name="$r" trainer.max_epochs=$EPOCHS num_workers="$w"
      sleep 600
    ) 9> "$LOGS/train/.gpu_gate.lock"; do sleep 120; done
  fi
  while alive "$pidf"; do sleep 60; done
  [ -f "$ck" ] && log "trained $r" || { log "training $r FAILED (see $LOGS/train/$r.log)"; return 1; }
}
train "${RUN}_pcd_dexart" 4500 1000 6 data="${DATA}_pcd" model=lewm_pcd_dexart & p3d=$!
sleep 5
train "${RUN}_pix" 15000 50 8 data="$DATA" model=lewm & p2d=$!
wait $p3d; s3d=$?; wait $p2d; s2d=$?
[ $s2d = 0 ] && [ $s3d = 0 ] || exit 1

# ----------------------------------------------------------------------------------------------------- 3 demos
CBD=$STABLEWM_HOME/datasets/cortexbench
if ls "$CBD"/scene-expert/scene-task{1,2,3,4,5}-v0.pickle "$CBD"/scene-expert-pcd/scene-task{1,2,3,4,5}-v0.pickle > /dev/null 2>&1; then
  log "3/4 demos exist"
else
  log "3/4 demos: 25 per task"
  for k in 1 2 3 4 5; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY bench/cortexbench/make_scene_demos.py $k \
      > "$LOGS/cortexbench/make_scene_demos_task$k.log" 2>&1 &
  done
  wait
  ls "$CBD"/scene-expert/scene-task{1,2,3,4,5}-v0.pickle "$CBD"/scene-expert-pcd/scene-task{1,2,3,4,5}-v0.pickle > /dev/null \
    || { log "demos FAILED (see $LOGS/cortexbench/make_scene_demos_task*.log)"; exit 1; }
  grep -h "demos" "$LOGS"/cortexbench/make_scene_demos_task*.log
fi

# ----------------------------------------------------------------------------------------------- 4 CortexBench
log "4/4 CortexBench: ${ENC}_pix, ${ENC}_pcd x scene-task1..5 x seeds $SEEDS, proprioception $PROPRIO"
RUNS=scene/runs_$([ "$PROPRIO" = hand6 ] && echo hand6 || echo noproprio)  # scene_grid.py's run root under logs/cortexbench
(
  export MW_PY=$PY PY=$CB_PY ENC_SOCKET=@${ENC}_enc
  cd bench/cortexbench && source env.sh
  srv=$LOGS/cortexbench/${ENC}_enc_server.pid
  if ! alive "$srv"; then
    detach "$srv" "$LOGS/cortexbench/${ENC}_enc_server.log" $PY enc_server.py "$ENC_SOCKET" "${ENC}_pix,${ENC}_pcd"
    until grep -q "loaded ${ENC}_pcd" "$LOGS/cortexbench/${ENC}_enc_server.log"; do
      alive "$srv" || { log "enc_server FAILED (see $LOGS/cortexbench/${ENC}_enc_server.log)"; exit 1; }
      sleep 5
    done
  fi
  $PY scene_grid.py --models "${ENC}_pcd,${ENC}_pix" --seeds "$SEEDS" --proprio "$PROPRIO" --slots "$SLOTS" \
    >> "$LOGS/cortexbench/scene_grid.log" 2>&1; st=$?
  kill "$(cat "$srv")"
  # judged by scene_grid.py's exit code and the DONE files (scene_grid.log is appended to by every attempt)
  missing=0
  for m in "${ENC}_pcd" "${ENC}_pix"; do for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$LOGS/cortexbench/$RUNS/$m/scene-task$k-v0/s$s/DONE" ] || missing=$((missing + 1))
  done; done; done
  [ $st = 0 ] && [ $missing = 0 ] \
    || { log "CortexBench FAILED: scene_grid.py exit $st, $missing runs without DONE (see scene_grid.log)"; exit 1; }
  $PY summarize_scene.py "${ENC}_pcd" "${ENC}_pix" --seeds "$SEEDS" --runs "$RUNS"
) || exit 1
log "ALL_DONE"
