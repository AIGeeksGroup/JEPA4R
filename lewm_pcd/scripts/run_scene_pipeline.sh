#!/usr/bin/env bash
# OGBench scene from collection to CortexBench scores (results_ogbench.md):
#   1/4 collection   scripts/data/run_scene_collection.sh: the four object tasks (envs/ogb_scene_env.py: cube, button,
#                    drawer, window), N episodes each of 200 steps with MetaWorld's three-part plan
#                    -> datasets/ogbench/<NAME>{,_pcd}.h5
#   2/4 trainings    EPOCHS epochs each (config/train/lewm.yaml), one model for the whole scene:
#                    2D  official LeWM (ViT-tiny)            data=<DATA>     model=lewm            -> checkpoints/<RUN>_pix
#                    3D  point-cloud LeWM (DexArt PointNet)  data=<DATA>_pcd model=lewm_pcd_dexart -> checkpoints/<RUN>_pcd_dexart
#                    each, like the collection and the CortexBench lane, started through scripts/mem_gate.py: only
#                    when the container's memory (anon + shmem) with the reservations of the jobs still starting and
#                    its own stays within the budget (72 GiB; the container throttles everything above 88 GiB) and the
#                    GPU has room for it; one lock with the other OGBench pipelines. Keep `mem_gate.py watch` running
#                    alongside (it pauses the newest job when the memory still runs high, see mem_gate.py)
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
# Keep TMP/ops/cache_guard.py (page cache) and scripts/mem_gate.py watch running alongside (the container's memory cap,
# see README).
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
N=${N:-2500}; SHARDS=${SHARDS:-6}; MAXPAR=${MAXPAR:-12}; NAME=${NAME:-scene_wm10000}
DATA=${DATA:-scene}; RUN=${RUN:-scene}; ENC=${ENC:-lewm_scene}; EPOCHS=${EPOCHS:-10}
SEEDS=${SEEDS:-100,200,300}; PROPRIO=${PROPRIO:-hand6}; SLOTS=${SLOTS:-10}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
MG=$PWD/scripts/mem_gate.py
# each job's memory (GiB of anon + shmem at most) and GPU (MiB) for the gate: 2D training 36.9 GiB measured
# (puzzle-4x4's: the prefetched batches of 8 workers in shared memory + the process); 3D training at most 15 (the memory
# rose 3 -> 18 GiB while puzzle-3x3's started); collection ~1.4 GiB a running shard; CortexBench: scene_grid.py with
# its SLOTS jobs ~16 GiB (measured with 10 slots), the encoder server ~1 GiB
MEM_2D=38 MEM_3D=15 MEM_GRID=16
DS=$STABLEWM_HOME/datasets/ogbench
LOGS=$STABLEWM_HOME/logs
mkdir -p "$LOGS/train" "$LOGS/collect" "$LOGS/cortexbench"
log() { echo "$(date +%F_%T) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
gate() {  # name mem_gib gpu_mib pidfile log [--cwd dir] -- cmd...: cmd started detached (its own session, output
          # appended to log, pid in pidfile) once the memory budget and the GPU have room (mem_gate.py admit, blocks)
  local name=$1 mem=$2 gpu=$3 pidf=$4 out=$5; shift 5
  rm -f "$pidf"
  $PY "$MG" admit --name "$name" --mem "$mem" --gpu "$gpu" --pidfile "$pidf" --log "$out" "$@" 2>&1 \
    | while IFS= read -r l; do log "$l"; done
  [ -s "$pidf" ] || { log "$name not started (see above)"; return 1; }
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
    gate "collect_$NAME" $((MAXPAR * 3 / 2 + 1)) 0 "$pidf" "$LOGS/collect/$NAME.log" -- env N=$N SHARDS=$SHARDS \
      MAXPAR=$MAXPAR NAME=$NAME bash scripts/data/run_scene_collection.sh || exit 1
  fi
  while alive "$pidf"; do sleep 60; done
  grep -q ALL_DONE "$LOGS/collect/$NAME.log" || { log "collection FAILED (see $LOGS/collect/$NAME.log)"; exit 1; }
  log "collection done"
fi

# -------------------------------------------------------------------------------------------------- 2 trainings
log "2/4 trainings: ${RUN}_pix, ${RUN}_pcd_dexart ($EPOCHS epochs)"
train() {  # run mem_gib gpu_mib workers overrides...
  local r=$1 mem=$2 gpu=$3 w=$4; shift 4
  local ck=$STABLEWM_HOME/checkpoints/$r/weights_epoch_$EPOCHS.pt pidf=$LOGS/train/$r.pid
  [ -f "$ck" ] && { log "$r already trained"; return 0; }
  if alive "$pidf"; then
    log "$r: training running (pid $(cat "$pidf")), waiting"
  else
    log "training $r: waiting for the gate ($mem GiB, $gpu MiB GPU)"
    gate "$r" "$mem" "$gpu" "$pidf" "$LOGS/train/$r.log" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 \
      $PY train.py "$@" subdir="$r" output_model_name="$r" trainer.max_epochs=$EPOCHS num_workers="$w" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  [ -f "$ck" ] && log "trained $r" || { log "training $r FAILED (see $LOGS/train/$r.log)"; return 1; }
}
train "${RUN}_pcd_dexart" "$MEM_3D" 4500 6 data="${DATA}_pcd" model=lewm_pcd_dexart & p3d=$!
sleep 5
train "${RUN}_pix" "$MEM_2D" 15000 8 data="$DATA" model=lewm & p2d=$!
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
  CBPY=$CB_PY  # the gate itself runs on $PY (the lewm_mw python; standard library only)
  export MW_PY=$PY ENC_SOCKET=@${ENC}_enc
  CBD_=$PWD/bench/cortexbench
  srv=$LOGS/cortexbench/${ENC}_enc_server.pid grid=$LOGS/cortexbench/${ENC}_scene_grid.pid
  if ! alive "$srv"; then
    gate "${ENC}_enc_server" 2 3000 "$srv" "$LOGS/cortexbench/${ENC}_enc_server.log" --cwd "$CBD_" -- \
      bash -c 'PY=$0; source env.sh && exec "$PY" enc_server.py "$@"' "$CBPY" "$ENC_SOCKET" "${ENC}_pix,${ENC}_pcd" \
      || exit 1
    until grep -q "loaded ${ENC}_pcd" "$LOGS/cortexbench/${ENC}_enc_server.log"; do
      alive "$srv" || { log "enc_server FAILED (see $LOGS/cortexbench/${ENC}_enc_server.log)"; exit 1; }
      sleep 5
    done
  fi
  if ! alive "$grid"; then
    gate "${ENC}_scene_grid" "$MEM_GRID" 0 "$grid" "$LOGS/cortexbench/scene_grid.log" --cwd "$CBD_" -- \
      bash -c 'PY=$0; source env.sh && exec "$PY" scene_grid.py "$@"' "$CBPY" --models "${ENC}_pcd,${ENC}_pix" \
      --seeds "$SEEDS" --proprio "$PROPRIO" --slots "$SLOTS" || exit 1
  fi
  while alive "$grid"; do sleep 60; done
  kill "$(cat "$srv")"
  # judged by the DONE files (scene_grid.log is appended to by every attempt)
  missing=0
  for m in "${ENC}_pcd" "${ENC}_pix"; do for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$LOGS/cortexbench/$RUNS/$m/scene-task$k-v0/s$s/DONE" ] || missing=$((missing + 1))
  done; done; done
  [ $missing = 0 ] || { log "CortexBench FAILED: $missing runs without DONE (see scene_grid.log)"; exit 1; }
  cd "$CBD_" && export PY=$CBPY && source env.sh && $PY summarize_scene.py "${ENC}_pcd" "${ENC}_pix" --seeds "$SEEDS" \
    --runs "$RUNS"
) || exit 1
log "ALL_DONE"
