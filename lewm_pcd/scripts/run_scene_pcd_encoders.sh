#!/usr/bin/env bash
# Encoder ablation of the 3D LeWM on OGBench scene (results_ogbench.md): the other point-cloud encoders of
# config/train/model/encoder (ENCS), trained and scored exactly as scripts/run_scene_pipeline.sh trains and scores the default
# 3D model (DexArt PointNet, checkpoints/scene_pcd_dexart):
#   1/2 trainings    data=scene_pcd model=lewm_pcd model/encoder=<ENC>, EPOCHS epochs (config/train/lewm.yaml)
#                    -> checkpoints/scene_pcd_<ENC>; ENC ms2 = ManiSkill PointNet, ppt = PointPatchTransformer,
#                    pointnet2 = PointNet++, dgcnn = DGCNN, the
#                    rest of the model, the data and the training as the DexArt run; both at once, each started
#                    through scripts/mem_gate.py (keep `mem_gate.py watch` and TMP/ops/cache_guard.py running)
#   2/2 CortexBench  bench/cortexbench/scene_grid.py with conf/model/lewm_scene_pcd_<ENC>.yaml (epoch 10):
#                    scene-task1..5 x seeds 100 / 200 / 300, PROPRIO hand6, the encoders served by enc_server.py;
#                    runs under logs/cortexbench/scene/runs_hand6, table by summarize_scene.py next to the DexArt run
#                    (lewm_scene_pcd)
# Idempotent and restartable: a finished training / CortexBench run is skipped, a running one (pid files under logs/)
# waited for. Instances with different ENCS can run at the same time (their own server socket and pid files; the
# gate keeps their trainings within the memory and GPU budget). Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> CB_PY=<cortexbench python> setsid nohup \
#     bash scripts/run_scene_pcd_encoders.sh > $STABLEWM_HOME/logs/scene_pcd_encoders.log 2>&1 < /dev/null &
#   (ENCS="pointnet2" ... > $STABLEWM_HOME/logs/scene_pcd_encoders_pointnet2.log ... for another instance)
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
ENCS=${ENCS:-ms2 ppt}; EPOCHS=${EPOCHS:-10}
SEEDS=${SEEDS:-100,200,300}; PROPRIO=${PROPRIO:-hand6}; SLOTS=${SLOTS:-10}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
MG=$PWD/scripts/mem_gate.py
# the gate's memory (GiB of anon + shmem at most) and GPU (MiB) per training: the peaks of a 40-batch run of each
# encoder on scene_pcd with 6 workers (ms2 23.8 GiB / 14114 MiB, ppt 24.1 GiB / 9762 MiB, pointnet2 24.2 GiB /
# 8922 MiB, dgcnn 24.3 GiB / 24378 MiB) plus some room;
# CortexBench as run_scene_pipeline.sh (scene_grid.py with its SLOTS jobs ~16 GiB, the encoder server ~1 GiB)
declare -A MEM=([ms2]=26 [ppt]=26 [pointnet2]=26 [dgcnn]=26) GPU=([ms2]=15000 [ppt]=11000 [pointnet2]=10000 [dgcnn]=26000)
MEM_GRID=16
LOGS=$STABLEWM_HOME/logs
mkdir -p "$LOGS/train" "$LOGS/cortexbench"
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

# -------------------------------------------------------------------------------------------------- 1 trainings
log "1/2 trainings: $(for e in $ENCS; do printf 'scene_pcd_%s ' "$e"; done)($EPOCHS epochs)"
train() {  # encoder
  local e=$1 r=scene_pcd_$1
  local ck=$STABLEWM_HOME/checkpoints/$r/weights_epoch_$EPOCHS.pt pidf=$LOGS/train/$r.pid
  [ -f "$ck" ] && { log "$r already trained"; return 0; }
  if alive "$pidf"; then
    log "$r: training running (pid $(cat "$pidf")), waiting"
  else
    log "training $r: waiting for the gate (${MEM[$e]} GiB, ${GPU[$e]} MiB GPU)"
    gate "$r" "${MEM[$e]}" "${GPU[$e]}" "$pidf" "$LOGS/train/$r.log" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 \
      $PY train.py data=scene_pcd model=lewm_pcd model/encoder="$e" subdir="$r" output_model_name="$r" \
      trainer.max_epochs=$EPOCHS num_workers=6 || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  [ -f "$ck" ] && log "trained $r" || { log "training $r FAILED (see $LOGS/train/$r.log)"; return 1; }
}
pids=()
for e in $ENCS; do train "$e" & pids+=($!); sleep 5; done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done
[ $st = 0 ] || exit 1

# ----------------------------------------------------------------------------------------------- 2 CortexBench
MODELS=$(for e in $ENCS; do printf 'lewm_scene_pcd_%s,' "$e"; done); MODELS=${MODELS%,}
log "2/2 CortexBench: $MODELS x scene-task1..5 x seeds $SEEDS, proprioception $PROPRIO"
RUNS=scene/runs_$([ "$PROPRIO" = hand6 ] && echo hand6 || echo noproprio)  # scene_grid.py's run root under logs/cortexbench
(
  CBPY=$CB_PY  # the gate itself runs on $PY (the lewm_mw python; standard library only)
  TAG=lewm_scene_pcd_encoders$([ "$ENCS" = "ms2 ppt" ] || echo "_${ENCS// /_}")  # the first instance's names as before
  export MW_PY=$PY ENC_SOCKET=@${TAG}_enc
  CBD_=$PWD/bench/cortexbench
  srv=$LOGS/cortexbench/${TAG}_enc_server.pid grid=$LOGS/cortexbench/${TAG}_grid.pid
  last=${MODELS##*,}
  if ! alive "$srv"; then
    gate "${TAG}_enc_server" 2 3000 "$srv" "$LOGS/cortexbench/${TAG}_enc_server.log" \
      --cwd "$CBD_" -- bash -c 'PY=$0; source env.sh && exec "$PY" enc_server.py "$@"' "$CBPY" "$ENC_SOCKET" "$MODELS" \
      || exit 1
    until grep -q "loaded $last" "$LOGS/cortexbench/${TAG}_enc_server.log"; do
      alive "$srv" || { log "enc_server FAILED (see $LOGS/cortexbench/${TAG}_enc_server.log)"; exit 1; }
      sleep 5
    done
  fi
  if ! alive "$grid"; then
    gate "${TAG}_grid" "$MEM_GRID" 0 "$grid" "$LOGS/cortexbench/scene_grid.log" --cwd "$CBD_" -- \
      bash -c 'PY=$0; source env.sh && exec "$PY" scene_grid.py "$@"' "$CBPY" --models "$MODELS" \
      --seeds "$SEEDS" --proprio "$PROPRIO" --slots "$SLOTS" || exit 1
  fi
  while alive "$grid"; do sleep 60; done
  kill "$(cat "$srv")"
  # judged by the DONE files (scene_grid.log is appended to by every attempt)
  missing=0
  for m in ${MODELS//,/ }; do for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$LOGS/cortexbench/$RUNS/$m/scene-task$k-v0/s$s/DONE" ] || missing=$((missing + 1))
  done; done; done
  [ $missing = 0 ] || { log "CortexBench FAILED: $missing runs without DONE (see scene_grid.log)"; exit 1; }
  cd "$CBD_" && export PY=$CBPY && source env.sh && $PY summarize_scene.py lewm_scene_pcd ${MODELS//,/ } --seeds "$SEEDS" \
    --runs "$RUNS"
) || exit 1
log "ALL_DONE"
