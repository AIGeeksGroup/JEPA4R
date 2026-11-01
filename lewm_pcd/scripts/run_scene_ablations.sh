#!/usr/bin/env bash
# Scene ablations of the 3D LeWM (DexArt PointNet; results_ogbench.md), one change each against the default run
# scene_pcd_dexart (learning rate 5e-5, batch 128, 3 history frames), everything else as scripts/run_scene_pipeline.sh
# trains (data=scene_pcd model=lewm_pcd_dexart, 10 epochs) and scores it:
#   hyper  learning rate 2.5e-5 / 1e-4; batch size 64 / 256 (learning rate and epochs unchanged)
#   time   history frames 2 / 4 (frameskip 5: windows of 10 / 20 steps; the default 3 = 15)
# Order: once the DGCNN encoder training (scripts/run_scene_pcd_encoders.sh ENCS=dgcnn) has finished, the six trainings,
# two at a time, each started through scripts/mem_gate.py; a group's CortexBench (scene-task1..5 x seeds 100 / 200 /
# 300, proprioception hand6, conf/model/lewm_scene_pcd_<variant>.yaml) as soon as its trainings are done; after the six
# trainings the PointNet++ encoder ablation (scripts/run_scene_pcd_encoders.sh ENCS=pointnet2, its own log).
# Idempotent: finished trainings / CortexBench runs are skipped. Keep `mem_gate.py watch` and TMP/ops/cache_guard.py
# running. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> CB_PY=<cortexbench python> setsid nohup \
#     bash scripts/run_scene_ablations.sh > $STABLEWM_HOME/logs/scene_ablations.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
EPOCHS=10 SEEDS=${SEEDS:-100,200,300} PROPRIO=hand6 SLOTS=${SLOTS:-10} MAX_TRAIN=${MAX_TRAIN:-2}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
MG=$PWD/scripts/mem_gate.py
HYPER="lr2p5e5 lr1e4 bs64 bs256" TIME="hist2 hist4"
declare -A OVR=([lr2p5e5]="optimizer.lr=2.5e-5" [lr1e4]="optimizer.lr=1e-4" [bs64]="loader.batch_size=64"
                [bs256]="loader.batch_size=256" [hist2]="history_size=2" [hist4]="history_size=4")
# the gate's memory (GiB, anon + shmem) and GPU (MiB) per training: the default-size encoders' 40-batch runs on scene_pcd
# peaked at about 24 GiB with 6 workers; the larger batch / longer history with room in proportion
declare -A MEM=([lr2p5e5]=24 [lr1e4]=24 [bs64]=24 [bs256]=36 [hist2]=24 [hist4]=30)
declare -A GPU=([lr2p5e5]=6000 [lr1e4]=6000 [bs64]=6000 [bs256]=10000 [hist2]=6000 [hist4]=8000)
MEM_GRID=16
LOGS=$STABLEWM_HOME/logs CK=$STABLEWM_HOME/checkpoints
mkdir -p "$LOGS/train" "$LOGS/cortexbench"
log() { echo "$(date +%F_%T) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
gate() {  # name mem_gib gpu_mib pidfile log [--cwd dir] -- cmd...: started detached once memory and GPU have room
  local name=$1 mem=$2 gpu=$3 pidf=$4 out=$5; shift 5
  rm -f "$pidf"
  $PY "$MG" admit --name "$name" --mem "$mem" --gpu "$gpu" --pidfile "$pidf" --log "$out" "$@" 2>&1 \
    | while IFS= read -r l; do log "$l"; done
  [ -s "$pidf" ] || { log "$name not started (see above)"; return 1; }
}
trained() { [ -f "$CK/scene_pcd_dexart_$1/weights_epoch_$EPOCHS.pt" ]; }

# --------------------------------------------------------------------------------- 0 after the DGCNN training
dg=$LOGS/train/scene_pcd_dgcnn.pid
until [ -f "$CK/scene_pcd_dgcnn/weights_epoch_10.pt" ] && ! alive "$dg"; do sleep 300; done
log "DGCNN training finished; ablations: $HYPER | $TIME ($MAX_TRAIN trainings at a time)"

# -------------------------------------------------------------------------------------------- 1 trainings
train() {  # variant: one training through the gate, waited for
  local v=$1 r=scene_pcd_dexart_$1
  local pidf=$LOGS/train/$r.pid
  trained "$v" && { log "$r already trained"; return 0; }
  rm -f "$LOGS/train/$r.failed"
  if ! alive "$pidf"; then
    log "training $r (${OVR[$v]}): waiting for the gate (${MEM[$v]} GiB, ${GPU[$v]} MiB GPU)"
    gate "$r" "${MEM[$v]}" "${GPU[$v]}" "$pidf" "$LOGS/train/$r.log" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      nice -n 5 $PY train.py data=scene_pcd model=lewm_pcd_dexart ${OVR[$v]} subdir="$r" output_model_name="$r" \
      trainer.max_epochs=$EPOCHS num_workers=6 || { touch "$LOGS/train/$r.failed"; return 1; }
  fi
  while alive "$pidf"; do sleep 60; done
  trained "$v" && log "trained $r" || { log "training $r FAILED (see $LOGS/train/$r.log)"; touch "$LOGS/train/$r.failed"; return 1; }
}

group_trained() {  # variants...: wait until all are trained (0) or one of them failed (1); usable from any subshell
  local v
  for v in "$@"; do
    until trained "$v"; do
      [ -f "$LOGS/train/scene_pcd_dexart_$v.failed" ] && return 1
      sleep 60
    done
  done
}

cortexbench() {  # tag variants...: one scene_grid.py over the group's models, its own encoder server
  local tag=$1; shift
  local models m missing=0
  models=$(printf 'lewm_scene_pcd_%s,' "$@"); models=${models%,}
  local runs=scene/runs_hand6 sock=@lewm_scene_abl_${tag}_enc
  local srv=$LOGS/cortexbench/lewm_scene_abl_${tag}_enc_server.pid grid=$LOGS/cortexbench/lewm_scene_abl_${tag}_grid.pid
  local cbd=$PWD/bench/cortexbench
  log "CortexBench $tag: $models x scene-task1..5 x seeds $SEEDS, proprioception $PROPRIO"
  (
    export MW_PY=$PY ENC_SOCKET=$sock
    if ! alive "$srv"; then
      gate "lewm_scene_abl_${tag}_enc_server" 2 3000 "$srv" "$LOGS/cortexbench/lewm_scene_abl_${tag}_enc_server.log" \
        --cwd "$cbd" -- bash -c 'PY=$0; source env.sh && exec "$PY" enc_server.py "$@"' "$CB_PY" "$sock" "$models" || exit 1
      until grep -q "loaded ${models##*,}" "$LOGS/cortexbench/lewm_scene_abl_${tag}_enc_server.log"; do
        alive "$srv" || { log "enc_server $tag FAILED"; exit 1; }
        sleep 5
      done
    fi
    if ! alive "$grid"; then
      gate "lewm_scene_abl_${tag}_grid" "$MEM_GRID" 0 "$grid" "$LOGS/cortexbench/scene_grid.log" --cwd "$cbd" -- \
        bash -c 'PY=$0; source env.sh && exec "$PY" scene_grid.py "$@"' "$CB_PY" --models "$models" \
        --seeds "$SEEDS" --proprio "$PROPRIO" --slots "$SLOTS" || exit 1
    fi
    while alive "$grid"; do sleep 60; done
    kill "$(cat "$srv")"
  ) || { log "CortexBench $tag FAILED"; return 1; }
  for m in ${models//,/ }; do for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$LOGS/cortexbench/$runs/$m/scene-task$k-v0/s$s/DONE" ] || missing=$((missing + 1))
  done; done; done
  [ $missing = 0 ] || { log "CortexBench $tag: $missing runs without DONE (see scene_grid.log)"; return 1; }
  (cd bench/cortexbench && export PY=$CB_PY && source env.sh && $PY summarize_scene.py lewm_scene_pcd ${models//,/ } \
    --seeds "$SEEDS" --runs "$runs")
  log "CortexBench $tag done"
}

declare -A TPID=()
running() { local n=0 p; for p in "${TPID[@]:-}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
for v in $HYPER $TIME; do
  while [ "$(running)" -ge "$MAX_TRAIN" ]; do sleep 30; done
  train "$v" & TPID[$v]=$!
  sleep 5
done
( group_trained $HYPER && cortexbench hyper $HYPER ) & cb_hyper=$!
( group_trained $TIME && cortexbench time $TIME ) & cb_time=$!

# ---------------------------------------------------------- 2 after the six trainings: the PointNet++ encoder ablation
st=0
for v in $HYPER $TIME; do wait "${TPID[$v]}" || st=1; done
if [ $st = 0 ]; then
  log "six trainings done; PointNet++ encoder ablation (log: $LOGS/scene_pcd_encoders_pointnet2.log)"
  ENCS=pointnet2 bash scripts/run_scene_pcd_encoders.sh >> "$LOGS/scene_pcd_encoders_pointnet2.log" 2>&1 & pn2=$!
else
  log "not all six trainings finished; PointNet++ not started"
  pn2=
fi
wait $cb_hyper; wait $cb_time; [ -n "$pn2" ] && wait $pn2
log "ALL_DONE"
