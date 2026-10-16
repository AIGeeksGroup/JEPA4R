#!/usr/bin/env bash
# The JEPA-family baselines of the MetaWorld runs (scripts/run_mw_baselines.sh) on OGBench scene, trained as there and
# scored by LeWM's scene evaluation as is (run_scene_pipeline.sh step 4; results_ogbench.md):
#   LeJEPA      scripts/baselines/train_lejepa_mw.py on LeWM's scene data (config/train/data/scene.yaml's dataset,
#               ogbench/scene_wm10000.h5), 10 epochs                                          -> checkpoints/lejepa_scene
#   PLDM        the framework's scripts/train/pldm.py, data=ogb on the same dataset, 10 epochs -> checkpoints/pldm_scene
#   V-JEPA 2.1  ViT-B as released, frozen, no training (model vjepa21_vitb)
# CortexBench: bench/cortexbench/scene_grid.py per model (conf/model/{lejepa_scene,pldm_scene,vjepa21_vitb}.yaml):
# scene-task1..5 x seeds 100 / 200 / 300, the 6-d proprioception, each encoder served by its own enc_server.py; it starts
# once LeWM's own scene CortexBench is over (all 30 of its runs done), V-JEPA 2.1 first, then each model at epoch 10;
# table by summarize_scene.py next to LeWM's.
# Every training, encoder server and scene_grid.py run starts through scripts/mem_gate.py (the OGBench pipelines' gate,
# see run_scene_pipeline.sh): only when the container's memory (anon + shmem) with the reservations of the jobs still
# starting and its own stays within the budget (72 GiB) and the GPU has room for it. Memory each (GiB): LeJEPA 26
# (25.8 measured: 15.7 of it the 8 workers' prefetched batches in shared memory), PLDM 32 (28.4 freed when its first
# run was stopped at the cap), scene_grid.py with 10 slots 32 (V-JEPA 2.1's reached 29.4 within 6 minutes; LeWM's
# 16), an encoder server 2. Keep `mem_gate.py watch` running alongside.
# Idempotent: finished trainings and CortexBench runs are skipped, a running training (pid file) is waited for. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_pcd python> MW_PY=<lewm_mw python> CB_PY=<cortexbench python> \
#     setsid nohup bash scripts/run_scene_baselines.sh > $STABLEWM_HOME/logs/scene_baselines.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_pcd python}"; : "${MW_PY:?set MW_PY to the lewm_mw python}"
: "${CB_PY:?set CB_PY to the cortexbench python}"
L=$PWD
F=$L/third_party/stable-worldmodel
DATASET=ogbench/scene_wm10000.h5
SEEDS=${SEEDS:-100,200,300}; SLOTS=${SLOTS:-10}; PROPRIO=hand6
export MUJOCO_GL=egl HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache WANDB_MODE=offline WANDB_DIR=$STABLEWM_HOME/logs/train
LOGS=$STABLEWM_HOME/logs
RUNS=$LOGS/cortexbench/scene/runs_hand6
mkdir -p "$LOGS/train" "$LOGS/cortexbench"
log() { echo "$(date +%F_%T) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
MG=$L/scripts/mem_gate.py
gate() {  # name mem_gib gpu_mib pidfile log [--cwd dir] -- cmd...: cmd started detached (its own session, output
          # appended to log, pid in pidfile) once the memory budget and the GPU have room (mem_gate.py admit, blocks)
  local name=$1 mem=$2 gpu=$3 pidf=$4 out=$5; shift 5
  rm -f "$pidf"
  $PY "$MG" admit --name "$name" --mem "$mem" --gpu "$gpu" --pidfile "$pidf" --log "$out" "$@" 2>&1 \
    | while IFS= read -r l; do log "$l"; done
  [ -s "$pidf" ] || { log "$name not started (see above)"; return 1; }
}
trained() { [ -f "$STABLEWM_HOME/checkpoints/$1/weights_epoch_10.pt" ]; }

train() {  # run mem_gib gpu_mib cmd...: start through the gate, wait, check epoch 10
  local r=$1 mem=$2 gpu=$3; shift 3
  local pidf=$LOGS/train/$r.pid
  trained "$r" && { log "$r already trained"; return 0; }
  if alive "$pidf"; then
    log "$r: training running (pid $(cat "$pidf")), waiting"
  else
    log "training $r: waiting for the gate ($mem GiB, $gpu MiB GPU)"
    gate "$r" "$mem" "$gpu" "$pidf" "$LOGS/train/train_$r.log" --cwd "$PWD" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      nice -n 5 "$@" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  trained "$r" && log "trained $r" || { log "training $r FAILED (see $LOGS/train/train_$r.log)"; return 1; }
}

runs_done() {  # model: number of its finished runs
  local m=$1 n=0 k s
  for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$RUNS/$m/scene-task$k-v0/s$s/DONE" ] && n=$((n + 1))
  done; done
  echo $n
}

cortexbench() {  # model: scene_grid.py with its own enc_server, LeWM's scene evaluation as is
  local m=$1 want n st
  want=$((5 * $(echo "${SEEDS//,/ }" | wc -w)))
  [ "$(runs_done "$m")" -ge "$want" ] && { log "CortexBench $m done already"; return 0; }
  log "CortexBench $m: scene-task1..5 x seeds $SEEDS, proprioception $PROPRIO"
  (
    export MW_PY ENC_SOCKET=@scene_bl_${m}_enc  # scene_grid.py runs the scene env in MW_PY
    CBD_=$L/bench/cortexbench
    srv=$LOGS/cortexbench/scene_bl_${m}_enc_server.pid grid=$LOGS/cortexbench/scene_bl_${m}_grid.pid
    if ! alive "$srv"; then
      gate "scene_bl_${m}_enc_server" 2 4000 "$srv" "$LOGS/cortexbench/scene_bl_${m}_enc_server.log" --cwd "$CBD_" -- \
        bash -c 'PY=$0; source env.sh && exec "$PY" enc_server.py "$@"' "$CB_PY" "$ENC_SOCKET" "$m" || exit 1
      until grep -q "loaded $m" "$LOGS/cortexbench/scene_bl_${m}_enc_server.log"; do
        alive "$srv" || { log "enc_server for $m FAILED (see $LOGS/cortexbench/scene_bl_${m}_enc_server.log)"; exit 1; }
        sleep 5
      done
    fi
    if ! alive "$grid"; then
      gate "scene_bl_${m}_grid" 32 0 "$grid" "$LOGS/cortexbench/scene_grid_baselines.log" --cwd "$CBD_" -- \
        bash -c 'PY=$0; source env.sh && exec "$PY" scene_grid.py "$@"' "$CB_PY" --models "$m" --seeds "$SEEDS" \
        --proprio "$PROPRIO" --slots "$SLOTS" || exit 1
    fi
    while alive "$grid"; do sleep 60; done
    kill "$(cat "$srv")"
    n=$(runs_done "$m")  # judged by the DONE files (scene_grid_baselines.log is appended to by every attempt)
    [ "$n" -ge "$want" ] \
      || { log "CortexBench $m FAILED: $n of $want runs done (see scene_grid_baselines.log)"; exit 1; }
    log "CortexBench $m finished"
  )
}

cb_lane() {  # after LeWM's own scene CortexBench: V-JEPA 2.1 first, then every trained model
  local m left ok=0
  until [ "$(runs_done lewm_scene_pcd)" -ge 15 ] && [ "$(runs_done lewm_scene_pix)" -ge 15 ]; do sleep 600; done
  log "LeWM's scene CortexBench is over"
  cortexbench vjepa21_vitb || ok=1
  left="lejepa_scene pldm_scene"
  while [ -n "$left" ]; do
    for m in $left; do
      if trained "$m"; then
        cortexbench "$m" || ok=1
        left=$(echo "$left" | tr ' ' '\n' | grep -vx "$m" | tr '\n' ' ' | xargs)
      elif [ -f "$LOGS/train/.scene_baselines_trainings_over" ]; then
        log "$m not trained, no CortexBench"; ok=1
        left=$(echo "$left" | tr ' ' '\n' | grep -vx "$m" | tr '\n' ' ' | xargs)
      fi
    done
    [ -n "$left" ] && sleep 300
  done
  return $ok
}

[ -f "$STABLEWM_HOME/datasets/$DATASET" ] || { log "missing dataset $DATASET (run_scene_pipeline.sh step 1)"; exit 1; }
rm -f "$LOGS/train/.scene_baselines_trainings_over"
cb_lane & CB=$!
train lejepa_scene 26 8000 env PYTHONPATH=$L:$F $PY scripts/baselines/train_lejepa_mw.py --task scene \
  --dataset "$DATASET" --name lejepa_scene & LJ=$!
sleep 30
mkdir -p "$LOGS/train/pldm_scene"
( cd "$LOGS/train/pldm_scene" && train pldm_scene 32 15000 env PYTHONPATH=$L:$F $PY "$F/scripts/train/pldm.py" \
    data=ogb data.dataset.name="$DATASET" '~data.dataset.keys_to_merge' subdir=pldm_scene output_model_name=pldm_scene \
    trainer.max_epochs=10 ) & PL=$!
wait $LJ; s1=$?; wait $PL; s2=$?
touch "$LOGS/train/.scene_baselines_trainings_over"
log "trainings over (LeJEPA exit $s1, PLDM exit $s2)"
wait $CB; s3=$?
( export PY=$CB_PY && cd bench/cortexbench && source env.sh \
    && $PY summarize_scene.py lewm_scene_pcd lewm_scene_pix vjepa21_vitb lejepa_scene pldm_scene --seeds "$SEEDS" \
       --runs scene/runs_hand6 )
[ $s1 = 0 ] && [ $s2 = 0 ] && [ $s3 = 0 ] || exit 1
log "ALL_DONE"
