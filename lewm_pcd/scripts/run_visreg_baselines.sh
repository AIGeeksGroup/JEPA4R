#!/usr/bin/env bash
# VISReg ViT-B/16 (arXiv 2606.02572; ImageNet-1K, 400 epochs; the authors' visreg-vit-b-inet1k.pth, frozen as released,
# bench/cortexbench/jepa_baselines.load_visreg: the CLS after the final norm, 768-d, input 224) on the benchmarks of the
# other JEPA-family baselines, scored exactly as V-JEPA 2.1 there:
#   MetaWorld  CortexBench, the five tasks x seeds 100 / 200 / 300 (bench/cortexbench/grid.py, model visreg_vitb), as
#              scripts/run_mw_baselines.sh scores V-JEPA 2.1
#   scene      CortexBench scene-task1..5 x seeds 100 / 200 / 300, proprioception hand6 (enc_server.py + scene_grid.py),
#              as scripts/run_scene_baselines.sh
#   puzzle     3x3 / 4x4 / 4x5: OGBench's puzzle-<b>-play data re-rendered with the LeWM data's camera through the frozen
#              encoder (bench/dqc/extract_feats_baselines.py --encoder visreg=...), DQC with the board's configuration
#              of the LeWM runs (run_dqc.py --obs visreg), as scripts/run_puzzle_baselines.sh
# Every job starts through scripts/mem_gate.py (keep `mem_gate.py watch` running). Idempotent: finished runs are
# skipped, running ones (pid files) waited for. Tables: bench/cortexbench/summarize.py, summarize_scene.py,
# bench/dqc/summarize_dqc.py. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_pcd python> CB_PY=<cortexbench python> MW_PY=<lewm_mw python> \
#   DQC_PY=<lewm_dqc python> setsid nohup bash scripts/run_visreg_baselines.sh \
#     > $STABLEWM_HOME/logs/visreg_baselines.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_pcd python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
: "${MW_PY:?set MW_PY to the lewm_mw python}"; : "${DQC_PY:?set DQC_PY to the lewm_dqc python}"
L=$PWD MG=$PWD/scripts/mem_gate.py
M=visreg_vitb ENC=visreg
CKPT=$STABLEWM_HOME/checkpoints/visreg/visreg-vit-b-inet1k.pth
BOARDS=${BOARDS:-3x3 4x4 4x5}
SEEDS=${SEEDS:-100,200,300}; MW_SLOTS=${MW_SLOTS:-4}; SCENE_SLOTS=${SCENE_SLOTS:-10}
export MUJOCO_GL=egl HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=$STABLEWM_HOME/datasets/ogbench LOGS=$STABLEWM_HOME/logs
CB_RUNS=$LOGS/cortexbench/runs SCENE_RUNS=$LOGS/cortexbench/scene/runs_hand6
MW_TASKS=bin-picking,assembly,button-press-topdown,drawer-open,hammer
MW_ENVS="bin-picking assembly button-press-topdown drawer-open hammer"
mkdir -p "$LOGS/cortexbench" "$LOGS/dqc"
log() { echo "$(date +%F_%T) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
gate() {  # name mem_gib gpu_mib pidfile log [--cwd dir] -- cmd...: cmd started detached once memory and GPU have room
  local name=$1 mem=$2 gpu=$3 pidf=$4 out=$5; shift 5
  rm -f "$pidf"
  $PY "$MG" admit --name "$name" --mem "$mem" --gpu "$gpu" --pidfile "$pidf" --log "$out" "$@" 2>&1 \
    | while IFS= read -r l; do log "$l"; done
  [ -s "$pidf" ] || { log "$name not started (see above)"; return 1; }
}
[ -f "$CKPT" ] || { log "missing $CKPT"; exit 1; }

mw_done() {  # finished MetaWorld runs of the model
  local n=0 e s
  for e in $MW_ENVS; do for s in ${SEEDS//,/ }; do
    [ -f "$CB_RUNS/$M/$e-v2-goal-observable/s$s/DONE" ] && n=$((n + 1))
  done; done
  echo $n
}

metaworld() {  # CortexBench MetaWorld, the five tasks x SEEDS (grid.py: one run_bc.py per job, MW_SLOTS at a time)
  local want pidf=$LOGS/cortexbench/visreg_mw_grid.pid
  want=$((5 * $(echo "${SEEDS//,/ }" | wc -w)))
  [ "$(mw_done)" -ge "$want" ] && { log "MetaWorld $M done already"; return 0; }
  if ! alive "$pidf"; then
    log "MetaWorld $M: $MW_TASKS x seeds $SEEDS, $MW_SLOTS slots"
    gate "visreg_mw_grid" $((MW_SLOTS * 4)) 8000 "$pidf" "$LOGS/cortexbench/grid.log" --cwd "$L/bench/cortexbench" -- \
      bash -c 'PY=$0; source env.sh && exec "$PY" grid.py "$@"' "$CB_PY" --models "$M" --tasks "$MW_TASKS" \
      --seeds "$SEEDS" --slots "$MW_SLOTS" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  [ "$(mw_done)" -ge "$want" ] && log "MetaWorld $M finished" \
    || { log "MetaWorld $M FAILED: $(mw_done) of $want runs done (see grid.log)"; return 1; }
}

scene_done() {
  local n=0 k s
  for k in 1 2 3 4 5; do for s in ${SEEDS//,/ }; do
    [ -f "$SCENE_RUNS/$M/scene-task$k-v0/s$s/DONE" ] && n=$((n + 1))
  done; done
  echo $n
}

scene() {  # CortexBench scene: scene_grid.py with its own enc_server, LeWM's scene evaluation as is
  local want srv=$LOGS/cortexbench/scene_bl_${M}_enc_server.pid grid=$LOGS/cortexbench/scene_bl_${M}_grid.pid
  want=$((5 * $(echo "${SEEDS//,/ }" | wc -w)))
  [ "$(scene_done)" -ge "$want" ] && { log "scene $M done already"; return 0; }
  log "scene $M: scene-task1..5 x seeds $SEEDS, proprioception hand6, $SCENE_SLOTS slots"
  export ENC_SOCKET=@scene_bl_${M}_enc MW_PY
  if ! alive "$srv"; then
    gate "scene_bl_${M}_enc_server" 2 4000 "$srv" "$LOGS/cortexbench/scene_bl_${M}_enc_server.log" \
      --cwd "$L/bench/cortexbench" -- bash -c 'PY=$0; source env.sh && exec "$PY" enc_server.py "$@"' "$CB_PY" \
      "$ENC_SOCKET" "$M" || return 1
    until grep -q "loaded $M" "$LOGS/cortexbench/scene_bl_${M}_enc_server.log"; do
      alive "$srv" || { log "enc_server for $M FAILED (see scene_bl_${M}_enc_server.log)"; return 1; }
      sleep 5
    done
  fi
  if ! alive "$grid"; then
    gate "scene_bl_${M}_grid" $(( (SCENE_SLOTS * 32 + 9) / 10 )) 0 "$grid" "$LOGS/cortexbench/scene_grid_baselines.log" \
      --cwd "$L/bench/cortexbench" -- bash -c 'PY=$0; source env.sh && exec "$PY" scene_grid.py "$@"' "$CB_PY" \
      --models "$M" --seeds "$SEEDS" --proprio hand6 --slots "$SCENE_SLOTS" || return 1
  fi
  while alive "$grid"; do sleep 60; done
  kill "$(cat "$srv")" 2>/dev/null
  [ "$(scene_done)" -ge "$want" ] && log "scene $M finished" \
    || { log "scene $M FAILED: $(scene_done) of $want runs done (see scene_grid_baselines.log)"; return 1; }
}

rows_m() { [ "$1" = 4x5 ] && echo 3 || echo 1; }  # millions of rows of the board's play data

feats() {  # board: VISReg's latents of the board's play data
  local b=$1 fe=$DS/dqc_feats/puzzle_$1 pidf=$LOGS/dqc/extract_bl_${1}_$ENC.pid
  $DQC_PY -c "import numpy, sys; sys.exit(0 if numpy.load('$fe/train_done_$ENC.npy').all() else 1)" 2>/dev/null \
    && { log "latents $ENC ($b) exist"; return 0; }
  if ! alive "$pidf"; then
    log "latents of puzzle-$b-play: $ENC"
    gate "extract_bl_${b}_$ENC" 16 8000 "$pidf" "$LOGS/dqc/extract_bl_${b}_$ENC.log" --cwd "$L/bench/dqc" -- \
      env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 $DQC_PY extract_feats_baselines.py --env-type "$b" \
      --out "$fe" --split train --workers 6 --encoder "$ENC=$CKPT" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  $DQC_PY -c "import numpy, sys; sys.exit(0 if numpy.load('$fe/train_done_$ENC.npy').all() else 1)" \
    && log "latents $ENC ($b) done" || { log "latents $ENC ($b) FAILED (see extract_bl_${b}_$ENC.log)"; return 1; }
}

dqc() {  # board: run_dqc.py with the board's configuration of the LeWM runs (as run_puzzle_baselines.sh)
  local b=$1 runs=$LOGS/dqc/puzzle_$1 hp mem
  local pidf=$runs/$ENC.pid
  if [ "$b" = 3x3 ]; then
    hp=(--agent.backup_horizon=40 --agent.q_agg=min --agent.kappa_b=0.93 --agent.kappa_d=0.8)
  else
    hp=(--agent.backup_horizon=25 --agent.q_agg=mean --agent.kappa_b=0.9 --agent.kappa_d=0.5)
  fi
  hp+=(--agent.num_qs=2 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True --agent.distill_method=expectile
       --agent.implicit_backup_type=quantile --env_name="puzzle-$b-play-oraclerep-v0"
       --dataset_path="$DS/official/puzzle-$b-play-v0.npz" --offline_steps=1000000 --eval_interval=100000
       --eval_episodes=50 --save_every=25000)
  mem=$(( 6 + ($(rows_m "$b") * 768 * 4 + 1023) / 1024 ))  # the 768-d latents held once + 6 GiB (as V-JEPA 2.1's)
  mkdir -p "$runs/.tmp"
  [ -f "$runs/$ENC/DONE" ] && { log "DQC $ENC ($b) done already"; return 0; }
  if ! alive "$pidf"; then
    log "DQC $ENC ($b) (resumes from its checkpoint if there is one): waiting for the gate ($mem GiB)"
    gate "dqc_puzzle_${b}_$ENC" "$mem" 4000 "$pidf" "$runs/$ENC.log" --cwd "$L/bench/dqc" -- env TMPDIR="$runs/.tmp" \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false $DQC_PY run_dqc.py --obs "$ENC" \
      --ckpt "$CKPT" --lewm_epoch 10 --feats_dir "$DS/dqc_feats/puzzle_$b" --run_dir "$runs/$ENC" "${hp[@]}" || return 1
  fi
  while alive "$pidf"; do sleep 120; done
  [ -f "$runs/$ENC/DONE" ] && log "DQC $ENC ($b) done" || { log "DQC $ENC ($b) FAILED (see $runs/$ENC.log)"; return 1; }
}

pids=() st=0
metaworld & pids+=($!)
sleep 30
scene & pids+=($!)
for b in $BOARDS; do
  sleep 60
  ( feats "$b" && dqc "$b" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || st=1; done
( export PY=$CB_PY && cd bench/cortexbench && source env.sh && $PY summarize.py > /dev/null \
    && $PY summarize_scene.py lewm_scene_pcd lewm_scene_pix vjepa21_vitb "$M" lejepa_scene pldm_scene --seeds "$SEEDS" \
       --runs scene/runs_hand6 )
(cd bench/dqc && $DQC_PY summarize_dqc.py --boards $BOARDS)
[ $st = 0 ] || exit 1
log "ALL_DONE"
