#!/usr/bin/env bash
# The JEPA-family baselines of the MetaWorld and scene runs on OGBench puzzle 3x3 / 4x4 / 4x5, trained and scored as
# LeWM there (run_puzzle_pipeline.sh, results_ogbench.md):
#   LeJEPA      scripts/baselines/train_lejepa_mw.py, the board's LeWM data, 10 epochs -> checkpoints/lejepa_puzzle_<b>
#   PLDM        the framework's scripts/train/pldm.py, data=ogb, the same data, 10 epochs -> checkpoints/pldm_puzzle_<b>
#   V-JEPA 2.1  ViT-B as released, frozen, no training
# Scoring as LeWM's DQC runs of the board: OGBench's puzzle-<b>-play data re-rendered with the LeWM data's camera,
# through the frozen encoder (bench/dqc/extract_feats_baselines.py: each encoder as CortexBench loads it), input = its
# latent + the true lights, goals the lights' goal state; DQC with the board's configuration of the LeWM runs (3x3:
# h=40, kappa_b 0.93, kappa_d 0.8, q_agg min; 4x4 / 4x5: DQC's puzzle-4x5 h=25, kappa_b 0.9, kappa_d 0.5, q_agg mean),
# 1 M steps, evaluation every 100 k steps on OGBench's 5 tasks x 50 episodes -> logs/dqc/puzzle_<b>/<encoder>, beside
# LeWM's lewm2d / lewm3d; table by bench/dqc/summarize_dqc.py.
# Starts once puzzle-4x5's LeWM DQC runs are done (WAIT_FOR); every job through scripts/mem_gate.py (memory / GPU).
# Idempotent: finished trainings, latents and DQC runs are skipped, running ones (pid files) waited for. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_pcd python> DQC_PY=<lewm_dqc python> \
#     setsid nohup bash scripts/run_puzzle_baselines.sh > $STABLEWM_HOME/logs/puzzle_baselines.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_pcd python}"; : "${DQC_PY:?set DQC_PY to the lewm_dqc python}"
L=$PWD F=$PWD/third_party/stable-worldmodel MG=$PWD/scripts/mem_gate.py
BOARDS=${BOARDS:-3x3 4x4 4x5}
export PYTHONPATH=$L:$F${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       WANDB_MODE=offline WANDB_DIR=$STABLEWM_HOME/logs/train
DS=$STABLEWM_HOME/datasets/ogbench CK=$STABLEWM_HOME/checkpoints LOGS=$STABLEWM_HOME/logs
VJEPA=$CK/cortexbench_baselines/vjepa2_1_vitb_dist_vitG_384.pt
WAIT_FOR=${WAIT_FOR:-$LOGS/dqc/puzzle_4x5/lewm2d/DONE $LOGS/dqc/puzzle_4x5/lewm3d/DONE}
mkdir -p "$LOGS/train" "$LOGS/dqc"
log() { echo "$(date +%F_%T) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
gate() {  # name mem_gib gpu_mib pidfile log [--cwd dir] -- cmd...: cmd started detached once memory and GPU have room
  local name=$1 mem=$2 gpu=$3 pidf=$4 out=$5; shift 5
  rm -f "$pidf"
  $PY "$MG" admit --name "$name" --mem "$mem" --gpu "$gpu" --pidfile "$pidf" --log "$out" "$@" 2>&1 \
    | while IFS= read -r l; do log "$l"; done
  [ -s "$pidf" ] || { log "$name not started (see above)"; return 1; }
}
data_of() { case $1 in 4x5) echo ogbench/puzzle_4x5_noisy3000.h5 ;; *) echo "ogbench/puzzle_$1_noisy1000.h5" ;; esac; }
rows_m() { [ "$1" = 4x5 ] && echo 3 || echo 1; }  # millions of rows of the board's play data

train() {  # name mem_gib gpu_mib cwd cmd...: start through the gate, wait, check epoch 10
  local r=$1 mem=$2 gpu=$3 dir=$4; shift 4
  local pidf=$LOGS/train/$r.pid
  [ -f "$CK/$r/weights_epoch_10.pt" ] && { log "$r already trained"; return 0; }
  if alive "$pidf"; then
    log "$r: training running (pid $(cat "$pidf")), waiting"
  else
    log "training $r: waiting for the gate ($mem GiB, $gpu MiB GPU)"
    gate "$r" "$mem" "$gpu" "$pidf" "$LOGS/train/train_$r.log" --cwd "$dir" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      nice -n 5 "$@" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  [ -f "$CK/$r/weights_epoch_10.pt" ] && log "trained $r" || { log "training $r FAILED (see train_$r.log)"; return 1; }
}

feats() {  # board name=path...: the encoders' latents of the board's play data (one rendering pass)
  local b=$1; shift
  local fe=$DS/dqc_feats/puzzle_$b pidf=$LOGS/dqc/extract_bl_${b}_${1%%=*}.pid args=() e k missing=0
  for e in "$@"; do
    k=${e%%=*}; args+=(--encoder "$e")
    $DQC_PY -c "import numpy, sys; sys.exit(0 if numpy.load('$fe/train_done_$k.npy').all() else 1)" 2>/dev/null \
      || missing=1
  done
  [ $missing = 0 ] && { log "latents $* ($b) exist"; return 0; }
  if alive "$pidf"; then
    log "latents ($b, $*) running (pid $(cat "$pidf")), waiting"
  else
    log "latents of puzzle-$b-play: $*"
    gate "extract_bl_${b}_${1%%=*}" 12 8000 "$pidf" "$LOGS/dqc/extract_bl_$b.log" --cwd "$L/bench/dqc" -- \
      env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 $DQC_PY extract_feats_baselines.py --env-type "$b" \
      --out "$fe" --split train --workers 6 "${args[@]}" || return 1
  fi
  while alive "$pidf"; do sleep 60; done
  for e in "$@"; do
    k=${e%%=*}
    $DQC_PY -c "import numpy, sys; sys.exit(0 if numpy.load('$fe/train_done_$k.npy').all() else 1)" \
      || { log "latents $k ($b) FAILED (see extract_bl_$b.log)"; return 1; }
  done
  log "latents $* ($b) done"
}

dqc() {  # board encoder checkpoint: run_dqc.py with the board's configuration of the LeWM runs
  local b=$1 enc=$2 ck=$3 runs=$LOGS/dqc/puzzle_$b hp mem
  local pidf=$runs/$enc.pid
  if [ "$b" = 3x3 ]; then
    hp=(--agent.backup_horizon=40 --agent.q_agg=min --agent.kappa_b=0.93 --agent.kappa_d=0.8)
  else
    hp=(--agent.backup_horizon=25 --agent.q_agg=mean --agent.kappa_b=0.9 --agent.kappa_d=0.5)
  fi
  hp+=(--agent.num_qs=2 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True --agent.distill_method=expectile
       --agent.implicit_backup_type=quantile --env_name="puzzle-$b-play-oraclerep-v0"
       --dataset_path="$DS/official/puzzle-$b-play-v0.npz" --offline_steps=1000000 --eval_interval=100000
       --eval_episodes=50 --save_every=25000)
  # memory: LeWM's 192-d input took 5-6 GiB a million rows (puzzle-4x5's DQC 12 GiB with 3 M), the latent the main part
  case $enc in vjepa21) mem=$(( 7 * $(rows_m "$b") + 2 )) ;; lejepa) mem=$(( 5 * $(rows_m "$b") + 2 )) ;;
               *) mem=$(( 4 * $(rows_m "$b") + 2 )) ;; esac
  mkdir -p "$runs/.tmp"
  [ -f "$runs/$enc/DONE" ] && { log "DQC $enc ($b) done already"; return 0; }
  if alive "$pidf"; then
    log "DQC $enc ($b) running (pid $(cat "$pidf")), waiting"
  else
    log "DQC $enc ($b) (resumes from its checkpoint if there is one): waiting for the gate ($mem GiB)"
    gate "dqc_puzzle_${b}_$enc" "$mem" 4000 "$pidf" "$runs/$enc.log" --cwd "$L/bench/dqc" -- env TMPDIR="$runs/.tmp" \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false $DQC_PY run_dqc.py --obs "$enc" \
      --ckpt "$ck" --lewm_epoch 10 --feats_dir "$DS/dqc_feats/puzzle_$b" --run_dir "$runs/$enc" "${hp[@]}" || return 1
  fi
  while alive "$pidf"; do sleep 120; done
  [ -f "$runs/$enc/DONE" ] && log "DQC $enc ($b) done" || { log "DQC $enc ($b) FAILED (see $runs/$enc.log)"; return 1; }
}

board() {  # one board: V-JEPA 2.1 at once; LeJEPA and PLDM after their trainings
  local b=$1 data ok=0 p pids=()
  data=$(data_of "$b")
  [ -f "$DS/${data#ogbench/}" ] || { log "missing $data"; return 1; }
  ( feats "$b" "vjepa21=$VJEPA" && dqc "$b" vjepa21 "$VJEPA" ) & pids+=($!)
  (
    mkdir -p "$LOGS/train/pldm_puzzle_$b"
    train "lejepa_puzzle_$b" 26 8000 "$L" env PYTHONPATH="$L:$F" $PY scripts/baselines/train_lejepa_mw.py \
      --task "puzzle_$b" --dataset "$data" --name "lejepa_puzzle_$b" & l=$!
    sleep 30
    train "pldm_puzzle_$b" 32 15000 "$LOGS/train/pldm_puzzle_$b" env PYTHONPATH="$L:$F" $PY "$F/scripts/train/pldm.py" \
      data=ogb data.dataset.name="$data" '~data.dataset.keys_to_merge' subdir="pldm_puzzle_$b" \
      output_model_name="pldm_puzzle_$b" trainer.max_epochs=10 & m=$!
    wait $l; s1=$?; wait $m; s2=$?
    [ $s1 = 0 ] && [ $s2 = 0 ] || exit 1
    feats "$b" "lejepa=$CK/lejepa_puzzle_$b" "pldm=$CK/pldm_puzzle_$b" || exit 1
    dqc "$b" lejepa "$CK/lejepa_puzzle_$b" & d1=$!
    dqc "$b" pldm "$CK/pldm_puzzle_$b" & d2=$!
    wait $d1 && wait $d2
  ) & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || ok=1; done
  return $ok
}

for f in $WAIT_FOR; do
  until [ -f "$f" ]; do sleep 600; done
done
log "waited for: $WAIT_FOR"
pids=() st=0
for b in $BOARDS; do
  board "$b" & pids+=($!)
  sleep 60
done
for p in "${pids[@]}"; do wait "$p" || st=1; done
(cd bench/dqc && $DQC_PY summarize_dqc.py --boards $BOARDS)
[ $st = 0 ] || exit 1
log "ALL_DONE"
