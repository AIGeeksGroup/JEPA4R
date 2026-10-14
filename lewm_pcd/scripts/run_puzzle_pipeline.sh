#!/usr/bin/env bash
# OGBench puzzle-3x3 from collection to DQC scores (results_ogbench.md):
#   1/4 collection   scripts/data/run_puzzle_collection.sh: the official noisy recipe (envs/ogb_puzzle_env.py), N training
#                    episodes of 1001 steps + 20 test episodes -> datasets/ogbench/<NAME>{,_pcd}.h5
#   2/4 trainings    EPOCHS epochs each (config/train/lewm.yaml):
#                    2D  official LeWM (ViT-tiny)            data=<DATA>     model=lewm            -> checkpoints/<RUN>_pix
#                    3D  point-cloud LeWM (DexArt PointNet)  data=<DATA>_pcd model=lewm_pcd_dexart -> checkpoints/<RUN>_pcd_dexart
#                    each starts once the GPU has room for it (3D 4.5 GB, 2D 15 GB) and, for the 2D, the container's
#                    unreclaimable memory (anon + shmem) is below 50 GiB: the 2D training holds ~25 GB of prefetched
#                    batches (8 workers x prefetch 3) against the 90 GiB cap; the gate is shared with
#                    run_scene_pipeline.sh (a lock file), so two trainings never take the same free memory
#   3/4 latents      bench/dqc/extract_feats.py: OGBench's puzzle-3x3-play training set re-rendered with the LeWM data's
#                    camera, through both frozen encoders (epoch EPOCHS) -> datasets/ogbench/dqc_feats/<RUN>
#   4/4 DQC          bench/dqc/run_dqc.py, inputs lewm2d and lewm3d (the 192-d latent + the 9 true lights), 1 M steps,
#                    evaluation every 100 k steps on OGBench's 5 tasks x 50 episodes -> logs/dqc/<RUN>/<input>
# Defaults = the full scale: 1000 episodes (puzzle-3x3-noisy-v0's size), 10 epochs. The small run of results_ogbench.md:
# N=200 SHARDS=5 NAME=puzzle_3x3_noisy TEST=puzzle_3x3_noisy_small_test DATA=puzzle_3x3_small RUN=puzzle_3x3_small
# (its DQC also ran the state input).
# Idempotent and restartable: finished steps are skipped, a running collection / training / DQC run (pid files under
# logs/) is waited for instead of started again; DQC runs resume from their last checkpoint. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> DQC_PY=<lewm_dqc python> setsid nohup \
#     bash scripts/run_puzzle_pipeline.sh > $STABLEWM_HOME/logs/puzzle_pipeline.log 2>&1 < /dev/null &
# Keep TMP/ops/cache_guard.py running alongside (the container's memory cap, see README).
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${DQC_PY:?set DQC_PY to the lewm_dqc python}"
N=${N:-1000}; SHARDS=${SHARDS:-8}; NAME=${NAME:-puzzle_3x3_noisy1000}
DATA=${DATA:-puzzle_3x3}; RUN=${RUN:-puzzle_3x3}; EPOCHS=${EPOCHS:-10}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 MUJOCO_GL=egl \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
DS=$STABLEWM_HOME/datasets/ogbench
LOGS=$STABLEWM_HOME/logs
mkdir -p "$LOGS/train" "$LOGS/collect" "$LOGS/dqc"
log() { echo "$(date +%F_%T) $*"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
fixed_gib() { awk '$1 == "anon" || $1 == "shmem" {s += $2} END {print int(s / 2^30)}' /sys/fs/cgroup/memory.stat; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
detach() {  # pidfile log cmd...: run cmd in its own session (survives this script), its pid in pidfile
  local pidf=$1 out=$2; shift 2
  rm -f "$pidf"
  setsid nohup bash -c 'echo $$ > "$0"; exec "$@"' "$pidf" "$@" > "$out" 2>&1 < /dev/null &
  until [ -s "$pidf" ]; do sleep 1; done
}

# ------------------------------------------------------------------------------------------------- 1 collection
pidf=$LOGS/collect/$NAME.pid
if [ -f "$DS/$NAME.h5" ] && [ -f "$DS/${NAME}_pcd.h5" ] && ! alive "$pidf"; then
  log "1/4 collection: datasets exist"
else
  if alive "$pidf"; then
    log "1/4 collection running (pid $(cat "$pidf")), waiting"
  else
    log "1/4 collection: $N training + 20 test episodes"
    N=$N SHARDS=$SHARDS NAME=$NAME detach "$pidf" "$LOGS/collect/$NAME.log" bash scripts/data/run_puzzle_collection.sh
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

# ---------------------------------------------------------------------------------------------------- 3 latents
CK=$STABLEWM_HOME/checkpoints
FE=$DS/dqc_feats/$RUN
if [ -f "$FE/train_done.npy" ] && $DQC_PY -c "import numpy, sys; sys.exit(0 if numpy.load('$FE/train_done.npy').all() else 1)"; then
  log "3/4 latents exist"
else
  log "3/4 latents: puzzle-3x3-play training set through ${RUN}_pix / ${RUN}_pcd_dexart (epoch $EPOCHS)"
  (cd bench/dqc && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 $DQC_PY extract_feats.py --ckpt-2d "$CK/${RUN}_pix" \
    --ckpt-3d "$CK/${RUN}_pcd_dexart" --epoch "$EPOCHS" --out "$FE" --split train --workers 6 \
    >> "$LOGS/dqc/extract_$RUN.log" 2>&1) || { log "latents FAILED (see $LOGS/dqc/extract_$RUN.log)"; exit 1; }
  log "latents done"
fi

# -------------------------------------------------------------------------------------------------------- 4 DQC
log "4/4 DQC: lewm2d, lewm3d (latent + lights), 1 M steps"
HP=(--agent.num_qs=2 --agent.policy_chunk_size=5 --agent.backup_horizon=40 --agent.use_chunk_critic=True
    --agent.distill_method=expectile --agent.implicit_backup_type=quantile --agent.q_agg=min --agent.kappa_b=0.93
    --agent.kappa_d=0.8 --offline_steps=1000000 --eval_interval=100000 --eval_episodes=50 --save_every=25000)
RUNS=$LOGS/dqc/$RUN
mkdir -p "$RUNS/.tmp"
dqc() {  # input checkpoint
  local obs=$1 ck=$2 pidf=$RUNS/$1.pid
  [ -f "$RUNS/$obs/DONE" ] && { log "DQC $obs done already"; return 0; }
  if alive "$pidf"; then
    log "DQC $obs running (pid $(cat "$pidf")), waiting"
  else
    log "DQC $obs (resumes from its checkpoint if there is one)"
    (cd bench/dqc && TMPDIR=$RUNS/.tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      detach "$pidf" "$RUNS/$obs.log" $DQC_PY run_dqc.py --obs "$obs" --ckpt "$ck" --lewm_epoch "$EPOCHS" \
      --feats_dir "$FE" --run_dir "$RUNS/$obs" "${HP[@]}")
  fi
  while alive "$pidf"; do sleep 120; done
  [ -f "$RUNS/$obs/DONE" ] && log "DQC $obs done" || { log "DQC $obs FAILED (see $RUNS/$obs.log)"; return 1; }
}
dqc lewm2d "$CK/${RUN}_pix" & d2=$!
dqc lewm3d "$CK/${RUN}_pcd_dexart" & d3=$!
wait $d2; s2=$?; wait $d3; s3=$?
for obs in lewm2d lewm3d; do
  echo "$obs: $(tail -1 "$RUNS/$obs/eval.jsonl" 2>/dev/null)"
done
[ $s2 = 0 ] && [ $s3 = 0 ] || exit 1
log "ALL_DONE"
