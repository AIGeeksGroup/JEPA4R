#!/usr/bin/env bash
# CortexBench's TriFinger tasks from collection to scores: the steps of scripts/data/run_mw_collection.sh and
# scripts/run_mw_pipeline.sh with CortexBench's own TriFinger envs (envs/trifinger: MoveCubeEnv / CubeReachEnv) and the
# three-part plan around CortexBench's experts (envs/trifinger/expert.py, reproduced from the demos) instead of a
# MetaWorld task.
#   1/4 collection   NUM_TRAJ move + NUM_TRAJ reach episodes (official lengths: 10 / 5 steps) in SHARDS shards, half per
#                    task (scripts/data/collect_tf_sharded.py: shard world seeds in consecutive disjoint ranges from
#                    SEED), MAXPAR at a time; the test set (TEST_TRAJ move episodes at world seed TEST_SEED) beside them;
#                    merged, 2D / 3D pairing checked, ep_idx added -> datasets/trifinger/tf_cube{,_pcd,_test,_test_pcd}.h5
#   2/4 trainings    at the same time, EPOCHS epochs (config/train/lewm.yaml):
#                    2D  official LeWM (ViT-tiny)            data=tf_cube     model=lewm            -> checkpoints/tf_cube_pix
#                    3D  point-cloud LeWM (DexArt PointNet)  data=tf_cube_pcd model=lewm_pcd_dexart -> checkpoints/tf_cube_pcd_dexart
#   3/4 planning     LeWM paper protocol in MoveCubeEnv (config/eval/tf_cube_{pix,pcd}.yaml), last epoch: seeds 42..45 x
#                    50 starts of the test set; on the same starts the random policy and the zero action
#                    (scripts/eval/eval_hold.py: the protocol's floor)
#   4/4 CortexBench  bench/cortexbench/run_tf_grid.sh with both encoders (last epoch): reach_cube / move_cube x seeds
#                    100 / 200 / 300
# Defaults: the full scale, 2 x 120000 episodes (~2.04 M rows, MetaWorld's ~2 M), 10 epochs. results_trifinger.md's
# run: NUM_TRAJ=2000 SHARDS=14 EPOCHS=100. Idempotent: finished steps are skipped (rerun to resume). Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_tf python> setsid nohup bash scripts/run_tf_pipeline.sh \
#     > $STABLEWM_HOME/logs/tf_cube_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_tf python}"
NUM_TRAJ=${NUM_TRAJ:-120000}; SHARDS=${SHARDS:-56}; MAXPAR=${MAXPAR:-14}; SEED=${SEED:-3072}
TEST_TRAJ=${TEST_TRAJ:-150}; TEST_SEED=${TEST_SEED:-2000000}; EPOCHS=${EPOCHS:-10}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} HYDRA_FULL_ERROR=1 \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=$STABLEWM_HOME/datasets/trifinger
LOGS=$STABLEWM_HOME/logs
mkdir -p "$DS" "$LOGS/train" "$LOGS/eval" "$LOGS/collect"
log() { echo "$(date +%F_%T) $*"; }
rate() { grep -oE "'success_rate': [0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }

# ------------------------------------------------------------------------------------------------- 1 collection
if [ -f "$DS/tf_cube.h5" ] && [ -f "$DS/tf_cube_pcd.h5" ] && [ -f "$DS/tf_cube_test.h5" ] && [ -f "$DS/tf_cube_test_pcd.h5" ]; then
  log "1/4 collection: datasets exist"
else
  log "1/4 collection: test set ($TEST_TRAJ move episodes, world seed $TEST_SEED) + train set ($NUM_TRAJ move + $NUM_TRAJ reach episodes in $SHARDS shards, $MAXPAR at a time)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY scripts/data/collect_tf.py --task move --num-traj "$TEST_TRAJ" \
    --world-seed "$TEST_SEED" --num-envs 10 --out trifinger/tf_cube_test_pcd.h5 > "$LOGS/collect/tf_cube_test.log" 2>&1 &
  test_pid=$!
  $PY scripts/data/collect_tf_sharded.py --num-traj "$NUM_TRAJ" --tasks move reach --shards "$SHARDS" \
    --max-parallel "$MAXPAR" --seed "$SEED" --out trifinger/tf_cube_pcd.h5 || { log "train collection FAILED"; kill $test_pid; exit 1; }
  wait $test_pid || { log "test collection FAILED, see $LOGS/collect/tf_cube_test.log"; exit 1; }
  $PY -c "
import sys; from pathlib import Path; sys.path.insert(0, 'scripts/data')
from collect_cube_pcd_sharded import check_paired
check_paired(Path('$DS/tf_cube_test_pcd.h5'), Path('$DS/tf_cube_test.h5'))" || exit 1
  for n in tf_cube tf_cube_pcd tf_cube_test tf_cube_test_pcd; do $PY scripts/data/add_eval_columns.py "$DS/$n.h5" | head -1; done
  du -h "$DS"/tf_cube*.h5
  log "collection done"
fi

# -------------------------------------------------------------------------------------------------- 2 trainings
log "2/4 trainings: tf_cube_pix, tf_cube_pcd_dexart ($EPOCHS epochs)"
train() {  # run workers overrides...
  local r=$1 w=$2; shift 2
  [ -f "$STABLEWM_HOME/checkpoints/$r/weights_epoch_$EPOCHS.pt" ] && { log "$r already trained"; return 0; }
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 $PY train.py "$@" subdir="$r" output_model_name="$r" trainer.max_epochs=$EPOCHS \
    num_workers="$w" > "$LOGS/train/$r.log" 2>&1 || { log "training $r FAILED (see $LOGS/train/$r.log)"; return 1; }
  log "trained $r"
}
train tf_cube_pix 8 data=tf_cube model=lewm & p2d=$!
train tf_cube_pcd_dexart 6 data=tf_cube_pcd model=lewm_pcd_dexart & p3d=$!
wait $p2d; s2d=$?; wait $p3d; s3d=$?
[ $s2d = 0 ] && [ $s3d = 0 ] || exit 1

# ------------------------------------------------------------------------------------------------- 3 planning
log "3/4 planning evaluation (epoch $EPOCHS, seeds 42..45, 50 starts each)"
ev() {  # tag script config policy seed [overrides...]
  local tag=$1 script=$2 cfg=$3 pol=$4 s=$5; shift 5
  local l=$LOGS/eval/tf_cube_${tag}_s$s.log
  [ -n "$(rate "$l")" ] && return 0
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $script --config-name="$cfg" policy="$pol" seed="$s" "$@" \
    hydra.run.dir="$LOGS/eval/tf_cube_${tag}_s$s" > "$l" 2>&1
  [ -n "$(rate "$l")" ] || { log "evaluation $tag seed $s FAILED, see $l"; return 1; }
  log "$tag seed $s: $(rate "$l") %"
}
for s in 42 43 44 45; do
  ev pix_e$EPOCHS eval.py tf_cube_pix tf_cube_pix/weights_epoch_$EPOCHS.pt $s || exit 1
  ev pcd_e$EPOCHS eval.py tf_cube_pcd tf_cube_pcd_dexart/weights_epoch_$EPOCHS.pt $s || exit 1
  # the 2D and 3D test sets are row-aligned, so a seed draws the same starts for both; one baseline run serves both
  ev random eval.py tf_cube_pix random $s output.filename="$LOGS/eval/tf_cube_random_results.txt" || exit 1
  ev hold scripts/eval/eval_hold.py tf_cube_pix random $s output.filename="$LOGS/eval/tf_cube_hold_results.txt" || exit 1
done

# ----------------------------------------------------------------------------------------------- 4 CortexBench
log "4/4 CortexBench TriFinger: lewm_tf_pix, lewm_tf_pcd (epoch $EPOCHS), reach_cube / move_cube x seeds 100 / 200 / 300"
TF_CKPT_PIX=$STABLEWM_HOME/checkpoints/tf_cube_pix TF_CKPT_PCD=$STABLEWM_HOME/checkpoints/tf_cube_pcd_dexart TF_EPOCH=$EPOCHS \
  PY=$PY bash bench/cortexbench/run_tf_grid.sh || exit 1
$PY bench/cortexbench/summarize_tf.py
log "ALL_DONE"
