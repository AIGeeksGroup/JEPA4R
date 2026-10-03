#!/usr/bin/env bash
# One MetaWorld task after its collection (scripts/data/run_mw_collection.sh): train, then score.
#   1/3 trainings, at the same time, 10 epochs each (config/train/lewm.yaml):
#         2D  official LeWM (ViT-tiny)         data=<TASK>      model=lewm             -> checkpoints/<TASK>_pix
#         3D  point-cloud LeWM (DexArt PointNet) data=<TASK>_pcd model=lewm_pcd_dexart -> checkpoints/<TASK>_pcd_dexart
#   2/3 planning evaluation, LeWM paper protocol (config/eval/<TASK>_{pix,pcd}.yaml): epoch 10, seeds 42..45 x 50 starts
#       from the held-out test set; the random policy on the same starts
#   3/3 CortexBench (bench/cortexbench/grid.py): both encoders on CB_TASK, seeds 100/200/300
# Idempotent: finished trainings / evaluations / CortexBench runs are skipped. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> CB_PY=<cortexbench python> TASK=mw_binpick CB_TASK=bin-picking \
#     setsid nohup bash scripts/run_mw_pipeline.sh > $STABLEWM_HOME/logs/mw_binpick_pipeline.log 2>&1 < /dev/null &
# A task on the RM75-B + EG2-4C2 (envs/metaworld/rm75), e.g. TASK=mw_binpick_rm75 CB_TASK=rm75-bin-picking (also
# mw_button_rm75 / rm75-button-press-topdown, mw_drawer_rm75 / rm75-drawer-open), after its CortexBench demos
# (bench/cortexbench/make_rm75_demos.py); its CortexBench env runs in the lewm_mw python (PY, as MW_PY).
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_mw python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
TASK=${TASK:-mw_binpick}; CB_TASK=${CB_TASK:-bin-picking}
EPOCHS=10
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOGS=$STABLEWM_HOME/logs
mkdir -p "$LOGS/train" "$LOGS/eval" "$LOGS/cortexbench"
log() { echo "$(date +%F_%T) $*"; }
rate() { grep -oE "'success_rate': [0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }

for f in "$TASK" "${TASK}_pcd" "${TASK}_test" "${TASK}_test_pcd"; do
  [ -f "$STABLEWM_HOME/datasets/metaworld/$f.h5" ] || { log "missing dataset metaworld/$f.h5 (run scripts/data/run_mw_collection.sh)"; exit 1; }
done

log "1/3 trainings: ${TASK}_pix, ${TASK}_pcd_dexart ($EPOCHS epochs)"
train() {  # run workers overrides...
  local r=$1 w=$2; shift 2
  [ -f "$STABLEWM_HOME/checkpoints/$r/weights_epoch_$EPOCHS.pt" ] && { log "$r already trained"; return 0; }
  nice -n 5 $PY train.py "$@" subdir="$r" output_model_name="$r" trainer.max_epochs=$EPOCHS num_workers="$w" > "$LOGS/train/$r.log" 2>&1 \
    || { log "training $r FAILED (see $LOGS/train/$r.log)"; return 1; }
  log "trained $r"
}
train "${TASK}_pix" 8 data="$TASK" model=lewm & p2d=$!
train "${TASK}_pcd_dexart" 6 data="${TASK}_pcd" model=lewm_pcd_dexart & p3d=$!
wait $p2d; s2d=$?; wait $p3d; s3d=$?
[ $s2d = 0 ] && [ $s3d = 0 ] || exit 1

log "2/3 planning evaluation (epoch $EPOCHS, seeds 42..45, 50 starts each)"
ev() {  # tag config policy seed [overrides...]
  local tag=$1 cfg=$2 pol=$3 s=$4; shift 4
  local l=$LOGS/eval/${TASK}_${tag}_s$s.log
  [ -n "$(rate "$l")" ] && return 0
  $PY eval.py --config-name="$cfg" policy="$pol" seed="$s" "$@" hydra.run.dir="$LOGS/eval/${TASK}_${tag}_s$s" > "$l" 2>&1
  [ -n "$(rate "$l")" ] || { log "evaluation $tag seed $s FAILED, see $l"; return 1; }
  log "$tag seed $s: $(rate "$l") %"
}
for s in 42 43 44 45; do
  ev pix_e$EPOCHS "${TASK}_pix" "${TASK}_pix/weights_epoch_$EPOCHS.pt" $s || exit 1
  ev pcd_e$EPOCHS "${TASK}_pcd" "${TASK}_pcd_dexart/weights_epoch_$EPOCHS.pt" $s || exit 1
  # the 2D and 3D test sets are row-aligned, so a seed draws the same starts for both; one random run serves both
  ev random "${TASK}_pix" random $s output.filename="$LOGS/eval/${TASK}_random_results.txt" || exit 1
done

log "3/3 CortexBench: lewm_${TASK}_pix, lewm_${TASK}_pcd on $CB_TASK"
(
  export MW_PY=$PY && export PY=$CB_PY && cd bench/cortexbench && source env.sh \
    && $PY grid.py --models "lewm_${TASK}_pix,lewm_${TASK}_pcd" --tasks "$CB_TASK" --slots 6 >> "$LOGS/cortexbench/grid.log" 2>&1 \
    && $PY summarize.py > /dev/null
) || { log "CortexBench FAILED, see $LOGS/cortexbench/grid.log"; exit 1; }
log "ALL_DONE (CortexBench tables: $LOGS/cortexbench/results/scores.md)"
