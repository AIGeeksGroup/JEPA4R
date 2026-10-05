#!/usr/bin/env bash
# Official 2D (pixel) LeWM reproduction on the cube: collection -> merge -> eval columns -> training -> per-epoch eval.
# Everything uses the released configs unchanged (data=ogb, model=lewm, eval cube.yaml); only the on-disk encoding
# of the dataset differs (gzip HDF5, see scripts/data/collect_cube_pix.py). Detached:
#   setsid nohup bash scripts/run_pix_pipeline.sh > $STABLEWM_HOME/logs/pix_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)  # lewm_pcd
source "$P/env.sh"
CPY=/root/autodl-tmp/ms2_lewm_model_only_20260906/.venv-worldmodel/bin/python   # collection env (swm stack, CPU torch)
TPY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python                              # training / eval env (CUDA)
SHARDS=${SHARDS:-20}; MAXPAR=${MAXPAR:-8}; RUN=${RUN:-cube_pix}; DS=$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5
# MAXPAR: shards running at once -- a pixel shard grows to ~5 GB RAM, 20 at once hit the 90 GB cgroup cap
log() { echo "$(date +%F_%T) $*"; }
cd "$P"
if [ ! -f "$DS" ]; then
  log "1/4 collection: $SHARDS shards x $((10000 / SHARDS)) episodes, $MAXPAR at a time (EGL) -> $DS"
  PYTHONPATH=$P/TMP/pydeps:${PYTHONPATH:-} $CPY scripts/data/collect_cube_pcd_sharded.py --shards "$SHARDS" --num-traj 10000 --seed 3072 \
      --script collect_cube_pix.py --out ogbench/cube_single_expert.h5 --max-parallel "$MAXPAR" || { log "collection FAILED"; exit 1; }
  log "2/4 eval columns"; $TPY scripts/data/add_eval_columns.py "$DS" || { log "add_eval_columns FAILED"; exit 1; }
else
  log "dataset exists, skipping collection: $DS"
fi
$TPY - "$DS" <<'PY'
import sys, h5py, os
f = h5py.File(sys.argv[1]); px = f['pixels']
print(f"dataset: {px.shape[0]} rows, {f['ep_len'].shape[0]} episodes, pixels {px.chunks} {px.compression}, file {os.path.getsize(sys.argv[1])/1e9:.1f} GB, ep_idx {'ep_idx' in f}")
PY
log "3/4 training: DATA=ogb model=lewm run=$RUN (10 epochs)"
DATA=ogb bash scripts/train/run_train_detached.sh lewm "$RUN" || { log "training launch FAILED"; exit 1; }
log "4/4 per-epoch eval loop (official cube.yaml, videos off)"
RUNS="$RUN" setsid nohup bash scripts/eval/auto_eval_checkpoints.sh >> "$STABLEWM_HOME/logs/eval/auto_eval.log" 2>&1 < /dev/null &
log "pipeline launched; results -> $STABLEWM_HOME/logs/eval/success_rates.csv"
