#!/usr/bin/env bash
# Paired (2D + 3D) collection of OGBench scene into $STABLEWM_HOME/datasets/ogbench, as run_mw_collection.sh for a
# MetaWorld task: SHARDS runs of scripts/data/collect_ogb.py (config ogb_scene_pcd) per object task (cube, button,
# drawer, window), MAXPAR at a time; task k's episode i at seed (k + 1) * 10^6 + i, shard s of a task collecting
# i = s, s + SHARDS, ...; then both shard sets merged in the same order, task by task and shard by shard
# (merge_h5_shards.py --delete), so the rows run cube, button, drawer, window (task_id 0..3); 2D / 3D pairing checked,
# ep_idx added (add_eval_columns.py). Output: ogbench/<NAME>.h5 (2D) + <NAME>_pcd.h5 (3D).
# Defaults = the full scale: 4 x 2500 episodes = 10000 (2 M rows). The small run of results_ogbench.md: N=250 SHARDS=5
# NAME=scene_wm (the same seeds, so its episodes are the first 250 of each task here).
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> bash scripts/data/run_scene_collection.sh
set -euo pipefail
cd "$(dirname "$0")/../.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
PY=${PY:-python}
N=${N:-2500}; SHARDS=${SHARDS:-6}; MAXPAR=${MAXPAR:-12}; NAME=${NAME:-scene_wm10000}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
DS=$STABLEWM_HOME/datasets/ogbench
LOGS=$STABLEWM_HOME/logs/collect/$NAME
mkdir -p "$DS/shards" "$LOGS"
log() { echo "$(date +%F_%T) $*"; }
for f in "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5"; do [ ! -e "$f" ] || { log "refusing to run: $f exists"; exit 1; }; done
if ls "$DS/shards/$NAME".*.shard*.h5 "$DS/shards/${NAME}_pcd".*.shard*.h5 > /dev/null 2>&1; then
  log "refusing to run: shards of $NAME exist in $DS/shards"; exit 1
fi

log "train set: $N episodes x 4 tasks, $SHARDS shards per task, $MAXPAR at a time"
pids=()
k=0
for task in cube button drawer window; do
  base=$(( (k + 1) * 1000000 ))
  for ((s = 0; s < SHARDS; s++)); do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
    sh=$(printf 'shard%03d' "$s")
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY scripts/data/collect_ogb.py --config-name ogb_scene_pcd env.task="$task" \
      seed_base="$base" episodes="$N" shard="$s" num_shards="$SHARDS" dataset_name="ogbench/shards/${NAME}_pcd.$task.$sh.h5" \
      > "$LOGS/${task}_$sh.log" 2>&1 & pids+=($!)
  done
  k=$((k + 1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { log "a shard FAILED (see $LOGS)"; exit 1; }
log "shards done; $(grep -h 'DONE' "$LOGS"/*_shard*.log | grep -oE '[0-9]+ unhealthy' | awk '{s += $1} END {print s + 0}') unhealthy episodes collected again"

for suffix in _pcd ""; do  # the same order for both
  files=()
  for task in cube button drawer window; do files+=($(ls "$DS/shards/${NAME}${suffix}.$task".shard*.h5 | sort)); done
  $PY scripts/data/merge_h5_shards.py --out "$DS/$NAME$suffix.h5" --delete "${files[@]}"
done
rmdir "$DS/shards" 2> /dev/null || true
$PY - "$DS" "$NAME" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/data')
from collect_cube_pcd_sharded import check_paired
ds, name = Path(sys.argv[1]), sys.argv[2]
check_paired(ds / f'{name}_pcd.h5', ds / f'{name}.h5')
PYEOF
for n in "$NAME" "${NAME}_pcd"; do $PY scripts/data/add_eval_columns.py "$DS/$n.h5" | head -1; done
du -h "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5"
log "ALL_DONE"
