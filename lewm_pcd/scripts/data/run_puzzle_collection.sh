#!/usr/bin/env bash
# Paired (2D + 3D) collection of OGBench puzzle-3x3 with the official noisy recipe into $STABLEWM_HOME/datasets/ogbench,
# as run_mw_collection.sh for a MetaWorld task: N training episodes of 1001 steps at seeds 1000000 + i in SHARDS runs of
# scripts/data/collect_ogb.py (config ogb_puzzle_pcd; shard s collects i = s, s + SHARDS, ...) and TEST_N test episodes
# at seeds 2000000 + i (for probes; not trained on) in 2 runs; each set merged in shard order (merge_h5_shards.py
# --delete), 2D / 3D pairing checked, ep_idx added (add_eval_columns.py).
# Output: ogbench/<NAME>{,_pcd}.h5 and ogbench/puzzle_3x3_noisy_test{,_pcd}.h5.
# Defaults = the full scale: 1000 episodes (puzzle-3x3-noisy-v0's size). The small run of results_ogbench.md: N=200
# SHARDS=5 NAME=puzzle_3x3_noisy. The random actions are not reproducible from the seeds (envs/ogb_puzzle_env.py), so
# a rerun gives other rows.
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> bash scripts/data/run_puzzle_collection.sh
set -euo pipefail
cd "$(dirname "$0")/../.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
PY=${PY:-python}
N=${N:-1000}; SHARDS=${SHARDS:-8}; NAME=${NAME:-puzzle_3x3_noisy1000}; TEST_N=${TEST_N:-20}; TEST=${TEST:-puzzle_3x3_noisy_test}
export PYTHONPATH=$PWD:$PWD/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
DS=$STABLEWM_HOME/datasets/ogbench
LOGS=$STABLEWM_HOME/logs/collect/$NAME
mkdir -p "$DS/shards" "$LOGS"
log() { echo "$(date +%F_%T) $*"; }
for f in "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5" "$DS/$TEST.h5" "$DS/${TEST}_pcd.h5"; do
  [ ! -e "$f" ] || { log "refusing to run: $f exists"; exit 1; }
done

run_set() {  # name episodes seed_base shards
  local name=$1 n=$2 base=$3 k=$4 pids=() s sh
  for ((s = 0; s < k; s++)); do
    sh=$(printf 'shard%03d' "$s")
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY scripts/data/collect_ogb.py --config-name ogb_puzzle_pcd \
      seed_base="$base" episodes="$n" shard="$s" num_shards="$k" dataset_name="ogbench/shards/${name}_pcd.$sh.h5" \
      > "$LOGS/${name}_$sh.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { log "$name: a shard FAILED (see $LOGS)"; return 1; }; done
  for suffix in _pcd ""; do  # the same order for both
    $PY scripts/data/merge_h5_shards.py --out "$DS/$name$suffix.h5" --delete $(ls "$DS/shards/$name$suffix".shard*.h5 | sort)
  done
}

log "train set: $N episodes in $SHARDS shards; test set: $TEST_N episodes in 2 shards"
run_set "$NAME" "$N" 1000000 "$SHARDS" & a=$!
run_set "$TEST" "$TEST_N" 2000000 2 & b=$!
wait $a; wait $b
rmdir "$DS/shards" 2> /dev/null || true
$PY - "$DS" "$NAME" "$TEST" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/data')
from collect_cube_pcd_sharded import check_paired
ds = Path(sys.argv[1])
for n in sys.argv[2:]:
    check_paired(ds / f'{n}_pcd.h5', ds / f'{n}.h5')
PYEOF
for n in "$NAME" "${NAME}_pcd" "$TEST" "${TEST}_pcd"; do $PY scripts/data/add_eval_columns.py "$DS/$n.h5" | head -1; done
du -h "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5" "$DS/$TEST.h5" "$DS/${TEST}_pcd.h5"
log "ALL_DONE"
