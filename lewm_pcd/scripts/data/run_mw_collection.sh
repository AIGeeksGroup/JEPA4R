#!/usr/bin/env bash
# Full paired (2D + 3D) collection of a MetaWorld task into $STABLEWM_HOME/datasets/metaworld:
#   train  collect_cube_pcd_sharded.py --script collect_mw.py --no-merge: SHARDS runs of collect_mw.py (config CFG),
#          NUM_TRAJ episodes in all, MAXPAR at a time, shard seeds drawn from SEED; every shard writes a 3D and a 2D file
#   test   one collect_mw.py run of TEST_TRAJ episodes at world seed TEST_SEED (above every shard's seed range), alongside
#   then   both train shard sets merged in the same order (merge_h5_shards.py --delete), 2D/3D pairing checked for train
#          and test, ep_idx added (add_eval_columns.py, for eval.py)
# Output: metaworld/<NAME>.h5 + <NAME>_pcd.h5 (train), metaworld/<NAME>_test.h5 + <NAME>_test_pcd.h5.
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_mw python> CFG=mw_binpick NAME=mw_binpick bash scripts/data/run_mw_collection.sh
# (on the RM75-B + EG2-4C2, envs/metaworld/rm75: CFG=NAME=mw_binpick_rm75, mw_button_rm75 or mw_drawer_rm75)
set -euo pipefail
cd "$(dirname "$0")/../.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
PY=${PY:-python}
CFG=${CFG:-mw_binpick}; NAME=${NAME:-mw_binpick}
NUM_TRAJ=${NUM_TRAJ:-10000}; SHARDS=${SHARDS:-40}; MAXPAR=${MAXPAR:-10}; SEED=${SEED:-3072}
TEST_TRAJ=${TEST_TRAJ:-150}; TEST_SEED=${TEST_SEED:-2000000}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} MUJOCO_GL=egl
DS=$STABLEWM_HOME/datasets/metaworld
LOGS=$STABLEWM_HOME/logs/collect
mkdir -p "$DS" "$LOGS"
log() { echo "$(date +%F_%T) $*"; }

for f in "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5" "$DS/${NAME}_test.h5" "$DS/${NAME}_test_pcd.h5"; do
  [ ! -e "$f" ] || { log "refusing to run: $f exists"; exit 1; }
done

log "test set: $TEST_TRAJ episodes, world seed $TEST_SEED"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $PY scripts/data/collect_mw.py --config-name "$CFG" num_traj="$TEST_TRAJ" \
    world_seed="$TEST_SEED" dataset_name="metaworld/${NAME}_test_pcd.h5" > "$LOGS/${NAME}_test.log" 2>&1 &
test_pid=$!
log "train set: $NUM_TRAJ episodes in $SHARDS shards, $MAXPAR at a time"
nice -n 10 $PY scripts/data/collect_cube_pcd_sharded.py --script collect_mw.py --no-merge --shards "$SHARDS" \
    --num-traj "$NUM_TRAJ" --seed "$SEED" --max-parallel "$MAXPAR" --out "metaworld/${NAME}_pcd.h5" -- --config-name="$CFG"
wait $test_pid || { log "test collection FAILED, see $LOGS/${NAME}_test.log"; exit 1; }
log "collection done"

for suffix in _pcd ""; do  # the same shard order for both
  $PY scripts/data/merge_h5_shards.py --out "$DS/$NAME$suffix.h5" --delete $(ls "$DS"/shards/"$NAME$suffix".shard*.h5 | sort)
done
rmdir "$DS/shards"
$PY - "$DS" "$NAME" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/data')
from collect_cube_pcd_sharded import check_paired
ds, name = Path(sys.argv[1]), sys.argv[2]
for n in (name, f'{name}_test'):
    check_paired(ds / f'{n}_pcd.h5', ds / f'{n}.h5')
EOF
for n in "$NAME" "${NAME}_pcd" "${NAME}_test" "${NAME}_test_pcd"; do
  $PY scripts/data/add_eval_columns.py "$DS/$n.h5"
done
du -h "$DS/$NAME.h5" "$DS/${NAME}_pcd.h5" "$DS/${NAME}_test.h5" "$DS/${NAME}_test_pcd.h5"
log "ALL_DONE"
