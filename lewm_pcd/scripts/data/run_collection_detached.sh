#!/usr/bin/env bash
# Full sharded point-cloud collection, detached from the terminal / Claude session
# (setsid + nohup), so it survives disconnects. Usage:
#   bash scripts/data/run_collection_detached.sh            # 20 shards, 10000 episodes
#   SHARDS=20 bash scripts/data/run_collection_detached.sh [extra hydra overrides]
set -euo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
export PATH=/root/autodl-tmp/ms2_lewm_model_only_20260906/.venv-worldmodel/bin:$PATH   # collection env
export PYTHONPATH=$P/TMP/pydeps:$PYTHONPATH                                            # pymunk shim
SHARDS=${SHARDS:-20}
mkdir -p "$STABLEWM_HOME/logs/collect"
LOG=$STABLEWM_HOME/logs/collect/launch_$(date +%Y%m%d_%H%M%S).log
cd "$P"
setsid nohup python scripts/data/collect_cube_pcd_sharded.py --shards "$SHARDS" --num-traj 10000 --seed 3072 \
    --out ogbench/cube_single_expert_pcd.h5 "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! (own session, survives terminal exit)"
echo "launch log: $LOG"
