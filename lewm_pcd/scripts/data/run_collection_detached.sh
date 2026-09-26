#!/usr/bin/env bash
# Full sharded paired collection (3D point-cloud + 2D pixel dataset of the same rollouts), detached from the
# terminal / Claude session (setsid + nohup), so it survives disconnects. Usage:
#   bash scripts/data/run_collection_detached.sh            # 20 shards, 10000 episodes, 8 shards at a time
#   SHARDS=20 MAXPAR=8 bash scripts/data/run_collection_detached.sh [extra hydra overrides]
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$P/env.sh"
PYTHON="${PYTHON:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "collection Python is not executable: $PYTHON" >&2
  exit 1
fi
SHARDS=${SHARDS:-20}; MAXPAR=${MAXPAR:-8}   # MAXPAR: shards at once -- a shard that renders pixels grows to ~5 GB RAM (90 GB cgroup cap)
mkdir -p "$STABLEWM_HOME/logs/collect"
LOG=$STABLEWM_HOME/logs/collect/launch_$(date +%Y%m%d_%H%M%S).log
cd "$P"
setsid nohup "$PYTHON" scripts/data/collect_cube_pcd_sharded.py --shards "$SHARDS" --num-traj 10000 --seed 3072 --max-parallel "$MAXPAR" \
    --out ogbench/cube_single_expert_pcd.h5 --out-pix ogbench/cube_single_expert.h5 "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! (own session, survives terminal exit)"
echo "launch log: $LOG"
