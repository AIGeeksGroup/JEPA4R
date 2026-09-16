#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${LEWM_PCD_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
DATA=${MANISKILL_PICKCUBE_DATA_ROOT:-/root/autodl-tmp/workspace/users/lewm/datasets/maniskill/pickcube}
PY=${LEWM_MPC_PY:-/root/autodl-tmp/conda/envs/maniskill-pickcube/bin/python}
OUT="$DATA/derived/object_aware_v2"
SHARDS="$OUT/episodes"
LOG="$DATA/logs/object_aware_v2_replay_1000.log"
STATUS="$DATA/logs/object_aware_v2_replay_1000.status"

mkdir -p "$SHARDS" "$DATA/logs"
printf 'RUNNING started=%s pid=%s\n' "$(date --iso-8601=seconds)" "$$" > "$STATUS"

finish() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        printf 'COMPLETE finished=%s exit_code=0\n' "$(date --iso-8601=seconds)" > "$STATUS"
    else
        printf 'FAILED finished=%s exit_code=%s\n' "$(date --iso-8601=seconds)" "$rc" > "$STATUS"
    fi
}
trap finish EXIT

cd "$REPO"
"$PY" scripts/data/collect_pickcube_object_aware.py \
    --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
    --trajectory-h5 "$DATA/raw/PickCube-v1/motionplanning/trajectory.h5" \
    --output-dir "$SHARDS" \
    --start-episode 0 \
    --num-episodes 1000 \
    --extra-resolution 384 \
    --voxel-size 0.0025 2>&1 | tee -a "$LOG"

"$PY" scripts/checks/check_pickcube_object_aware.py \
    --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
    --trajectory-h5 "$DATA/raw/PickCube-v1/motionplanning/trajectory.h5" \
    --shard-dir "$SHARDS" \
    --expected-episodes 1000 \
    --extra-resolution 384 \
    --voxel-size 0.0025 \
    --summary-json "$OUT/validation_summary_1000.json" \
    --manifest-json "$OUT/manifest.json" 2>&1 | tee -a "$LOG"

"$PY" scripts/data/merge_pickcube_maniskill.py \
    --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
    --shard-dir "$SHARDS" \
    --output-h5 "$OUT/pickcube_object_aware_v2.h5" \
    --num-episodes 1000 \
    --num-points 1200 \
    --sampling-protocol object_aware_v2 \
    --extra-resolution 384 \
    --voxel-size 0.0025 2>&1 | tee -a "$LOG"
