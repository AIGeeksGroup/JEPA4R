#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${LEWM_PCD_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DATA=${MANISKILL_PICKCUBE_DATA_ROOT:-/root/autodl-tmp/workspace/users/lewm/datasets/maniskill/pickcube}
RUN_ROOT=${MANISKILL_RUN_ROOT:-/root/autodl-tmp/workspace/users/lewm/maniskill-pcd}
TRAIN_PY=${LEWM_TRAIN_PY:-/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python}
MPC_PY=${LEWM_MPC_PY:-/root/autodl-tmp/conda/envs/maniskill-pickcube/bin/python}
UPSTREAM_SWM=${STABLE_WORLDMODEL_ROOT:-/root/autodl-tmp/workspace/users/lewm/JEPA4R/lewm_pcd/third_party/stable-worldmodel}
DATA_STATUS="$DATA/logs/object_aware_v2_replay_1000.status"
OUT="$DATA/derived/object_aware_v2"
SHARDS="$OUT/episodes"
RUN_NAME=pickcube_object_aware_pcd_ms2_10ep_seed3072
WEIGHTS="$RUN_ROOT/checkpoints/$RUN_NAME/weights_epoch_10.pt"
STATUS="$RUN_ROOT/logs/object_aware_pipeline.status"

mkdir -p "$RUN_ROOT/logs/train" "$RUN_ROOT/logs/eval" "$RUN_ROOT/datasets/maniskill"

stage() {
    printf '%s time=%s pid=%s\n' "$1" "$(date --iso-8601=seconds)" "$$" > "$STATUS"
}

finish() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'FAILED exit_code=%s time=%s pid=%s\n' "$rc" "$(date --iso-8601=seconds)" "$$" > "$STATUS"
    fi
}
trap finish EXIT

stage WAITING_DATA
while true; do
    data_state=$(head -n 1 "$DATA_STATUS" 2>/dev/null || true)
    case "$data_state" in
        COMPLETE*) break ;;
        FAILED*) echo "object-aware replay failed: $data_state" >&2; exit 3 ;;
        *) sleep 60 ;;
    esac
done

cd "$REPO"
export LEWM_PCD_ROOT="$REPO"
export STABLEWM_HOME="$RUN_ROOT"
export SPT_CACHE_DIR="$RUN_ROOT/checkpoints/_spt_cache"
export WANDB_DIR="$RUN_ROOT/logs/train"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$REPO:$UPSTREAM_SWM"

stage PREPARING_TRAIN_H5
for split in train eval; do
    if [ "$split" = train ]; then
        start=0
        count=900
    else
        start=900
        count=100
    fi
    "$MPC_PY" scripts/data/merge_pickcube_maniskill.py \
        --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
        --shard-dir "$SHARDS" \
        --output-h5 "$RUN_ROOT/datasets/maniskill/pickcube_object_aware_v2_${split}.h5" \
        --start-episode "$start" \
        --num-episodes "$count" \
        --num-points 1200 \
        --sampling-protocol object_aware_v2 \
        --extra-resolution 384 \
        --voxel-size 0.0025 2>&1 | tee "$RUN_ROOT/logs/${split}_merge_object_aware_v2.log"
done

stage VERIFYING
grep -q '"status": "PASS"' "$OUT/validation_summary_1000.json"
test -s "$RUN_ROOT/datasets/maniskill/pickcube_object_aware_v2_train.h5"
test -s "$RUN_ROOT/datasets/maniskill/pickcube_object_aware_v2_eval.h5"
"$MPC_PY" -m pytest -q \
    scripts/checks/test_maniskill_pickcube_data.py \
    scripts/checks/test_maniskill_train_config.py \
    scripts/checks/test_maniskill_eval.py \
    scripts/checks/test_object_aware_merge.py

if [ ! -f "$WEIGHTS" ]; then
    if [ -d "$RUN_ROOT/checkpoints/$RUN_NAME" ]; then
        echo "Refusing to overwrite partial formal checkpoint directory: $RUN_ROOT/checkpoints/$RUN_NAME" >&2
        exit 2
    fi
    stage TRAINING
    "$TRAIN_PY" train.py \
        data=maniskill_pickcube_pcd \
        data.dataset.name=maniskill/pickcube_object_aware_v2_train.h5 \
        model=lewm_pcd \
        subdir="$RUN_NAME" \
        output_model_name="$RUN_NAME" \
        trainer.max_epochs=10 \
        loader.batch_size=128 \
        wandb.enabled=false 2>&1 | tee "$RUN_ROOT/logs/train/$RUN_NAME.log"
fi
test -s "$WEIGHTS"

stage CEM_SMOKE
"$MPC_PY" eval_maniskill.py \
    --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
    --trajectory-h5 "$DATA/raw/PickCube-v1/motionplanning/trajectory.h5" \
    --shard-dir "$SHARDS" \
    --train-h5 "$RUN_ROOT/datasets/maniskill/pickcube_object_aware_v2_train.h5" \
    --checkpoint "$WEIGHTS" \
    --output-json "$RUN_ROOT/logs/eval/cem_object_aware_smoke_episode900.json" \
    --start-episode 900 \
    --num-episodes 1 \
    --sampling-protocol object_aware_v2 \
    --extra-resolution 384 \
    --voxel-size 0.0025 \
    --num-samples 32 \
    --cem-steps 3 \
    --topk 8 2>&1 | tee "$RUN_ROOT/logs/eval/cem_object_aware_smoke_episode900.log"

stage CEM_100
"$MPC_PY" eval_maniskill.py \
    --metadata-json "$DATA/raw/PickCube-v1/motionplanning/trajectory.json" \
    --trajectory-h5 "$DATA/raw/PickCube-v1/motionplanning/trajectory.h5" \
    --shard-dir "$SHARDS" \
    --train-h5 "$RUN_ROOT/datasets/maniskill/pickcube_object_aware_v2_train.h5" \
    --checkpoint "$WEIGHTS" \
    --output-json "$RUN_ROOT/logs/eval/cem_object_aware_100_episodes_900_999.json" \
    --start-episode 900 \
    --num-episodes 100 \
    --sampling-protocol object_aware_v2 \
    --extra-resolution 384 \
    --voxel-size 0.0025 \
    --num-samples 300 \
    --cem-steps 30 \
    --topk 30 2>&1 | tee "$RUN_ROOT/logs/eval/cem_object_aware_100_episodes_900_999.log"

stage COMPLETE
