#!/usr/bin/env bash
set -euo pipefail

P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$P/env_franka.sh"
PY=${PYTHON:-python}
STATUS="$STABLEWM_HOME/logs/franka_collection.status"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$STABLEWM_HOME/logs/collection/$STAMP"
mkdir -p "$(dirname "$STATUS")" "$LOG_DIR"

write_status() {
    local stage=$1 dataset=$2 tmp
    tmp=$(mktemp "${STATUS}.tmp.XXXXXX")
    printf 'stage=%s time=%s pid=%s dataset=%s log_dir=%s\n' \
        "$stage" "$(date --iso-8601=seconds)" "$$" "$dataset" "$LOG_DIR" > "$tmp"
    mv "$tmp" "$STATUS"
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        write_status "FAILED exit_code=$rc" "${CURRENT_DATASET:-none}"
    fi
}
trap on_exit EXIT

refuse_existing() {
    if [ -e "$1" ]; then
        echo "Refusing to overwrite existing output: $1" >&2
        exit 2
    fi
}

collect_shards() {
    local stage=$1 out=$2 episodes=$3 shards=$4 parallel=$5 resume=${6:-0}
    local final="$STABLEWM_HOME/datasets/$out"
    CURRENT_DATASET="$final"
    if [ "$resume" -eq 0 ]; then
        refuse_existing "$final"
    fi
    write_status "COLLECTING_$stage" "$final"
    cd "$P"
    local args=(--shards "$shards" --num-traj "$episodes" --seed 3072 --out "$out" \
        --max-parallel "$parallel" --no-merge)
    if [ "$resume" -eq 1 ]; then
        args+=(--resume)
    fi
    "$PY" scripts/data/collect_cube_pcd_sharded.py "${args[@]}" \
        -- --config-name=ogb_franka_pcd \
        2>&1 | tee "$LOG_DIR/${stage,,}_shards.log"
}

publish_and_check() {
    local stage=$1 out=$2 episodes=$3 shards=$4 parallel=$5
    local final="$STABLEWM_HOME/datasets/$out"
    local report="$LOG_DIR/$(basename "${final%.h5}").outcomes.json"
    CURRENT_DATASET="$final"
    write_status "PUBLISHING_$stage" "$final"
    cd "$P"
    "$PY" scripts/data/collect_cube_pcd_sharded.py \
        --shards "$shards" --num-traj "$episodes" --seed 3072 --out "$out" \
        --max-parallel "$parallel" --resume -- --config-name=ogb_franka_pcd \
        2>&1 | tee "$LOG_DIR/${stage,,}_publish.log"
    test -s "$final"
    write_status "SUMMARIZING_$stage" "$final"
    "$PY" scripts/data/summarize_episode_outcomes.py "$final" --out "$report" \
        2>&1 | tee "$LOG_DIR/${stage,,}_outcomes.log"
    write_status "VALIDATING_$stage" "$final"
    "$PY" scripts/checks/verify_franka_replay.py "$final" --expected-episodes "$episodes" \
        --validate-only --report "$LOG_DIR/$(basename "${final%.h5}").validation.json" \
        2>&1 | tee "$LOG_DIR/${stage,,}_validate.log"
}

postprocess_full() {
    local raw="$STABLEWM_HOME/datasets/ogbench/franka_full/cube_franka_single_expert_pcd.h5"
    local rechunked="$STABLEWM_HOME/datasets/ogbench/franka_full/cube_franka_single_expert_pcd.rechunk20.h5"
    local partial="${rechunked}.partial"
    CURRENT_DATASET="$raw"
    write_status "POSTPROCESSING_FULL" "$raw"
    cd "$P"
    "$PY" scripts/data/add_eval_columns.py "$raw" 2>&1 | tee "$LOG_DIR/full_add_eval_columns.log"
    if [ -s "$rechunked" ]; then
        echo "Reusing completed rechunked output: $rechunked" | tee "$LOG_DIR/full_rechunk.log"
    else
        refuse_existing "$rechunked"
        if [ -e "$partial" ]; then
            echo "Removing incomplete rechunk output: $partial" | tee "$LOG_DIR/full_rechunk.log"
            rm -f -- "$partial"
        fi
        "$PY" scripts/data/rechunk_h5.py "$raw" "$partial" --rows 20 \
            2>&1 | tee -a "$LOG_DIR/full_rechunk.log"
        test -s "$partial"
        mv "$partial" "$rechunked"
    fi
    write_status "VALIDATING_RECHUNKED_FULL" "$rechunked"
    "$PY" scripts/checks/verify_franka_replay.py "$rechunked" --expected-episodes 10000 \
        --validate-only --report "$LOG_DIR/full_rechunk_validation.json" \
        2>&1 | tee "$LOG_DIR/full_rechunk_validate.log"
}

pilot() {
    local out=ogbench/franka_pilot/cube_franka_pcd_pilot_500.h5
    collect_shards PILOT "$out" 500 10 5
    publish_and_check PILOT "$out" 500 10 5
    write_status "PILOT_ACCEPTED" "$STABLEWM_HOME/datasets/$out"
}

full() {
    local out=ogbench/franka_full/cube_franka_single_expert_pcd.h5
    collect_shards FULL "$out" 10000 20 8
    publish_and_check FULL "$out" 10000 20 8
    postprocess_full
    write_status "FULL_ACCEPTED" "$STABLEWM_HOME/datasets/$out"
}

resume_full() {
    local out=ogbench/franka_full/cube_franka_single_expert_pcd.h5
    local final="$STABLEWM_HOME/datasets/$out"
    if [ -e "$final" ]; then
        CURRENT_DATASET="$final"
        write_status "RESUMING_POSTPROCESS_FULL" "$final"
    else
        collect_shards FULL "$out" 10000 20 8 1
        publish_and_check FULL "$out" 10000 20 8
    fi
    postprocess_full
    write_status "FULL_ACCEPTED" "$STABLEWM_HOME/datasets/$out"
}

case "${1:-}" in
    pilot) pilot ;;
    full) full ;;
    resume-full) resume_full ;;
    all) pilot; full ;;
    *)
        echo "usage: $0 {pilot|full|resume-full|all}" >&2
        exit 2
        ;;
esac
