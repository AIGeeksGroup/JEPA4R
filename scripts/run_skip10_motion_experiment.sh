#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-multistep-evaluation
python=/root/.venvs/4d-jepa-worldmodel/bin/python
git_bin=/nix/store/bcnisk3ydfgv26v2gw3zlky24g00yww2-git-2.54.0/bin/git
data=/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0
run_root=/workspace/users/lewm/runs/4d-jepa-architecture
run_dir=${run_root}/group1_skip10_motion_3k
status=${run_root}/group1_skip10_motion_3k.status

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$repo"
if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite existing run: $run_dir" >&2
    exit 1
fi

printf 'RUNNING\ncommit=%s\ncuda_visible_devices=%s\nstarted_utc=%s\n' \
    "$($git_bin rev-parse HEAD)" "$CUDA_VISIBLE_DEVICES" "$(date -u +%FT%TZ)" > "${status}.tmp"
mv "${status}.tmp" "$status"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "FAILED\nexit_code=%s\nfinished_utc=%s\n" "$code" "$(date -u +%FT%TZ)" > "${status}.tmp"; mv "${status}.tmp" "$status"; fi' EXIT

"$python" scripts/train_maniskill_group1.py \
    --config configs/maniskill_group1_skip10_motion.yaml \
    --run-dir "$run_dir"

for checkpoint in best last; do
    "$python" scripts/eval_maniskill_group1_rollout.py \
        --checkpoint "$run_dir/${checkpoint}.pt" \
        --data-root "$data" \
        --horizons 1 2 4 \
        --batch-size 4 \
        --output "$run_dir/rollout_metrics_${checkpoint}.json"
done

printf 'COMPLETE\ncommit=%s\ncuda_visible_devices=%s\nfinished_utc=%s\n' \
    "$($git_bin rev-parse HEAD)" "$CUDA_VISIBLE_DEVICES" "$(date -u +%FT%TZ)" > "${status}.tmp"
mv "${status}.tmp" "$status"
