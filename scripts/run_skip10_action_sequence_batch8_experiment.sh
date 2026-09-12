#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-multistep-evaluation
python=/nix/lewm/4d-jepa-worldmodel/bin/python
git_bin=/nix/store/bcnisk3ydfgv26v2gw3zlky24g00yww2-git-2.54.0/bin/git
data=/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0
run_root=/workspace/users/lewm/runs/4d-jepa-architecture
name=group1_skip10_action_sequence_batch8_2500
run_dir=${run_root}/${name}
status=${run_root}/${name}.status
pid_file=${run_root}/${name}.pid
log=${run_root}/${name}.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cd "$repo"
if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite existing run: $run_dir" >&2
    exit 1
fi
if [[ -f "$pid_file" ]]; then
    previous_pid=$(<"$pid_file")
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        echo "refusing to duplicate live process: $previous_pid" >&2
        exit 1
    fi
fi

exec >>"$log" 2>&1
commit="$($git_bin rev-parse HEAD)"
printf '%s\n' "$$" > "${pid_file}.tmp"
mv "${pid_file}.tmp" "$pid_file"
printf 'RUNNING\npid=%s\ncommit=%s\ncuda_visible_devices=%s\nstarted_utc=%s\n' \
    "$$" "$commit" "$CUDA_VISIBLE_DEVICES" "$(date -u +%FT%TZ)" > "${status}.tmp"
mv "${status}.tmp" "$status"

on_exit() {
    code=$?
    if [[ $code -ne 0 ]]; then
        printf 'FAILED\npid=%s\nexit_code=%s\ncommit=%s\nfinished_utc=%s\n' \
            "$$" "$code" "$commit" "$(date -u +%FT%TZ)" > "${status}.tmp"
        mv "${status}.tmp" "$status"
    fi
}
trap on_exit EXIT

"$python" scripts/train_maniskill_group1.py \
    --config configs/maniskill_group1_skip10_action_sequence_batch8.yaml \
    --run-dir "$run_dir"

for checkpoint in best last; do
    "$python" scripts/eval_maniskill_group1_rollout.py \
        --checkpoint "$run_dir/${checkpoint}.pt" \
        --data-root "$data" \
        --horizons 1 2 4 \
        --batch-size 8 \
        --output "$run_dir/rollout_metrics_${checkpoint}.json"
done

printf 'COMPLETE\npid=%s\ncommit=%s\ncuda_visible_devices=%s\nfinished_utc=%s\n' \
    "$$" "$commit" "$CUDA_VISIBLE_DEVICES" "$(date -u +%FT%TZ)" > "${status}.tmp"
mv "${status}.tmp" "$status"
