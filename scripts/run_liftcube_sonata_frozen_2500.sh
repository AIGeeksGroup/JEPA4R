#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-sonata-ptv3
python="${WORLDMODEL_PYTHON:-/nix/lewm/4d-jepa-worldmodel/bin/python}"
git_bin="${GIT_BIN:-$(command -v git)}"
data=/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0
run_root=/workspace/users/lewm/runs/4d-jepa-architecture
name=group1_skip10_sonata_frozen_batch8_2500
run_dir="${run_root}/${name}"
status="${run_root}/${name}.status"
pid_file="${run_root}/${name}.pid"
log="${run_root}/${name}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

if [[ -f "$pid_file" ]]; then
    previous_pid=$(<"$pid_file")
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        echo "refusing to duplicate live process: $previous_pid" >&2
        exit 1
    fi
fi
if [[ -e "$run_dir" || -e "$status" || -e "$pid_file" || -e "$log" ]]; then
    echo "refusing to overwrite existing run: $run_dir" >&2
    exit 1
fi

mkdir -p "$run_root"
commit="$($git_bin -C "$repo" rev-parse HEAD)"

write_status() {
    local state="$1"
    local exit_code="${2:-0}"
    local temporary="${status}.tmp.$$"
    printf '%s\npid=%s\nexit_code=%s\ncommit=%s\ncuda_visible_devices=%s\nutc=%s\n' \
        "$state" "$$" "$exit_code" "$commit" "$CUDA_VISIBLE_DEVICES" \
        "$(date -u +%FT%TZ)" > "$temporary"
    mv "$temporary" "$status"
}

printf '%s\n' "$$" > "${pid_file}.tmp"
mv "${pid_file}.tmp" "$pid_file"
write_status PREPARING
exec >>"$log" 2>&1

on_exit() {
    code=$?
    if [[ $code -ne 0 ]]; then
        write_status FAILED "$code"
    fi
}
trap on_exit EXIT

cd "$repo"
write_status RUNNING
"$python" scripts/train_maniskill_group1.py \
    --config configs/maniskill_group1_sonata_frozen.yaml \
    --run-dir "$run_dir"

write_status EVALUATING
for checkpoint_file in \
    checkpoint_000500.pt checkpoint_001000.pt checkpoint_001500.pt \
    checkpoint_002000.pt checkpoint_002500.pt best.pt; do
    checkpoint="$run_dir/$checkpoint_file"
    if [[ ! -f "$checkpoint" ]]; then
        echo "missing scheduled checkpoint: $checkpoint" >&2
        exit 1
    fi
    checkpoint_name="${checkpoint_file%.pt}"
    "$python" -m scripts.eval_liftcube_latent_diagnostics \
        --checkpoint "$checkpoint" \
        --data-root "$data" \
        --output "$run_dir/latent_diagnostics_${checkpoint_name}.json" \
        --horizons 1 2 4 \
        --batch-size 8 \
        --device cuda
done

write_status COMPLETE
