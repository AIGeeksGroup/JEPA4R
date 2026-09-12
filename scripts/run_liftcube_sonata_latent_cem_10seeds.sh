#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin=/nix/lewm/4d-jepa-mpc/bin/python
checkpoint=/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500/best.pt
checkpoint_sha256=38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32
run_dir=/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_sonata_latent_cem_best_10seeds
status_file="${run_dir}.status"
pid_file="${run_dir}.pid"
log_file="${run_dir}.log"

write_status() {
    local value=$1
    local temporary="${status_file}.tmp.$$"
    printf '%s\n' "$value" > "$temporary"
    mv "$temporary" "$status_file"
}

if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        printf 'refusing duplicate run: PID %s is active\n' "$existing_pid" >&2
        exit 2
    fi
fi
if [[ -e "$run_dir" || -e "$status_file" || -e "$log_file" ]]; then
    printf 'refusing to overwrite existing Sonata latent CEM output or state\n' >&2
    exit 2
fi
if [[ -e "$pid_file" ]]; then
    printf 'refusing to overwrite existing PID file: %s\n' "$pid_file" >&2
    exit 2
fi

mkdir -p "$(dirname "$run_dir")"
pid_temporary="${pid_file}.tmp.$$"
printf '%s\n' "$$" > "$pid_temporary"
mv "$pid_temporary" "$pid_file"
touch "$log_file"
exec > >(stdbuf -oL tee -a "$log_file") 2>&1

finish() {
    exit_code=$?
    trap - EXIT
    if [[ $exit_code -eq 0 && -f "$run_dir/summary.json" ]]; then
        write_status COMPLETE
    else
        write_status FAILED
    fi
    exit "$exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_status RUNNING
cd "$repo_dir"
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'checkpoint_sha256=%s\n' "$(sha256sum "$checkpoint" | cut -d' ' -f1)"
printf 'runner_pid=%s CUDA_VISIBLE_DEVICES=2\n' "$$"

CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 "$python_bin" \
    scripts/check_liftcube_mpc_env.py \
    --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha256" \
    --require-renderer

CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 stdbuf -oL -eL "$python_bin" \
    -m scripts.eval_liftcube_latent_cem \
    --checkpoint "$checkpoint" \
    --output "$run_dir" \
    --data-root /workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0 \
    --seeds 4100 4101 4102 4103 4104 4105 4106 4107 4108 4109 \
    --cem-seed 31000 \
    --candidates 128 \
    --model-batch-size 32 \
    --iterations 5 \
    --elites 16 \
    --horizon 2 \
    --chunk-size 10 \
    --max-steps 200 \
    --device cuda

test -f "$run_dir/summary.json"
