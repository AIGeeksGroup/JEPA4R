#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
project_dir="$(cd -- "$repo_dir/.." && pwd)"
python_bin="$project_dir/envs/worldmodel/bin/python"
checkpoint="$project_dir/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt"
checkpoint_sha256=681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28
data_root="$project_dir/data/compact_rgb_proprio/LiftCube-v0"
run_dir="${LIFTCUBE_RUN_DIR:-$project_dir/runs/liftcube_model_sim_ranking_10states_20260909}"
status_file="${run_dir}.status"
pid_file="${run_dir}.pid"
log_file="${run_dir}.log"
gpu_index="${LIFTCUBE_GPU:-0}"

write_status() {
    local value=$1
    local temporary="${status_file}.tmp.$$"
    printf '%s\n' "$value" > "$temporary"
    mv "$temporary" "$status_file"
}

if [[ "${1:-}" == "--detach" ]]; then
    shift
    if [[ $# -ne 0 ]]; then
        printf 'unexpected arguments with --detach\n' >&2
        exit 2
    fi
    nohup env LIFTCUBE_GPU="$gpu_index" "$0" --foreground \
        > "${log_file}.launcher" 2>&1 < /dev/null &
    printf 'detached_pid=%s\n' "$!"
    exit 0
fi
if [[ "${1:-}" != "--foreground" && $# -ne 0 ]]; then
    printf 'usage: %s [--detach]\n' "$0" >&2
    exit 2
fi

if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf 'LIFTCUBE_GPU must be a non-negative integer\n' >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    printf 'missing world-model Python: %s\n' "$python_bin" >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    printf 'missing checkpoint: %s\n' "$checkpoint" >&2
    exit 2
fi
if [[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" != "$checkpoint_sha256" ]]; then
    printf 'checkpoint sha256 mismatch\n' >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    printf 'missing data root: %s\n' "$data_root" >&2
    exit 2
fi
if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        printf 'refusing duplicate run: PID %s is active\n' "$existing_pid" >&2
        exit 2
    fi
fi
if [[ -e "$run_dir" || -e "$status_file" || -e "$log_file" || -e "$pid_file" ]]; then
    printf 'refusing to overwrite existing ranking output or state\n' >&2
    exit 2
fi

gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F ', ' -v target="$gpu_index" '$1 == target {print $2}')
if [[ -z "$gpu_uuid" ]]; then
    printf 'GPU %s does not exist\n' "$gpu_index" >&2
    exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits \
    | awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {found=1} END {exit !found}'; then
    printf 'GPU %s (%s) already has a compute process\n' "$gpu_index" "$gpu_uuid" >&2
    exit 3
fi

mkdir -p "$(dirname "$run_dir")"
printf '%s\n' "$$" > "${pid_file}.tmp.$$"
mv "${pid_file}.tmp.$$" "$pid_file"
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
export PYTHONPATH="$project_dir/envs/mpc_packages:$project_dir/envs/worldmodel/lib/python3.11/site-packages:$repo_dir"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

git_head=$(git rev-parse HEAD)
git_status_before=$(git status --short --branch)
data_manifest_sha256=$(find "$data_root" -type f -printf '%P\n' | sort | sha256sum | cut -d' ' -f1)
config_json="${run_dir}.config.json"
cat > "${config_json}.tmp.$$" <<EOF
{
  "task": "LiftCube-v0 model-vs-simulator ranking audit",
  "repo": "$repo_dir",
  "git_head": "$git_head",
  "checkpoint": "$checkpoint",
  "checkpoint_sha256": "$checkpoint_sha256",
  "data_root": "$data_root",
  "data_manifest_sha256": "$data_manifest_sha256",
  "gpu_index": $gpu_index,
  "gpu_uuid": "$gpu_uuid",
  "state_seeds": [4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109],
  "candidates": 128,
  "model_batch_size": 32,
  "iterations": 5,
  "elites": 16,
  "horizon": 2,
  "chunk_size": 10,
  "render": true
}
EOF
mv "${config_json}.tmp.$$" "$config_json"
nvidia-smi > "${run_dir}.nvidia-smi.before.txt"
printf '%s\n' "$git_status_before" > "${run_dir}.git-status.before.txt"
printf '%s\n' "$git_head" > "${run_dir}.git-head.txt"
printf '%s\n' "$data_manifest_sha256" > "${run_dir}.data-manifest.sha256"
printf 'runner_pid=%s gpu_index=%s gpu_uuid=%s\n' "$$" "$gpu_index" "$gpu_uuid"

CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 stdbuf -oL -eL "$python_bin" \
    -m scripts.eval_liftcube_model_sim_ranking \
    --checkpoint "$checkpoint" \
    --output "$run_dir" \
    --data-root "$data_root" \
    --state-seeds 4100 4101 4102 4103 4104 4105 4106 4107 4108 4109 \
    --candidates 128 \
    --model-batch-size 32 \
    --iterations 5 \
    --elites 16 \
    --horizon 2 \
    --chunk-size 10 \
    --device cuda \
    --render

nvidia-smi > "$run_dir/nvidia-smi.after.txt"
mv "$config_json" "$run_dir/config.json"
mv "${run_dir}.nvidia-smi.before.txt" "$run_dir/nvidia-smi.before.txt"
mv "${run_dir}.git-status.before.txt" "$run_dir/git-status.before.txt"
mv "${run_dir}.git-head.txt" "$run_dir/git-head.txt"
mv "${run_dir}.data-manifest.sha256" "$run_dir/data-manifest.sha256"
test -f "$run_dir/states.jsonl"
test -f "$run_dir/summary.json"
test -d "$run_dir/videos"
