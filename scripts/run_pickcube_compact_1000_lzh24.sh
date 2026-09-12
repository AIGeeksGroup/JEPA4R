#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
project_root=/public/home/lzh_24/project/JEPA-4R
python_bin="$project_root/envs/worldmodel/bin/python"
source_root="$project_root/data/raw_maniskill2_v0_5_3/PickCube-v0"
source_h5="$source_root/trajectory.h5"
source_json="$source_root/trajectory.json"
source_h5_sha256=a5920e22e4a9ae49e16d6c56aeb728a082d6ea9c32b62d979b6a95208f27dcd1
source_json_sha256=aab69b6f89a2e6f2b6c168b7b85a97d3d0ed5ca78326b6a80484660f42d1665f
output_root="$project_root/data/compact_rgb_proprio_1000/PickCube-v0"
temp_root=/dev/shm/lzh24-jepa4r-pickcube-compact-1000
status_file="${output_root}.status"
pid_file="${output_root}.pid"
log_file="${output_root}.log"
gpu_index="${PICKCUBE_GPU:-0}"

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
    mkdir -p "$(dirname "$output_root")"
    nohup env PICKCUBE_GPU="$gpu_index" "$0" --foreground \
        > "${log_file}.launcher" 2>&1 < /dev/null &
    printf 'detached_pid=%s\n' "$!"
    exit 0
fi
if [[ "${1:-}" != "--foreground" || $# -ne 1 ]]; then
    printf 'usage: %s --detach|--foreground\n' "$0" >&2
    exit 2
fi
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf 'PICKCUBE_GPU must be a non-negative integer\n' >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    printf 'missing world-model Python: %s\n' "$python_bin" >&2
    exit 2
fi
if [[ "$(sha256sum "$source_h5" | cut -d' ' -f1)" != "$source_h5_sha256" ]]; then
    printf 'source H5 SHA-256 mismatch\n' >&2
    exit 2
fi
if [[ "$(sha256sum "$source_json" | cut -d' ' -f1)" != "$source_json_sha256" ]]; then
    printf 'source JSON SHA-256 mismatch\n' >&2
    exit 2
fi
if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        printf 'refusing duplicate run: PID %s is active\n' "$existing_pid" >&2
        exit 2
    fi
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
shm_available_kib=$(df --output=avail /dev/shm | tail -n 1)
public_available_kib=$(df --output=avail "$(dirname "$output_root")" | tail -n 1)
if (( shm_available_kib < 20 * 1024 * 1024 )); then
    printf '/dev/shm has less than 20 GiB available\n' >&2
    exit 3
fi
if (( public_available_kib < 5 * 1024 * 1024 )); then
    printf '/public has less than 5 GiB available\n' >&2
    exit 3
fi

mkdir -p "$(dirname "$output_root")" "$temp_root"
printf '%s\n' "$$" > "${pid_file}.tmp.$$"
mv "${pid_file}.tmp.$$" "$pid_file"
touch "$log_file"
exec > >(stdbuf -oL tee -a "$log_file") 2>&1

finish() {
    exit_code=$?
    trap - EXIT
    if [[ $exit_code -eq 0 && -f "$output_root/manifest.json" ]]; then
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_root/envs/mpc_packages:$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu_index"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

git rev-parse HEAD > "${output_root}.git-head.txt"
git status --short --branch > "${output_root}.git-status.before.txt"
sha256sum "$source_h5" "$source_json" > "${output_root}.source.sha256"
nvidia-smi > "${output_root}.nvidia-smi.before.txt"
printf '%q ' "$python_bin" -m scripts.prepare_pickcube_compact_1000 \
    --source-h5 "$source_h5" \
    --source-json "$source_json" \
    --output-root "$output_root" \
    --temp-root "$temp_root" \
    --total-episodes 1000 \
    --batch-size 100 \
    --point-count 1024 \
    --seed 3072 > "${output_root}.command.txt"
printf '\n' >> "${output_root}.command.txt"

"$python_bin" -m scripts.prepare_pickcube_compact_1000 \
    --source-h5 "$source_h5" \
    --source-json "$source_json" \
    --output-root "$output_root" \
    --temp-root "$temp_root" \
    --total-episodes 1000 \
    --batch-size 100 \
    --point-count 1024 \
    --seed 3072
