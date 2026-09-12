#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin=/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python
checkpoint=/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt
checkpoint_sha256=681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28
data_root=/public/home/lzh_24/project/JEPA-4R/data/compact_rgb_proprio/LiftCube-v0
run_dir=/public/home/lzh_24/project/JEPA-4R/runs/liftcube_sonata_proprio_goal_cem_10seeds_20260909
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

if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf 'LIFTCUBE_GPU must be a non-negative integer\n' >&2
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
    printf 'refusing to overwrite existing lzh-24 CEM output or state\n' >&2
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
export PYTHONPATH=/public/home/lzh_24/project/JEPA-4R/envs/mpc_packages:/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/lib/python3.11/site-packages:"$repo_dir"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
nvidia_smi_before=$(nvidia-smi)
git_status_before=$(git status --short --branch)
git_head=$(git rev-parse HEAD)
printf 'git_commit=%s\n' "$git_head"
printf 'checkpoint_sha256=%s\n' "$(sha256sum "$checkpoint" | cut -d' ' -f1)"
printf 'gpu_index=%s gpu_uuid=%s runner_pid=%s\n' "$gpu_index" "$gpu_uuid" "$$"

CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 "$python_bin" \
    scripts/check_liftcube_mpc_env.py \
    --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checkpoint_sha256" \
    --require-renderer

CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 stdbuf -oL -eL "$python_bin" \
    -m scripts.eval_liftcube_latent_cem \
    --checkpoint "$checkpoint" \
    --output "$run_dir" \
    --data-root "$data_root" \
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

printf '%s\n' "$nvidia_smi_before" > "$run_dir/nvidia-smi.before.txt"
printf '%s\n' "$git_status_before" > "$run_dir/git-status.before.txt"
printf '%s\n' "$git_head" > "$run_dir/git-head.txt"
test -f "$run_dir/summary.json"
