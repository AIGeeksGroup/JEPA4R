#!/usr/bin/env bash
set -euo pipefail

repo=/public/home/lzh_24/project/JEPA-4R/worktrees/liftcube-joint-cost-contact
python=/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python
config=configs/maniskill_group1_sonata_contact_lzh24.yaml
run_dir=/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_contact_phase4_init2500_steps1000_20260910_v2
init_checkpoint=/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt
init_sha256=681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28
contact_sidecar=/public/home/lzh_24/project/JEPA-4R/runs/liftcube_frozen_probe_20260910_v2/contact_sidecar.npz
contact_sha256=7c3868dcbeaf102e50461dc82f88c8cf605515c12692c0e47fc2199e5353af4f

if [[ -z "${LIFTCUBE_GPU:-}" ]] || ! [[ "$LIFTCUBE_GPU" =~ ^[0-9]+$ ]]; then
  echo "LIFTCUBE_GPU must name one verified idle GPU index" >&2
  exit 2
fi

cd "$repo"
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
  | awk -F ', ' -v target="$LIFTCUBE_GPU" '$1 == target {print $2}')
if [[ -z "$gpu_uuid" ]]; then
  echo "GPU index $LIFTCUBE_GPU does not exist" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {found=1} END {exit !found}'; then
  echo "GPU $LIFTCUBE_GPU ($gpu_uuid) already has a compute process" >&2
  exit 3
fi
printf '%s  %s\n' "$init_sha256" "$init_checkpoint" | sha256sum --check --status
printf '%s  %s\n' "$contact_sha256" "$contact_sidecar" | sha256sum --check --status
if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "run directory is not empty: $run_dir" >&2
  exit 4
fi
if [[ -n "$(git status --short)" ]]; then
  echo "training worktree must be clean" >&2
  git status --short >&2
  exit 5
fi

mkdir -p "$run_dir"
nvidia-smi > "$run_dir/nvidia-smi.before.txt"
git status --short --branch > "$run_dir/git-status.before.txt"
git rev-parse HEAD > "$run_dir/git-head.txt"
sha256sum "$config" > "$run_dir/config.sha256"
sha256sum "$init_checkpoint" "$contact_sidecar" > "$run_dir/input.sha256"
"$python" - <<'PY' > "$run_dir/runtime.txt"
import sys
import torch
print(f"python={sys.version}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

command=(
  "$python" -u scripts/train_maniskill_group1.py
  --config "$config"
  --run-dir "$run_dir"
  --init-checkpoint "$init_checkpoint"
)
printf 'CUDA_VISIBLE_DEVICES=%s ' "$LIFTCUBE_GPU" > "$run_dir/command.txt"
printf '%q ' "${command[@]}" >> "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

nohup env CUDA_VISIBLE_DEVICES="$LIFTCUBE_GPU" "${command[@]}" \
  > "$run_dir/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf 'launched pid=%s gpu=%s uuid=%s log=%s\n' \
  "$pid" "$LIFTCUBE_GPU" "$gpu_uuid" "$run_dir/train.log"
