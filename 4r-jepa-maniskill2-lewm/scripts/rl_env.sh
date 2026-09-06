#!/usr/bin/env bash
# Source from any working directory. Changes apply only to this process.
TASK_RL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_RL_ROOT/vendor/ManiSkill2-Learn:$TASK_RL_ROOT/vendor/pytorch3d:$TASK_RL_ROOT/src"
export PIP_CACHE_DIR="$TASK_RL_ROOT/.cache/pip"
export XDG_CACHE_HOME="$TASK_RL_ROOT/.cache"
export TORCH_HOME="$TASK_RL_ROOT/.cache/torch"
export MS2_ASSET_DIR="$TASK_RL_ROOT/data/mani_skill2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export DISPLAY=""
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export XDG_RUNTIME_DIR="$TASK_RL_ROOT/.cache/runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
