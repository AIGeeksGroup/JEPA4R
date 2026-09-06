#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
export PYTHONPATH="$TASK_RL_ROOT/vendor/stable-worldmodel-lewm-compatible:$TASK_RL_ROOT/vendor/le-wm:$PYTHONPATH"
export STABLEWM_HOME="$TASK_RL_ROOT/artifacts/worldmodel"
export WANDB_MODE=disabled
export HF_HOME="$TASK_RL_ROOT/.cache/huggingface"
export SPT_CACHE_DIR="$TASK_RL_ROOT/.cache/stable-pretraining"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export MUJOCO_GL=egl
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
