#!/usr/bin/env bash
# Runtime contract for the local OGBench-Franka worktree.
_FRANKA_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LEWM_PCD_ROOT="$_FRANKA_ENV_DIR"
: "${STABLEWM_HOME:?Set STABLEWM_HOME to a persistent data, log, and checkpoint directory before sourcing env_franka.sh}"
export UPSTREAM_SWM_ROOT="${UPSTREAM_SWM_ROOT:-$LEWM_PCD_ROOT/third_party/stable-worldmodel}"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$STABLEWM_HOME/logs/train"
export SPT_CACHE_DIR="$STABLEWM_HOME/checkpoints/_spt_cache"
export PYTHONPATH="$LEWM_PCD_ROOT:$UPSTREAM_SWM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset _FRANKA_ENV_DIR
