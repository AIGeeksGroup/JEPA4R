#!/usr/bin/env bash
# 冻结世界模型 encoder 的 PPO（LiftCube-v0）：启动参数与 PPO v4/续训完全一致，只多 wm_ckpt，algo 为纯 PPO（无重建辅助损失）。
# 用法：bash scripts/train_ppo_frozen_wm.sh [latent_wm.pt 路径] [额外 hydra 覆盖...]
# 前提：世界模型训练已结束、显存空闲。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pprl
export MS2_ASSET_DIR=/root/autodl-tmp/pprl/data
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
cd /root/autodl-tmp/pprl
WM_CKPT=${1:-$(ls -td /root/autodl-tmp/pprl/log_data/latent_wm/*/ | head -1)latent_wm.pt}
shift || true
exec python scripts/train_ppo_frozen_wm.py wm_ckpt="$WM_CKPT" env=push_chair model=pointgpt_ppo env.name=LiftCube env.env_id=LiftCube-v0 \
  env.env_kwargs.control_mode=pd_ee_delta_pose env.env_kwargs.camera_cfgs.width=128 env.env_kwargs.camera_cfgs.height=128 \
  env.pcd_kwargs.n_target_points=0 env.pcd_kwargs.voxel_grid_size=0.01 env.pcd_kwargs.max_expected_num_points=6000 \
  env.continuous_task=True batch_B=24 batch_T=256 algo.minibatches=24 "$@"
