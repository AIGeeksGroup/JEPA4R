#!/usr/bin/env bash
# 从 PPO v4（wandb 590ttznw，2,236,416 步时环境子进程因点数 2512 > 2500 崩溃）的 model.pt 续训剩余约 76.8 万步。
# 与 v4 启动参数完全相同，只改两处：max_expected_num_points 2500 -> 6000（修复崩溃）、runner.n_steps=768000（剩余步数）。
# 运行前提：卡死的 v4 主进程（pid 31808，占约 26 GB 显存）已结束。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pprl
export MS2_ASSET_DIR=/root/autodl-tmp/pprl/data
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
cd /root/autodl-tmp/pprl
exec python scripts/train_ppo_resume.py env=push_chair model=pointgpt_ppo env.name=LiftCube env.env_id=LiftCube-v0 \
  env.env_kwargs.control_mode=pd_ee_delta_pose env.env_kwargs.camera_cfgs.width=128 env.env_kwargs.camera_cfgs.height=128 \
  env.pcd_kwargs.n_target_points=0 env.pcd_kwargs.voxel_grid_size=0.01 env.pcd_kwargs.max_expected_num_points=6000 \
  env.continuous_task=True batch_B=24 batch_T=256 algo.minibatches=24 runner.n_steps=768000 \
  +resume_model_path=/root/autodl-tmp/pprl/wandb/offline-run-20260906_130305-590ttznw/files/model.pt
