#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
cd "$TASK_RL_ROOT"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  smoke)
    exec nice -n 10 "$TASK_RL_ROOT/.venv-rl/bin/python" "$TASK_RL_ROOT/scripts/launch_official_rl.py" \
      "$TASK_RL_ROOT/configs/ppo_pointnet_identity.py" \
      --work-dir "$TASK_RL_ROOT/artifacts/rl_identity_smoke" --num-gpus 0 --seed 20260906 \
      --cfg-options env_cfg.env_name=PickCube-v0 env_cfg.obs_mode=pointcloud \
      env_cfg.control_mode=pd_ee_delta_pose env_cfg.obs_frame=world env_cfg.n_points=1200 env_cfg.horizon=16 \
      rollout_cfg.num_procs=1 train_cfg.total_steps=32 train_cfg.n_steps=32 \
      train_cfg.n_checkpoint=32 train_cfg.n_log=32 train_cfg.n_eval=None \
      replay_cfg.capacity=32 agent_cfg.batch_size=16 eval_cfg=None "$@"
    ;;
  eval-smoke)
    exec nice -n 10 "$TASK_RL_ROOT/.venv-rl/bin/python" "$TASK_RL_ROOT/scripts/launch_official_rl.py" \
      "$TASK_RL_ROOT/configs/ppo_pointnet_identity.py" \
      --work-dir "$TASK_RL_ROOT/artifacts/rl_identity_smoke" --num-gpus 0 --seed 20260906 \
      --evaluation --resume-from "$TASK_RL_ROOT/artifacts/rl_identity_smoke/models/model_final.ckpt" \
      --cfg-options env_cfg.env_name=PickCube-v0 env_cfg.obs_mode=pointcloud \
      env_cfg.control_mode=pd_ee_delta_pose env_cfg.obs_frame=world env_cfg.n_points=1200 env_cfg.horizon=16 \
      eval_cfg.num=2 eval_cfg.num_procs=1 eval_cfg.save_video=False "$@"
    ;;
  train)
    # Full 5M-step PPO budget, shared Identity encoder, one renderer and CPU model.
    exec nice -n 10 "$TASK_RL_ROOT/.venv-rl/bin/python" "$TASK_RL_ROOT/scripts/launch_official_rl.py" \
      "$TASK_RL_ROOT/configs/ppo_pointnet_identity.py" \
      --work-dir "$TASK_RL_ROOT/artifacts/rl_identity_train" --num-gpus 0 --seed 20260906 \
      --cfg-options env_cfg.env_name=PickCube-v0 env_cfg.obs_mode=pointcloud \
      env_cfg.control_mode=pd_ee_delta_pose env_cfg.obs_frame=world env_cfg.n_points=1200 rollout_cfg.num_procs=1 eval_cfg=None "$@"
    ;;
  *) echo "Usage: $0 {smoke|eval-smoke|train} [upstream CLI overrides]" >&2; exit 2 ;;
esac
