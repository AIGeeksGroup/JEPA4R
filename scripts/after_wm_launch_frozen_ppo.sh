#!/usr/bin/env bash
# 链式任务（用户已批准）：等世界模型训练进程结束 -> 检查产物 -> 真实环境小规模冒烟 -> 正式启动冻结 latent2 的 PPO。
set -uo pipefail
cd /root/autodl-tmp/pprl
WM_LOG=/root/autodl-tmp/logs/40_train_latent_wm_liftcube_fs1.log
WM_DIR=$(ls -td /root/autodl-tmp/pprl/log_data/latent_wm/*/ | head -1); WM_DIR=${WM_DIR%/}
SMOKE_LOG=/root/autodl-tmp/logs/38_frozen_ppo_smoke.log
RUN_LOG=/root/autodl-tmp/logs/39_train_ppo_frozen_wm_liftcube.log
echo "[chain] 等待世界模型训练结束 $(date '+%F %T')"
while pgrep -f "train_latent_wm.p[y]" >/dev/null; do sleep 60; done
echo "[chain] 世界模型进程已结束 $(date '+%F %T')"
if ! grep -q "^完成" "$WM_LOG"; then echo "[chain] 世界模型未正常完成（日志无“完成”行），不启动 PPO"; tail -5 "$WM_LOG"; exit 1; fi
ls -la "$WM_DIR/latent_wm.pt" "$WM_DIR"/weights_epoch_10.pt 2>&1
sleep 20
echo "[chain] 冒烟开始 $(date '+%T')：batch_B=4 batch_T=64 两次迭代 + 评估"
timeout 1200 bash scripts/train_ppo_frozen_wm.sh "$WM_DIR/latent_wm.pt" batch_B=4 batch_T=64 algo.minibatches=2 runner.n_steps=512 runner.log_interval_steps=256 runner.eval_interval_steps=512 eval.n_eval_envs=2 > "$SMOKE_LOG" 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q "elapsed_steps" "$SMOKE_LOG" || grep -q "Traceback" "$SMOKE_LOG"; then
  echo "[chain] 冒烟失败 rc=$rc，不启动正式训练"; grep -n -E "Traceback|Error" -A 12 "$SMOKE_LOG" | tail -30; exit 1
fi
echo "[chain] 冒烟通过 $(date '+%T')，启动正式训练（参数同 v4，encoder=冻结世界模型 latent2）"
sleep 10
setsid nohup bash scripts/train_ppo_frozen_wm.sh "$WM_DIR/latent_wm.pt" > "$RUN_LOG" 2>&1 < /dev/null &
echo "[chain] 正式训练已启动 pid $! $(date '+%F %T')，日志 $RUN_LOG"
