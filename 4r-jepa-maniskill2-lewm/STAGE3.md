# 用冻结世界模型表征训练 PPO

时间单位统一为一个环境控制步：20Hz，即0.05秒。WM 预训练、冻结 encoder 的历史帧与原 PPO 决策间隔均为1；T=3表示输入 `[o_(t-2),o_(t-1),o_t]`。WM 用7维 `a_t` 预测 `o_(t+1)` 的 latent1。之前 stride5/35维动作版本的 checkpoint 不会被默认为新版本。

第三阶段模型为：

```
连续3帧点云
    ↓ 冻结的 PointNet（最后Identity）
latent1 [B,3,512]
    ↓ 冻结的 TemporalEncoder
latent2 [B,512]
    ├─ 新 actor：512→256→256→7 + 原GaussianHead
    └─ 新 critic：512→256→256→1
```

只加载 WM 的 encoder 部分，不加载预测器或损失模块。冻结参数不进入 actor/critic 优化器；encoder 始终 eval，关闭 dropout，forward 使用 no_grad。actor、critic 共用同一个冻结 encoder，训练使用原 PPO 的损失、GAE、裁剪、优化器与更新代码。不存在 WM 辅助损失、联合微调或本体状态旁路。

历史由每个环境自己的 `PointCloudHistory` 管理，reset 时清空，开头缺帧重复初始帧；每次 step 执行一个真实动作，得到当前 episode 的 next_obs，不提前切到下个 episode。三帧直接存入 PPO 缓冲区，所以 minibatch 打乱后仍可正确重算 latent2。

## 训练完成后的入口

先用新 stride1 配置完成真实 WM 训练，取得 `weights_epoch_*.pt` 及同目录的 `config.json`。支持读取完整 Lightning checkpoint，只要伴随配置明确记录当前时间尺度。

```bash
cd /path/to/JEPA4R/4r-jepa-maniskill2-lewm
source scripts/wm_env.sh

# 只提取/检查冻结 encoder、配置及命令；不启动仿真或GPU
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_100.pt \
  --prepare-only

# 正式PPO；默认网络CPU、单环境，保留500万步预算
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_100.pt \
  --work-dir artifacts/ppo_frozen_wm/run_01

# 小规模真实环境检查（只有资源足够时才会启动）
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_100.pt \
  --work-dir artifacts/ppo_frozen_wm/smoke_01 --smoke

# 重载PPO并评测；使用同一个源WM或准备好的encoder.pt
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/matching/encoder.pt \
  --resume-from /absolute/path/to/ppo/models/model_final.ckpt \
  --work-dir artifacts/ppo_frozen_wm/eval_01 --evaluation
```

网络用GPU时需用已安装好的CUDA环境执行并加 `--device cuda`；不会自动修改现有环境。实际启动前检查显存，资源不足直接拒绝，不终止其他任务。已有非空输出目录不被覆盖。

准备文件保存在 `artifacts/frozen_ppo_prepared/`：encoder-only 权重、完整PPO配置快照和来源/hash记录。PPO checkpoint 包含冻结 encoder 与两个头；原始 WM 文件删除后，准备好的 encoder 包仍可重建。配置或 encoder 实现发生变化时拒绝悄悄加载；恢复 PPO 时检查源 WM 指纹匹配。

## 验证边界

第三阶段已经使用明确标记的 CPU 合成测试 checkpoint/数据验证冻结、PPO更新和恢复：encoder 参数与 buffer 字节不变，actor/critic/Gaussian log_std 更新，dropout不重启，优化器不含encoder参数。合成测试不是训练好的专家表征，不证明任务性能。

当前真实专家采样门槛仍为0/30，真实 WM 数据训练尚未启动，因此没有自动开始第三阶段长程训练。

观测限制按“只用latent2”保留：原 PickCube 的目标标记不在默认点云中，当前也未拼接原30维状态/目标信息；这会限制随机目标任务的可观测性。此处未擅自加入状态或修改任务目标/成功判定。
