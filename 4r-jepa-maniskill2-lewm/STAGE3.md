# 用冻结世界模型表征训练 PPO

时间单位统一为一个环境控制步：20Hz，即0.05秒。WM 预训练、冻结 encoder 的历史帧与原 PPO 决策间隔均为1；T=3表示输入 `[o_(t-2),o_(t-1),o_t]`。WM 用7维 `a_t` 预测 `o_(t+1)` 的 latent1。之前 stride5/35维动作版本的 checkpoint 不会被默认为新版本。

第三阶段模型为：

```
连续3帧点云
    ↓ 冻结的 PointNet（最后Identity）
latent1 [B,3,512]
    ↓ 冻结的 TemporalEncoder
latent2 [B,512]
    ↓ 拼当前原始state [B,30]（含本体和目标等信息）
策略/价值头输入 [B,542]
    ├─ actor：542→256→256→7 + 原GaussianHead
    └─ critic：542→256→256→1
```

只加载 WM 的 encoder 部分，不加载预测器或损失模块。冻结参数不进入 actor/critic 优化器；encoder 始终 eval，forward 使用 no_grad。actor、critic 共用同一个冻结 encoder，随后通过原 Visuomotor.forward 拼接当前 state；头部层数、输入宽度、激活、原 PPO 的损失、GAE、裁剪、优化器与更新代码都保留。不存在 WM 辅助损失或联合微调。

历史由每个环境自己的 `PointCloudHistory` 管理，reset 时清空，开头缺帧重复初始帧；每次 step 执行一个真实动作，得到当前 episode 的 next_obs。仅 xyz/rgb 堆叠成三帧；state[30] 与其他数值字段从当前观测原样复制，保留坐标系、顺序和 dtype。三帧与当前 state 一起存入 PPO 缓冲区，所以 minibatch 打乱后仍可正确重算特征。

TemporalEncoder 是独立的 D=512 时间模块，配置来自 `configs/temporal_encoder.yaml`，不是 LeWM 预测器。训练导出会记录完整配置，冻结阶段按记录重建；缺失配置或旧架构会拒绝加载。

## 训练完成后的入口

先用新 stride1 配置完成真实 WM 训练。正式训练在本地入口覆盖 `max_epochs=10`，按 [LeWM 论文附录 E](https://arxiv.org/pdf/2603.19312#page=18)，vendor 官方配置仍保留 100。官方导出按已完成轮数从 1 开始编号，训练完成后取得 `weights_epoch_10.pt` 及同目录的 `config.json`。支持读取完整 Lightning checkpoint，只要伴随配置明确记录当前时间尺度。

```bash
cd /path/to/JEPA4R/4r-jepa-maniskill2-lewm
source scripts/wm_env.sh

# 只提取/检查冻结 encoder、配置及命令；不启动仿真或GPU
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_10.pt \
  --prepare-only

# 正式PPO；默认CUDA、单环境，保留500万步预算
# 先完成 bash scripts/setup_worldmodel.sh gpu
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_10.pt \
  --work-dir artifacts/ppo_frozen_wm/run_01

# 小规模真实环境检查（只有资源足够时才会启动）
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/trained_wm/weights_epoch_10.pt \
  --work-dir artifacts/ppo_frozen_wm/smoke_01 --smoke

# 重载PPO并评测；使用同一个源WM或准备好的encoder.pt
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint /absolute/path/to/matching/encoder.pt \
  --resume-from /absolute/path/to/ppo/models/model_final.ckpt \
  --work-dir artifacts/ppo_frozen_wm/eval_01 --evaluation
```

正式训练默认选择 `.venv-worldmodel-gpu` 和 CUDA；`--smoke` 默认 CPU，显式 `--device cpu` 仍可用于调试。GPU 环境通过 `bash scripts/setup_worldmodel.sh gpu` 单独安装，不会覆盖 CPU 环境，也不会悄悄退回 CPU。实际启动前检查显存，资源不足直接拒绝，不终止其他任务。已有非空输出目录不被覆盖。

准备文件保存在 `artifacts/frozen_ppo_prepared/`：encoder-only 权重、完整PPO配置快照和来源/hash记录。PPO checkpoint 包含冻结 encoder 与两个头；原始 WM 文件删除后，准备好的 encoder 包仍可重建。配置或 encoder 实现发生变化时拒绝悄悄加载；恢复 PPO 时检查源 WM 指纹匹配。

## 验证边界

第三阶段已经使用明确标记的 CPU 合成测试 checkpoint/数据验证冻结、PPO更新和恢复：encoder 参数与 buffer 字节不变，actor/critic/Gaussian log_std 更新，dropout不重启，优化器不含encoder参数。合成测试不是训练好的专家表征，不证明任务性能。

本机全量专家采集已启动，默认30步保持适配、官方噪声、10进程；成功率不再作为开采硬门槛。真实WM数据训练与第三阶段长程PPO尚未完成；CPU测试checkpoint不能代替训练好的世界模型。当前采集的旧coverage存在诊断列缺失，质量指标边界见 [SAMPLING.md](SAMPLING.md#覆盖统计与历史记录边界)。

原 PickCube 的目标标记不在默认点云中，因此保留原 PPO 的 state 通道很重要。当前实现已恢复 latent2 后的 state 拼接；encoder 本身仍只依赖点云。没有修改任务目标分布或原生 success 判定。
