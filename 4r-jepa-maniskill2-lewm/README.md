# 4R-JEPA — based on ManiSkill2-Learn and LeWM

基于 [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn) 与 [LeWorldModel](https://github.com/lucas-maes/le-wm) 的研究实现，包含点云 PPO、连续帧世界模型、冻结世界模型表征后的 PPO 三个阶段。Python 包名保留为 `ms2_lewm`。

模型与接口适配位于 `src/ms2_lewm/`。上游源码由 `scripts/bootstrap_sources.py` 按 `repro/sources.lock.json` 获取到本地 `vendor/`，不提交上游仓库、虚拟环境、数据集或 checkpoint。实现保持上游文件原样。说明文档：[采样](SAMPLING.md)、[架构](ARCHITECTURE_REVIEW.md)、[PPO](README_RL.md)、[冻结表征 PPO](STAGE3.md)。

## 模型

| 部件 | 实际实现 |
| --- | --- |
| 逐帧 encoder | PPO/WM 共用 `configs/ppo_pointnet_identity.py`，继承官方配置；PointNet 仅最后激活改为 Identity，xyz+RGB → 512 维全局向量 |
| latent1 | `[B,T+1,512]`；同一个逐帧 encoder，包括目标帧，不冻结、不截断目标梯度 |
| TemporalEncoder | 独立实现，`K=1,T=3`；原生 PyTorch 时间自注意力＋位置编码，输出 latent2 `[B,512]`，不使用 LeWM 预测器 |
| 时序参数 | 独立 `configs/temporal_encoder.yaml`：默认2层、8头、token宽度D=512、FFN=4D、dropout0、GELU、pre-norm；无512→192→512压缩 |
| predictor | 官方 PPO critic `LinearMLP`：`[512+A,256,256,512]`，ReLU，norm=None，无输出激活；保留官方输出初始化 |
| 时间与动作 | 按最新纠正统一 stride=1；连续三帧与当前7维动作预测紧接的下一帧，使用 `action[:,T-1]` |
| loss | 下一帧 latent1 的 MSE + 0.09×官方 SIGReg；输入 `[T+1,B,D]`，各帧沿 batch 计算再平均 |

原 PPO critic 是 V 头，其输入为 512 维视觉向量加原包装器的30维 state（本体、目标等信息），即 `[542,256,256,1]`。WM 预测头改为 latent2 拼单步7维动作，因此为 `[519,256,256,512]`；隐藏层、激活和初始化均未改。

按最新要求，PPO 和世界模型共用的逐帧 PointNet 现在是 `Conv+LN+ReLU → Conv+LN+ReLU → Conv+LN+Identity → max pool`。Identity 就是不加末层激活，不再施加 ReLU 非负截断或 tanh 限幅。修改由外部注册类 `PointNetLastIdentity` 完成，继承官方 forward；官方文件未编辑。前两层激活、归一化、池化和参数形状保持一致，SIGReg 和其余模型结构未改。

PPO 的更新、损失、actor/critic 头保持原实现；后续 `run_rl.sh` 默认使用 Identity 配置，输出到 `artifacts/rl_identity_smoke` / `artifacts/rl_identity_train`。历史 ReLU/tanh 配置和记录均保留，旧 tanh 配置文件仍实例化 tanh。激活函数不在权重张量中，重载需使用保存时的配置；新 WM 导出明确记录 Identity 配置路径。兼容性检查不代表强化学习性能不变，效果仍需实际训练评测。

Encoder 的有效输入仅为 XYZ+RGB 点云。时间编码器仅输入连续三帧 latent1。原 PPO 在 encoder 后将 latent1 与30维 state 拼接给 actor/critic；第三阶段保留这条信息通道，用 latent2 替换 latent1，同样拼当前 state 后输入542维。WM 的动作只进入预测头。测试验证固定点云、改变状态/动作/目标等辅助字段不会改变 encoder 输出，但 PPO 头可以响应当前 state。

公开 le-wm 没有名为 `TemporalEncoder/LeWMLoss/WorldModel` 的类；这些名称是本地接口。LeWM 用于训练流程、SIGReg 和 JEPA 对外接口；新增时间聚合完全由 `src/ms2_lewm/temporal.py` 与独立配置定义，不导入 LeWM 的预测器 Transformer 或读取它的超参。完整 encoder 是 PointNet 加这个独立时间模块。2层/8头等是本实现明确列出的初始选择，不是官方 LeWM 参数。

```python
from ms2_lewm.model import WorldModel

wm = WorldModel(action_dim=7, history_stride=1)
# pointcloud: [B,4,1200,6]; action: [B,4,7]
out = wm({"pointcloud": pointcloud, "action": action})
out["loss"].backward()
# out: latent1[B,4,512], latent2[B,512], prediction/target[B,512]

encoded = wm.encode({"pointcloud": context})  # context [B,H,1200,6]
# encoded["emb"] is latent1, encoded["latent2"] is the complete temporal encoding
rolled = wm.rollout(info, action_sequence)
```

`rollout` 直接调用固定版本官方 `JEPA.rollout`：输入 context `[B,S,H,N,C]`，动作 `[B,S,L,A]` **包含上下文动作**，输出 `[B,S,L+1,D]`。这与较新的 stable_worldmodel 的 strictly-future 接口不同，因此依赖专门固定在与 le-wm 匹配的旧版。官方初始规划历史 H=1 也支持。Lance 将点云展平为7200维；模型内部恢复1200×6并转回float32。存储精度调整见下文。

## 采样、训练与评测

使用固定版本 `stable_worldmodel/scripts/data/collect_cube.py` 的带噪声启发式专家流程。噪声默认值与官方一致：`action_noise=0.1,p_random_action=0,noise_smoothing=0.5,min_norm=0.4,seed=None`。每个 episode 的噪声强度与逐步噪声采样复用官方实现，不使用 PPO smoke checkpoint 充当专家。

采集入口先跑预检，并写 `*.preflight.json`。30 个 episode 的成功率预检仍用1个环境串行执行；成功数为零会抛错，**在创建全量数据集之前停止**。正式采集默认 `--env-backend process`：官方配置的10个环境分别在10个 `spawn` 工作进程中执行仿真和渲染，主进程保留官方专家策略调用顺序并统一写盘。总量仍为10000 episodes、每条200步。`--env-backend serial` 可显式选择旧串行后端；小规模 `--smoke` 验证才允许缩减 episode 数和环境数。

已有串行采样测量为34.5步/秒，全量约16.1小时。10进程完全线性扩展的理想值为1.61小时；考虑共享GPU渲染、IPC和writer，当前排程估计 **2–4小时**，尚未实测新并行实现，不能保证10倍提速。此估计从成功率和容量门槛通过后开始计时。

磁盘修复仅位于本项目writer适配器：优先直接写 `.lance`，删除与点云完全重复的 `rl_xyz/rl_rgb`、与 `observation` 完全重复的 `rl_state/proprio`；只把点云落盘为float16，在线观测和模型计算仍float32。JPEG保持官方质量95，动作、状态、仿真快照保留。训练、评测直接使用官方Lance reader，不再额外转换全量HDF5；选择 `.h5/.hdf5` 时直接流式写压缩HDF5，不创建中间Lance。

容量检查先允许少量探测数据；首个episode提交后按实际文件增长外推剩余量，并增加20%余量与默认10GiB保留空间，后续每10个episode重算，已经写入的部分不重复计入剩余空间。`--reserve-gib` 可显式设置保留空间，不会自动缩减采样配方。2026-09-07真实10回合存储短测，最终分批writer约47.38MiB，外推10000回合仍约46.27GiB；不能据“fp16”宣称当前磁盘够用。该短测0/10原生成功，仅是性能测量，不是可用于正式训练的专家数据。

训练从 `vendor/le-wm/config/train/` 读取参数：seed 3072、train split 0.9、batch 128、workers 6、bf16、梯度裁剪 1.0、AdamW lr=5e-5、weight_decay=1e-3。正式训练在本地入口覆盖 `max_epochs=10`，对应 [LeWM 论文附录 E](https://arxiv.org/pdf/2603.19312#page=18) 的训练轮数；vendor 官方配置仍保留原值 100，不修改上游文件。按用户最新纠正，本地覆盖数据 frameskip=1，不能再把官方数据 frameskip=5 当作新增 encoder 的时间参数。现在采集、WM 输入、下一帧预测、PPO 决策均为1个控制步（20Hz，0.05秒），输入 `[t-2,t-1,t]`，动作 `a_t`，目标 `t+1`；旧 stride=5 checkpoint 不进入新冻结 PPO。split、shuffle/drop_last、列归一化、Lightning/spt 管理和模型导出沿用官方。调度器参数推导及配置沿用匹配的 stable_worldmodel；固定 SPT 版本实际按优化更新推进，未额外改写。

WM 的 MPC 评测使用官方 CEM/World 实现和300 samples/30 iterations/topk30/seed42，50次评测、goal offset25、budget50。改用单步模型后 action_block=1、horizon/receding_horizon=25，保留原本5×5=25个控制步的规划和执行跨度；外部适配器提供实际连续历史和已执行动作前缀。这是模型时间单位转换，不应与 PPO 每步决策混淆。`--smoke` 是资源受限集成检查。

## 第三阶段：冻结世界模型表征训练 PPO

`scripts/run_frozen_ppo.py` 从训练好的 stride=1 世界模型读取并冻结 PointNet+TemporalEncoder（截至 latent2），随后原样拼接当前 state[30]。actor 为 `542→256→256→7`，critic 为 `542→256→256→1`，与原 PPO 的头部配置相同。原 Gaussian 策略头和 PPO 算法保留；没有 WM 预测器或 LeWM 损失参与 PPO。完整说明和启动命令见 `STAGE3.md`。

encoder 强制 eval、requires_grad=False，forward 不建梯度图；actor/critic 共享同一冻结实例，官方优化器排除冻结参数。每条 PPO 观测自带连续三帧点云和当前 state；只堆叠点云，state 不堆叠、不重新计算，reset 清空历史。模型导出完整记录独立时序配置；旧的 LeWM 预测器派生时间模块 checkpoint 不会静默加载。

必要的环境接口差异公开记录：官方 cube 指 OGBench，未提供 ManiSkill2 数据。这里把官方专家接到 ManiSkill2 状态和控制接口，保留 RGB/状态列并额外采集原版 RL 点云；模型消费点云时不应用 RGB ImageNet 归一化。环境状态回放和目标设置使用 ManiSkill2 对应方法。存储适配仅做去重、点云精度与文件布局调整，不改变专家、噪声、采样步数、训练窗口或损失。

## 复现入口

先见 `README_RL.md` 运行共用 Identity encoder 的 PPO。RL 与 WM 使用独立 Python 环境：WM 只读复用本项目 RL 的依赖，并将较新的训练依赖安装在自己的环境中。

```bash
cd JEPA4R/4r-jepa-maniskill2-lewm
python3 scripts/bootstrap_sources.py
bash scripts/setup_rl.sh
bash scripts/setup_rl_gpu.sh
bash scripts/run_rl.sh train

# CPU小规模验证，需显式选择：
bash scripts/run_rl.sh smoke
bash scripts/run_rl.sh eval-smoke

bash scripts/setup_worldmodel.sh
source scripts/wm_env.sh
.venv-worldmodel/bin/python scripts/collect_wm.py --output artifacts/expert_smoke.lance --smoke
.venv-worldmodel/bin/python scripts/train_wm.py --dataset artifacts/expert_smoke.lance --run-name wm_smoke --smoke
.venv-worldmodel/bin/python scripts/evaluate_wm.py --help
```

运行只使用合成测试输入的 CPU 检查，无需启动仿真或 GPU：

```bash
source scripts/wm_env.sh
CUDA_VISIBLE_DEVICES='' .venv-worldmodel/bin/python -m pytest -q tests
.venv-rl/bin/python scripts/check_identity_ppo_replay.py --synthetic
.venv-worldmodel/bin/python scripts/check_wm_driver.py
.venv-worldmodel/bin/python scripts/check_frozen_ppo.py
```

需要检查外部已记录的点云时，PPO 检查脚本使用显式 `--fixture PATH`；不会隐式读取本机旧实验目录。

全量入口去掉 `--smoke`；采集成功率闸门始终有效。完整训练保留官方 GPU/bf16 配方，先运行 `bash scripts/setup_worldmodel.sh gpu` 安装独立的 `.venv-worldmodel-gpu`，再用该环境启动正式训练。当前 `.venv-worldmodel` 是 CPU 集成验证环境。GPU 安装/完整训练未自动执行；显存不足时入口会拒绝启动，而不会终止其他进程。`train_wm.py --config-only` 可先打印生产配置；`--resume` 接受包含优化器/调度器状态的 Lightning checkpoint。官方模型导出为 `artifacts/worldmodel/checkpoints/<run>/weights_epoch_*.pt` 加 `config.json`，由官方 `load_pretrained` 重建。

原 PPO 的正式入口现在默认 CUDA：`run_rl.sh` 或 `run_rl.sh train` 选择 `.venv-rl-gpu`；`smoke/eval-smoke` 使用 CPU，`train-cpu` 是显式的 CPU 训练选择。原 PPO 正式训练的本机配置使用 **12个并行环境**：16个可用CPU核留4个给主进程与IPC，模型/优化参数及500万步预算保留。从已有1环境和5环境GPU测量拟合并行时间，12环境粗估4.14小时，排程 **4–6小时**；未逐个尝试并行数，也不宣称12是实测最优值。冻结表征 PPO 同样默认 CUDA，`--smoke` 才默认 CPU。缺少所选环境不会自动降级，所有实际仿真入口启动前检查显存。`run_rl.sh --print-command` 只打印计划命令。历史测量和新配置外推分别记录在 `validation/runtime-estimates.json`。 原 PPO 已于2026-09-07 02:19:18 UTC正式启动，输出目录 `artifacts/rl_identity_train_p12_20260907`；记录时已完成220,000步，首轮之后每20,000步耗时中位数约64.92秒，对应全程约4.51小时；继续执行500万步预算。首轮包含一次性critic warmup，不作为稳定吞吐。世界模型、冻结表征PPO和全量专家采集仍未启动。

CPU 专家预检使用 `physics_preflight.py` 中的无渲染子类：保留官方碰撞、物理、初始化、控制器和成功判定，省略摄像机和视觉资源。此路径没有图像或点云，不能采集训练数据；其作用仅是在 GPU 忙碌时扩大专家成功率测试。真实点云采集仍必须通过正式渲染环境及资源检查。

依赖边界：RL 环境使用 NumPy 1.23.5，依赖检查通过。官方 Lance 依赖要求 NumPy≥1.24，WM 隔离环境使用 1.24.4；与旧 ManiSkill2 的包声明 `numpy<1.24` 存在一项元数据冲突，已单独记录在 `artifacts/wm_pip_check.txt`，并实际验证 PointNet 与真实模拟可执行。没有为此修改官方包声明或 RL 环境。

## 证据与边界

此前已实际验证原版 ReLU PPO：32 个真实环境 transition、PPO 更新、checkpoint、独立加载评测；encoder/actor/critic 都有有限的参数更新，真实 encoder 输出 `[1,512]`。两次 16 步 smoke 评测成功率 0/2，未声称 PPO 收敛。当前 Identity 版本通过 CPU 兼容检查：使用已有真实点云/动作以及明确标记的合成状态、奖励和终止标志，执行官方 PPO 更新并恢复模型/优化器；此检查不等于新的真实环境回合或性能评估。历史 tanh 检查单独保留。

模型测试覆盖目标帧梯度、严格动作索引、未来帧不泄漏、逐帧 SIGReg、时序配置、官方 PointNet/critic 身份、多步 rollout、规划代价、形状恢复、冻结权重、checkpoint 迁移和来源校验。发布目录验证为 **116 项 CPU 测试通过**；记录见 [validation/summary.json](validation/summary.json) 和 [JUnit](validation/cpu-tests.xml)。重新运行的完整报告保存在忽略提交的 `artifacts/tests/`。

最新专家预检：修复 `set_new_target` 错误调用任务reset hook后，目标Z基准固定为桌面方块中心 `cube_half_size[2]=.02`，范围 `.02–.52m`；XY范围、RNG和最小间距拒绝采样保留。同一份记录噪声重测30回合，原生成功仍 **0/30**，仅位置到达 **28/30**，IK失败由199降至186，因此高度错误不能解释全部IK失败。旧版本的27/30是历史位置指标。证据见 [validation/retarget-fix.json](validation/retarget-fix.json)。未增加保持阶段，未修改oracle、官方噪声或成功判定。真实10回合点云短测仅用于测时/测空间；**正式专家数据采集、真实数据WM训练和规划评测仍未启动**。

训练入口另用 `artifacts/tests/` 内明确标记的合成测试夹具验证：执行真实 Lightning/spt 优化、保存官方模型、恢复优化器/调度器 checkpoint 并继续训练、重载模型。此测试仅验证代码链路，不是专家数据，不提供任务表征质量或成功率证据。生产训练在GPU分配前拒绝HDF5属性或邻接 `<dataset>.json` 中明确的 `SYNTHETIC_UNIT_TEST_ONLY` / `BENCHMARK_ONLY_NOT_TRAINING_DATA` 标记；普通外部数据不要求额外manifest。完整10-epoch训练实验不等同于集成检查。

固定源码提交与哈希见 [repro/sources.lock.json](repro/sources.lock.json)，WM wheel 版本、下载地址和校验见 [repro/wm_dependencies.json](repro/wm_dependencies.json)，依赖版本另见 `requirements-rl.lock` 和 `requirements-worldmodel-overlay.lock`。`python3 scripts/bootstrap_sources.py --check` 验证上游源码，`python3 scripts/download_wm_dependencies.py --check-lock` 验证依赖锁格式。

历史本地运行日志、真实轨迹及大型 checkpoint 未随仓库提交；上述相应运行路径仅描述初始验证，不是 fresh checkout 自带的数据。模型使用项目相对配置路径；冻结 encoder 包包含配置快照，并校验实现及来源哈希。
