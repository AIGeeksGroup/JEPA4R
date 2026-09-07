# 4R-JEPA — based on ManiSkill2-Learn and LeWM

基于 [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn) 与 [LeWorldModel](https://github.com/lucas-maes/le-wm) 的研究实现，包含点云 PPO、连续帧世界模型、冻结世界模型表征后的 PPO 三个阶段。Python 包名保留为 `ms2_lewm`。

模型与接口适配位于 `src/ms2_lewm/`。上游源码由 `scripts/bootstrap_sources.py` 按 `repro/sources.lock.json` 获取到本地 `vendor/`，不提交上游仓库、虚拟环境、数据集或 checkpoint。实现保持上游文件原样。说明文档：[采样](SAMPLING.md)、[架构](ARCHITECTURE_REVIEW.md)、[PPO](README_RL.md)、[冻结表征 PPO](STAGE3.md)。

**GitHub 分支：`4r-jepa-based-on-maniskill2-lewm`。核心实现已提交在本子目录，不在 `main`，也不藏在未提交的 `vendor/` 中。**

| 要查看的实现 | 本仓库源码 |
| --- | --- |
| WorldModel、LeWMLoss、逐帧 encoder 包装和三个对外接口 | [src/ms2_lewm/model.py](src/ms2_lewm/model.py) |
| 独立时间聚合 | [src/ms2_lewm/temporal.py](src/ms2_lewm/temporal.py)、[configs/temporal_encoder.yaml](configs/temporal_encoder.yaml) |
| 复用官方 PointNet / critic、最后 Identity 激活 | [src/ms2_lewm/rl_components.py](src/ms2_lewm/rl_components.py)、[src/ms2_lewm/pointnet_identity.py](src/ms2_lewm/pointnet_identity.py) |
| 冻结表征接原 PPO、每个环境的时间历史 | [src/ms2_lewm/frozen_ppo.py](src/ms2_lewm/frozen_ppo.py)、[src/ms2_lewm/history.py](src/ms2_lewm/history.py) |
| 原 PPO、WM、冻结 PPO 训练入口 | [scripts/run_ppo.py](scripts/run_ppo.py)、[scripts/train_wm.py](scripts/train_wm.py)、[scripts/run_frozen_ppo.py](scripts/run_frozen_ppo.py) |
| 并行采集、保持阶段、覆盖统计和存储 | [scripts/collect_wm.py](scripts/collect_wm.py)、[process_env_pool.py](src/ms2_lewm/process_env_pool.py)、[pickcube_oracle.py](src/ms2_lewm/pickcube_oracle.py)、[coverage.py](src/ms2_lewm/coverage.py)、[storage.py](src/ms2_lewm/storage.py) |
| WM 规划评测 | [scripts/evaluate_wm.py](scripts/evaluate_wm.py) |

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

以固定版本 `stable_worldmodel/scripts/data/collect_cube.py` 的带噪声启发式专家流程为基础。默认使用本地 `PickCubeHoldOracle`：保留原 `CubeMarkovOracle` 的接近、抓取、搬运、释放及退避，在夹持方块到达目标时增加30步保持阶段。外层 `ExpertPolicy.get_action` 与噪声默认值保持官方原样：`action_noise=0.1,p_random_action=0,noise_smoothing=0.5,min_norm=0.4,seed=None`，保持阶段同样加噪。`--oracle-mode legacy` 可显式使用无保持阶段的原 oracle。这项环境适配应与“官方噪声未改”区分，不能称为完全相同的原论文采样策略。

采集入口默认直接开始采集，**无前置预检、无原生 success 为零即阻断的门槛**。`--preflight-episodes N` 是可选诊断，`--preflight-only` 只做诊断，报告不筛选或阻止后续轨迹。正式采集默认 `--env-backend process`：官方配置的10个环境分别在10个 `spawn` 工作进程中执行仿真和渲染，主进程保留官方专家策略调用顺序并统一写盘。总量仍为10000 episodes、每条200步。所有成功、失败和中间操作均保留，不设成功率配额或人为失败比例。覆盖统计写在 `<dataset>.coverage.json`；它是诊断，不能保证既定质量比例。

固定10进程的历史独占短测为30回合、6000控制步、43.13秒（139.1步/秒），外推全量约4小时。它发生在当前保持阶段与 zstd12 存储启用之前，且未与16环境 PPO 同跑，不能作为当前采集耗时承诺。去掉独立RGB相机无用的Position读回后，6000帧的点云、RGB、动作和状态完全一致。连续CPU/GPU记录见 [validation/resource-profiles.json](validation/resource-profiles.json) 和 [资源曲线](validation/resource-profiles.png)。本机全量采集已启动，运行记录见下文。

存储适配优先直接写 `.lance`，删除与点云完全重复的 `rl_xyz/rl_rgb`、与 `observation` 完全重复的 `rl_state/proprio`；只把点云落盘为float16，在线观测和模型计算仍float32。Lance物理缓冲使用 **zstd level12 无损压缩**，JPEG保持官方质量95，动作、状态、仿真快照保留。这里“无损”指 zstd 对既有缓冲的压缩，float32→float16 点云量化仍有舍入误差。训练、评测直接使用官方Lance reader，不再额外转换全量HDF5；选择 `.h5/.hdf5` 时直接流式写压缩HDF5，不创建中间Lance。

容量检查允许少量探测数据，之后按实际写盘增长外推剩余全量空间，已经写入的部分不重复计数。CLI默认保留10GiB、空间安全系数1.2；本机本次全量运行显式使用 `--reserve-gib 2 --storage-safety-factor 1.05`，这是本机容量预算选择，其他机器无需照搬。检查继续运行，空间不足时停止，不自动删数据或减少配方。此前约46.27GiB的容量估计来自未启用 zstd12 的旧writer，不代表当前格式大小。

训练从 `vendor/le-wm/config/train/` 读取参数：seed 3072、train split 0.9、batch 128、workers 6、bf16、梯度裁剪 1.0、AdamW lr=5e-5、weight_decay=1e-3。正式训练在本地入口覆盖 `max_epochs=10`，对应 [LeWM 论文附录 E](https://arxiv.org/pdf/2603.19312#page=18) 的训练轮数；vendor 官方配置仍保留原值 100，不修改上游文件。按用户最新纠正，本地覆盖数据 frameskip=1，不能再把官方数据 frameskip=5 当作新增 encoder 的时间参数。现在采集、WM 输入、下一帧预测、PPO 决策均为1个控制步（20Hz，0.05秒），输入 `[t-2,t-1,t]`，动作 `a_t`，目标 `t+1`；旧 stride=5 checkpoint 不进入新冻结 PPO。split、shuffle/drop_last、列归一化、Lightning/spt 管理和模型导出沿用官方。调度器参数推导及配置沿用匹配的 stable_worldmodel；固定 SPT 版本实际按优化更新推进，未额外改写。

WM 的 MPC 评测使用官方 CEM/World 实现和300 samples/30 iterations/topk30/seed42，50次评测、goal offset25、budget50。改用单步模型后 action_block=1、horizon/receding_horizon=25，保留原本5×5=25个控制步的规划和执行跨度；外部适配器提供实际连续历史和已执行动作前缀。这是模型时间单位转换，不应与 PPO 每步决策混淆。`--smoke` 是资源受限集成检查。

## 第三阶段：冻结世界模型表征训练 PPO

`scripts/run_frozen_ppo.py` 从训练好的 stride=1 世界模型读取并冻结 PointNet+TemporalEncoder（截至 latent2），随后原样拼接当前 state[30]。actor 为 `542→256→256→7`，critic 为 `542→256→256→1`，与原 PPO 的头部配置相同。原 Gaussian 策略头和 PPO 算法保留；没有 WM 预测器或 LeWM 损失参与 PPO。完整说明和启动命令见 `STAGE3.md`。

encoder 强制 eval、requires_grad=False，forward 不建梯度图；actor/critic 共享同一冻结实例，官方优化器排除冻结参数。每条 PPO 观测自带连续三帧点云和当前 state；只堆叠点云，state 不堆叠、不重新计算，reset 清空历史。模型导出完整记录独立时序配置；旧的 LeWM 预测器派生时间模块 checkpoint 不会静默加载。

必要的环境接口差异公开记录：官方 cube 指 OGBench，未提供 ManiSkill2 数据。这里把官方专家接到 ManiSkill2 状态和控制接口，并增加上述保持阶段；保留 RGB/状态列并额外采集原版 RL 点云，模型消费点云时不应用 RGB ImageNet 归一化。环境状态回放和目标设置使用 ManiSkill2 对应方法。存储适配只做去重、点云精度与文件布局调整，不筛轨迹，不改噪声、采样步数、训练窗口或损失。

## 复现入口

需要 Linux x86_64、Python3.10；真实点云仿真还需要可用的 NVIDIA/Vulkan 渲染环境。RL 与 WM 使用独立 Python 环境：WM 只读复用本项目 RL 的依赖，并将较新的训练依赖安装在自己的环境中。以下先安装依赖；不会因为 clone 或 setup 自动启动训练。

```bash
git clone --branch 4r-jepa-based-on-maniskill2-lewm --single-branch \
  git@github.com:AIGeeksGroup/JEPA4R.git
cd JEPA4R/4r-jepa-maniskill2-lewm
python3 scripts/bootstrap_sources.py
python3 scripts/bootstrap_sources.py --check
bash scripts/setup_rl.sh
bash scripts/setup_rl_gpu.sh
bash scripts/setup_worldmodel.sh cpu
bash scripts/setup_worldmodel.sh gpu
```

`bootstrap_sources.py` 获取 [sources.lock.json](repro/sources.lock.json) 中固定的 ManiSkill2-Learn、pytorch3d、le-wm、stable-worldmodel 四个提交；`--check` 校验已获取的源码。无需从本机历史目录复制核心文件、环境或数据。Python3.10 不在默认路径时可设置 `RL_PYTHON=/path/to/python3.10`。

显式启动各阶段的命令如下；不要求把它们同时运行：

```bash
# 第一阶段：正式点云PPO，CUDA，16环境，500万步
bash scripts/run_rl.sh train

# 全量带噪声启发式采集：10进程，10000回合，每条200步
source scripts/wm_env.sh
.venv-worldmodel/bin/python scripts/collect_wm.py \
  --output artifacts/expert_full.lance

# 与本机当前全量运行一致的空间预算；只在接受该预算时显式选择
# .venv-worldmodel/bin/python scripts/collect_wm.py \
#   --output artifacts/expert_full.lance --reserve-gib 2 --storage-safety-factor 1.05

# 第二阶段：采集完成后，真实数据WM训练，CUDA/bf16，10 epochs
.venv-worldmodel-gpu/bin/python scripts/train_wm.py \
  --dataset artifacts/expert_full.lance --run-name pointcloud_lewm_identity

# 第三阶段：使用训练完成的WM，先检查冻结配置和来源，不启动PPO
.venv-worldmodel/bin/python scripts/run_frozen_ppo.py \
  --wm-checkpoint artifacts/worldmodel/checkpoints/pointcloud_lewm_identity/weights_epoch_10.pt \
  --prepare-only
```

第三阶段正式训练与评测命令见 [STAGE3.md](STAGE3.md)。采集命令虽然使用 CPU PyTorch 环境，但各环境仍使用 GPU 渲染；oracle 本身没有需要 CUDA 推理的神经网络。

```bash
# CPU小规模验证；真实模拟仍需渲染资源
bash scripts/run_rl.sh smoke
bash scripts/run_rl.sh eval-smoke

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

全量入口去掉 `--smoke`；成功率只作诊断，资源检查仍生效。完整训练使用 `.venv-worldmodel-gpu` 保留 GPU/bf16 配方，`.venv-worldmodel` 用于CPU检查与启发式采集。显存不足时入口拒绝启动，不终止其他进程。`train_wm.py --config-only` 可先打印生产配置；`--resume` 接受包含优化器/调度器状态的 Lightning checkpoint。官方模型导出为 `artifacts/worldmodel/checkpoints/<run>/weights_epoch_*.pt` 加 `config.json`，由官方 `load_pretrained` 重建。

原 PPO 的正式入口现在默认 CUDA：`run_rl.sh` 或 `run_rl.sh train` 选择 `.venv-rl-gpu`；`smoke/eval-smoke` 使用 CPU，`train-cpu` 是显式的 CPU 训练选择。按用户最新要求，原 PPO 正式训练的本机默认已改为 **16个并行环境**，模型/优化参数及500万步预算保留；16个环境worker与主进程、IPC共享本机16核CPU额度，不宣称此数值是实测最优。冻结表征 PPO 同样默认 CUDA，`--smoke` 才默认 CPU。缺少所选环境不会自动降级，所有实际仿真入口启动前检查显存。`run_rl.sh --print-command` 只打印计划命令。历史测量和配置外推分别记录在 `validation/runtime-estimates.json`。

本机2026-09-07的启动记录：16环境 PPO 在 `artifacts/rl_identity_train_p16_20260907`，launcher PID152283；全量采集在 `artifacts/wm_collection_full_20260907T044524Z`，launcher PID158972，10进程、10000×200步、默认保持阶段与 zstd12。两项已启动并行运行；PID和进度只是本机运行记录，不是 fresh clone 自带资源或长期状态承诺。真实数据世界模型训练、冻结表征 PPO 和完整规划评测尚未完成，不能声称已复现论文指标。旧12环境任务已停止。

本次早期采集的覆盖统计存在布尔诊断列遗漏问题，当前运行的旧 `coverage.json` 中抓取/原生成功计数不可作为成败证据，不能把记录的0解释为全部失败。点云、动作及状态数据不受此统计问题影响；详见 [SAMPLING.md](SAMPLING.md#覆盖统计与历史记录边界)。

设置 `MS2_PPO_RECORD_VIDEO` 为MP4输出路径，可录制PPO真实训练rollout中一个环境的首个episode。录制直接复制逐帧encoder实际收到的1200个XYZ+RGB点，后台CPU用固定视角生成点云视频，不调用RGB相机渲染、不改采样动作。本次PPO录制输出为 `artifacts/user_videos/ppo_training_pointcloud_episode0.mp4`。配套[启发式采样点云视频](artifacts/user_videos/expert_pointcloud_episode0.mp4)已从真实Lance episode导出，共200帧、20fps，点云按模型入口从fp16转float32；固定3D视图不添加目标球或场景mesh。两条视频都是encoder输入可视化，不另开评测episode，也不代表策略已经收敛。命令见 `README_RL.md`。

CPU 专家预检使用 `physics_preflight.py` 中的无渲染子类：保留官方碰撞、物理、初始化、控制器和成功判定，省略摄像机和视觉资源。此路径没有图像或点云，不能采集训练数据；其作用仅是在 GPU 忙碌时扩大专家成功率测试。真实点云采集仍必须通过正式渲染环境及资源检查。

依赖边界：RL 环境使用 NumPy 1.23.5，依赖检查通过。官方 Lance 依赖要求 NumPy≥1.24，WM 隔离环境使用 1.24.4；与旧 ManiSkill2 的包声明 `numpy<1.24` 存在一项元数据冲突，已单独记录在 `artifacts/wm_pip_check.txt`，并实际验证 PointNet 与真实模拟可执行。没有为此修改官方包声明或 RL 环境。

## 证据与边界

此前已实际验证原版 ReLU PPO：32 个真实环境 transition、PPO 更新、checkpoint、独立加载评测；encoder/actor/critic 都有有限的参数更新，真实 encoder 输出 `[1,512]`。两次 16 步 smoke 评测成功率 0/2，未声称 PPO 收敛。当前 Identity 版本通过 CPU 兼容检查：使用已有真实点云/动作以及明确标记的合成状态、奖励和终止标志，执行官方 PPO 更新并恢复模型/优化器；此检查不等于新的真实环境回合或性能评估。历史 tanh 检查单独保留。

模型测试覆盖目标帧梯度、严格动作索引、未来帧不泄漏、逐帧 SIGReg、时序配置、官方 PointNet/critic 身份、多步 rollout、规划代价、形状恢复、冻结权重、checkpoint 迁移和来源校验。已提交验证记录见 [validation/summary.json](validation/summary.json) 和 [JUnit](validation/cpu-tests.xml)；测试数量与覆盖范围以对应提交的报告为准，不把历史CPU集成测试当作真实实验结果。

历史 legacy oracle 的30回合诊断：修复 `set_new_target` 错误调用任务reset hook后，目标Z基准固定为桌面方块中心 `.02m`，范围 `.02–.52m`；原生成功0/30、位置到达28/30，记录见 [validation/retarget-fix.json](validation/retarget-fix.json)。这份记录先于当前 PickCubeHoldOracle，不能用于断言当前采集全失败，也不再阻止采集。当前完整策略和存储边界见 [SAMPLING.md](SAMPLING.md)。

训练入口另用 `artifacts/tests/` 内明确标记的合成测试夹具验证：执行真实 Lightning/spt 优化、保存官方模型、恢复优化器/调度器 checkpoint 并继续训练、重载模型。此测试仅验证代码链路，不是专家数据，不提供任务表征质量或成功率证据。生产训练在GPU分配前拒绝HDF5属性或邻接 `<dataset>.json` 中明确的 `SYNTHETIC_UNIT_TEST_ONLY` / `BENCHMARK_ONLY_NOT_TRAINING_DATA` 标记；普通外部数据不要求额外manifest。完整10-epoch训练实验不等同于集成检查。

固定源码提交与哈希见 [repro/sources.lock.json](repro/sources.lock.json)，WM wheel 版本、下载地址和校验见 [repro/wm_dependencies.json](repro/wm_dependencies.json)，依赖版本另见 `requirements-rl.lock` 和 `requirements-worldmodel-overlay.lock`。`python3 scripts/bootstrap_sources.py --check` 验证上游源码，`python3 scripts/download_wm_dependencies.py --check-lock` 验证依赖锁格式。

历史本地运行日志、真实轨迹及大型 checkpoint 未随仓库提交；上述相应运行路径仅描述初始验证，不是 fresh checkout 自带的数据。模型使用项目相对配置路径；冻结 encoder 包包含配置快照，并校验实现及来源哈希。
