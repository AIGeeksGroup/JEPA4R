# 4D JEPA 项目变更脉络

本文件只记录代码、配置和实验变更，采用只追加方式。每条记录包含日期、类型、变更、原因、验证和关联提交或实验目录。

## 2026-09-03 · 初始化基线

- 类型：Code / Config / Experiment
- 代码基线：实施前记录的仓库基线为 `ba66811`（该提交不是台账完成后的当前 HEAD）；代码初始提交为 `a49327c`。随后台账文件在 `cbdcf65` 提交，远端为 `git@github.com:gktdgktd/4d-jepa-architecture.git`。
- 配置基线：`configs/default.yaml` 和 `Makefile` 未因本次整理修改。
- 实验基线：尚未运行本项目实验；没有可记录的训练结果或 checkpoint（检查点）。
- 验证：`git status --short --branch` 干净；代码入口和配置文件可访问；本次未执行训练。
- 关联：`ba66811`（实施前基线）；`cbdcf65` 与本次修正提交（台账提交）。

### 2026-09-03 · ManiSkill2 三任务序列适配与 smoke

- 类型：Code / Config / Experiment
- 代码：新增 `data/maniskill.py`，读取 compact ManiSkill2 episode NPZ，并在单 episode 内构造 `[T+1,K,3]` 点云上下文与 `[T,7]` 动作窗口；新增 `action_stats()` 计算训练集动作归一化统计；`data/__init__.py` 导出适配器。
- 数据：使用服务器持久盘 `/root/autodl-tmp/workspace/datasets/maniskill2/v0.5.3/compact/` 中已有 `LiftCube-v0`、`PickCube-v0`、`StackCube-v0`，每个任务 100 个 episode（90 train / 10 val），未重复下载。
- 配置：新增 `configs/maniskill_smoke.yaml`，记录小型 WorldModel smoke 配置（`T=3`、`K=256`、`D=32`、点云输入 3 维、7 维动作）。
- 脚本：新增 `scripts/smoke_maniskill.py`，逐任务执行 DataLoader → WorldModel 前向 → loss → backward → AdamW 更新，并检查 loss/梯度有限性。
- 测试：`/root/miniconda3/bin/python -m pytest -q tests/test_maniskill_dataset.py` 输出 `4 passed`；三任务 smoke 输出 `ManiSkill2 data-to-model smoke: PASS`，结果为 Lift `loss=2.778037`、Pick `loss=2.986122`、Stack `loss=3.030535`（各 1 step，batch size 1）。
- 计划：`docs/superpowers/plans/2026-09-03-maniskill2-adapter-plan.md`。
- 关联：待提交 Git commit；完整训练和后续实验未启动。

### 2026-09-03 · 下载 ManiSkill2 官方原始轨迹

- 类型：Data / Experiment
- 数据：下载官方 `haosulab/ManiSkill2` v0 rigid-body demos 的 `trajectory.h5` 与 `trajectory.json`，任务为 `LiftCube-v0`、`PickCube-v0`、`StackCube-v0`。
- 路径：服务器持久盘 `/root/autodl-tmp/workspace/datasets/maniskill2/v0.5.3/demos/v0/rigid_body/<task>/`；未覆盖已有 `compact/` 目录。
- 内容：Lift 100 episodes；Pick 1000 episodes；Stack 1000 episodes。HDF5 每个 episode 含 `actions`、`env_states`、`success`，可用于后续 replay（轨迹回放）和状态/成功标签评估。
- 校验：六个文件均按官方 SHA-256 校验通过；HDF5/JSON 可读性检查通过。服务器目录合计约 82 MB。
- 说明：服务器当前未安装 ManiSkill2、SAPIEN、Gymnasium 的 Python 3.11 运行时，因此本次只完成原始轨迹下载与验证，未启动 replay 或仿真闭环评估。
- 关联：官方数据集 `https://huggingface.co/datasets/haosulab/ManiSkill2`；后续如需完整点云观测，应安装兼容运行时后再执行 replay。

### 2026-09-03 · Group 1 LiftCube scratch 训练流程就绪

- 类型：Code / Config / Experiment
- 目标边界：按 Group 1 设计，只使用 `LiftCube-v0` 的 90 个训练 episode 和 10 个验证 episode；不读取 `PickCube-v0`、`StackCube-v0`。
- 设计：新增 `docs/superpowers/specs/2026-09-03-group1-scratch-design.md` 和实施计划 `docs/superpowers/plans/2026-09-03-group1-scratch-train-plan.md`。
- 配置：新增 `configs/maniskill_group1_scratch.yaml`，默认 `T=3`、`K=512`、`D=64`、`C=3`、batch size 2、10000 steps、每 500 steps 验证/保存；动作均值和标准差由 LiftCube 训练 episode 动态计算。
- 入口：新增 `scripts/train_maniskill_group1.py`，包含 90/10 manifest 契约检查、训练/验证 DataLoader、随机初始化 WorldModel、preflight（单 batch 前向/反向且不更新参数）、checkpoint、验证指标和显式 `--resume` 恢复。
- 验证：Group 1 focused tests `2 passed`；全量测试 `22 passed, 1 warning`；checkpoint 序列化和验证路径各自通过。
- GPU preflight：RTX 5090（32607 MiB）上使用真实 LiftCube batch `[2,4,512,3]` / actions `[2,3,7]` 完成前向和反向；loss `3.075096`，峰值显存 allocated `250,983,424` bytes（约 239 MiB），reserved `358,612,992` bytes（约 342 MiB）。结束后 `nvidia-smi` 为显存占用 2 MiB、GPU 利用率 0%，没有训练进程运行。
- 启动命令：`cd /root/autodl-tmp/projects/4d-jepa-architecture && /root/miniconda3/bin/python scripts/train_maniskill_group1.py`；本次仅准备和 preflight，未执行该长训练命令。
- 关联：待提交 Git commit；运行目录默认 `/root/autodl-tmp/runs/4d-jepa-architecture/group1_scratch_liftcube`。

## 2026-09-03 · Group 1 LiftCube scratch 训练完成与表征诊断

- 类型：Experiment
- 运行：仅使用 `LiftCube-v0`，90 个 train episode / 10 个 val episode；配置为 `T=3`、`K=512`、`D=64`、batch size 2、10000 steps。运行目录为 `/root/autodl-tmp/runs/4d-jepa-architecture/group1_scratch_liftcube`。
- 训练结果：10000 steps 正常完成，耗时约 1422 秒（23.7 分钟）；最终 train loss `0.0723723`，最终 val loss `0.0723730`。`best.pt` 对应 step 8500，`last.pt` 对应 step 10000。
- 健康度：验证集 32768 个 latent 样本，`latent_std=0.1989`，effective rank（有效秩）`41/64`，`collapse_flag=false`，NaN/Inf 均为 0；未出现 OOM 或进程异常。
- 离线预测：`pred_mse=1.6535e-6`，persistence baseline MSE `9.6719e-8`，`prediction_gain=0.0585`（定义为 baseline/prediction）。该比值小于 1，表示当前模型尚未超过 persistence baseline；由于 baseline 本身很强，暂不能据此证明具备好的 embodied manipulation 表征能力。
- 产物：`best.pt`、`last.pt`、`metrics.jsonl`、`diagnostics.json`、`offline_metrics.json` 均已生成；本阶段未执行 MPC 或仿真闭环评估。
- 关联：`/root/autodl-tmp/runs/4d-jepa-architecture/group1_scratch_liftcube`；训练提交 `f72df32`，诊断/评估提交 `61bba03`。

## 2026-09-03 · Group 2 数据生成硬件阻塞

- 类型：Data / Experiment
- 计划：拟建立两个独立 Group 2 变体，源任务为 `PickCube-v0` + `StackCube-v0`，分别使用 200 条和 2000 条源轨迹；每个变体随后在 `LiftCube-v0` 上微调 10000 steps。
- 数据检查：原始轨迹中 `PickCube-v0` 和 `StackCube-v0` 各有 1000 条；现有 compact 目录各只有 100 条，因此必须先进行 ManiSkill2 replay（轨迹回放）。
- 冒烟结果：对 `PickCube-v0` 单条轨迹执行 replay 即在 SAPIEN Vulkan 初始化阶段失败：`vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`，并伴随无 `DISPLAY` 警告。
- 处理：按硬件阻塞规则停止，不启动 Group 2 训练；删除本次创建的 `/root/autodl-tmp/workspace/datasets/maniskill2/v0.5.3/replayed/PickCube-v0/count_0001` 临时目录。GPU 回到空闲状态。
- 关联：本次未修改训练源码、配置或正式数据集；Group 2 运行目录未创建。

## 2026-09-03 · 项目迁移至 A800 workspace

- 类型：Code / Config / Experiment
- 代码：将分支 `codex/persistent-project-layout` 及完整 Git 历史从 RTX 5090 服务器迁移到 `/workspace/users/lewm/projects/4d-jepa-architecture`；迁移基线为 `b797668`。
- 配置：将 Group 1 默认数据根目录改为 `/workspace/datasets/maniskill2/v0.5.3/compact`，默认运行根目录改为 `/workspace/users/lewm/runs/4d-jepa-architecture`。
- 数据：按用户要求不复制数据集；复用 A800 已有 compact 目录。`LiftCube-v0`、`PickCube-v0`、`StackCube-v0` 的当前 manifest 均为 100 episodes，90 train / 10 val。
- 实验产物：Group 1 完整运行目录迁移到 `/workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube`；`best.pt`、`last.pt`、训练日志、metrics 和诊断结果均保留。
- 验证：`best.pt` 与 `last.pt` 的源/目标 SHA-256 分别一致为 `a9622210...cce2b` 和 `a4137d27...f254`；A800 上 `last.pt` 严格加载成功，step 为 10000；真实 compact batch 形状为 `[2,4,512,3]` / `[2,3,7]` 且数值有限；完整测试为 `24 passed`。
- 关联：A800 Nix 环境 `/nix/lewm/4r-jepa-ms2/env`；数据迁移临时归档和失败分块均已清理。

## 后续记录格式

### YYYY-MM-DD · 简短标题

- 类型：Code / Config / Experiment
- 变更：改了什么。
- 原因：为什么改。
- 验证：运行了什么命令，结果是什么。
- 关联：Git commit 或 `/root/autodl-tmp/runs/4d-jepa-architecture/` 下的实验目录。
