# LiftCube 多步离线评估设计

## 目标

对现有 LiftCube `best.pt` 做可复现的多步潜变量预测评估，判断 WorldModel 在真实动作条件下的误差如何随预测步长增长，以及它是否优于简单基线。评估只读取已有 checkpoint 和 LiftCube 验证数据，不修改模型参数、不训练、不启动 ManiSkill/SAPIEN 仿真。

## 范围

本阶段新增一个独立评估入口 `scripts/eval_maniskill_group1_rollout.py`，并为数据对齐、指标和确定性行为增加测试。保留现有 `scripts/eval_maniskill_group1.py` 及其 `offline_metrics.json`，不改变既有单步评估的语义。

不在本阶段实现 MPC、CEM、奖励模型、成功分类头、环境闭环或跨任务评估，也不调整训练配置和 checkpoint。

## 输入与数据边界

- checkpoint：显式通过 `--checkpoint` 传入，首轮使用 `group1_scratch_liftcube/best.pt`。
- 数据：显式通过 `--data-root` 指向 `compact/LiftCube-v0`。
- 只使用 `manifest.json` 的 `val_episode_ids`；训练 episode 不参与评估。
- 当前数据契约为 `xyz [F,1024,3]`、`action [F-1,7]`。评估按 checkpoint 配置使用 `T=3`、`K=512`，复用项目现有的确定性点采样规则。
- 所有 horizon 使用能够支持最大 horizon 的同一批起点，避免不同样本集合造成不可比。

## 多步对齐与数据流

对 episode 中每个满足 `T + H_max` 帧要求的起点 `s`：

1. 上下文点云为 `xyz[s:s+T]`。
2. 未来动作从 `action[s+T-1:s+T-1+H_max]` 取得；第一个动作对应最后一帧上下文到第一帧目标的转移。
3. 真实未来点云为 `xyz[s+T:s+T+H_max]`。
4. `WorldModel.rollout(context, actions)` 生成预测潜变量 `[B,H_max,K,D]`。
5. 用同一模型的空间编码器编码真实未来点云，得到逐 horizon 的目标潜变量；编码和 rollout 全程使用 `eval()`、`torch.no_grad()`。

默认报告 `H={1,2,4,8}`。实现只执行一次 `H_max=8` rollout，再按 horizon 汇总，避免重复计算和样本漂移。

## 对照组

- `model_correct_action`：使用与轨迹严格对齐的真实动作。
- `persistence`：在所有 horizon 重复最后一帧上下文的空间潜变量。
- `model_zero_action`：将未来动作全部置零后执行同一 rollout。
- `model_shuffled_action`：使用固定种子在评估窗口之间做无固定点的循环置换，保留动作边际分布，并保证 `H=1` 也能测试动作敏感性。

打乱映射在评估开始时一次性生成，不依赖 DataLoader batch 边界。

## 指标与解释

每个 horizon 分别报告：

- latent MSE；
- 沿特征维计算、再对 token 和样本求均值的 cosine distance；
- `gain_vs_persistence = persistence_mse / model_correct_action_mse`，大于 1 才表示优于 persistence；
- 正确动作、零动作、打乱动作之间的误差和预测差异，用于判断模型是否真实利用动作。

同时记录评估 episode 数、窗口数、checkpoint step、checkpoint SHA-256、数据 manifest SHA-256、模型配置、固定随机种子和运行耗时。结果只作诊断，不设置“自动通过即模型可用于 MPC”的结论。

## 输出

默认写入 checkpoint 运行目录下的新文件 `rollout_metrics.json`，也允许通过 `--output` 指定其他位置。输出采用临时文件后原子替换，避免中断留下半截 JSON；不覆盖 `offline_metrics.json`、checkpoint 或训练日志。

控制台打印一张紧凑的逐 horizon 汇总表，并明确标记模型是否在各 horizon 击败 persistence、正确动作是否优于零动作和打乱动作。

## 环境隔离

A 阶段直接加载 WM，因此使用已经验证可加载 checkpoint 的 WM Python 运行时，但不安装、导入或启动 ManiSkill/SAPIEN，也不修改该环境或项目依赖。数据和评估产物位于项目源码目录之外。

未来 B 阶段的 LiftCube 仿真将使用独立 Python 环境和独立进程；仿真端只负责 `reset/step`、观测预处理和成功率统计，WM/CEM 端负责模型推理和候选动作评分。这样不会把仿真依赖重新耦合进 WM 项目。

## 错误处理

以下情况应在计算前失败并给出明确错误：checkpoint 或 manifest 缺失、训练/验证 ID 重叠、任务不是 LiftCube、动作维度不是 7、帧或动作长度不匹配、没有满足 `H_max` 的窗口、输入含 NaN/Inf、checkpoint 配置与输入形状不一致。

## 测试

- 人工小 episode 验证上下文、第一步动作和未来目标的索引对齐。
- 验证窗口不跨 episode，并且所有 horizon 使用同一评估集合。
- 验证 persistence、MSE、cosine distance 和 gain 公式。
- 验证零动作与跨窗口打乱动作的形状、无固定点和确定性。
- 验证结果 JSON 字段完整且重复运行结果一致。
- 用小模型/小数组做 CLI 冒烟测试，避免单元测试依赖完整数据集或长 GPU 计算。

## 验收标准

1. 新增测试全部通过，原有测试不回归。
2. `best.pt` 在 10 个 LiftCube 验证 episode 上完成 `H=1/2/4/8` 评估并退出码为 0。
3. JSON 和控制台表同时包含四种方法、全部 horizon、样本数量及数据与 checkpoint 来源信息。
4. 相同输入和随机种子重复运行得到一致结果。
5. checkpoint、已有指标、训练数据、Python 环境和依赖文件均未被修改。
