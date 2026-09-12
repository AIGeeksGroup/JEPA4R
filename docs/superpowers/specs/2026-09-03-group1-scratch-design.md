# Group 1 Scratch 训练流程设计

## 目标

为 `LiftCube-v0` 建立一个从随机初始化 WorldModel 开始的、可检查后直接启动的训练流程。训练只使用该任务的 90 条训练 episode，10 条验证 episode 仅用于验证，不读取 `PickCube-v0` 或 `StackCube-v0`。

## 数据与模型

- 数据根目录：`/root/autodl-tmp/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0`
- 输入窗口：`points [B,T+1,K,3]`、`actions [B,T,7]`
- 默认模型：`T=3`、`K=512`、`D=64`、`C=3`
- 动作均值/标准差从训练 episode 计算，不使用验证集统计量
- 模型每次启动都随机初始化；恢复训练只能显式传入 `--resume`

## 训练链路

1. 读取并验证 manifest，强制检查任务名和 90/10 episode 划分。
2. 构造训练/验证 DataLoader，确保窗口不跨 episode。
3. 用训练动作统计填充 `maniskill2` 动作编码器。
4. 随机初始化 WorldModel，执行前向、预测损失 + SIGReg、反向和梯度裁剪。
5. 按间隔记录训练/验证指标，保存 `last.pt` 和最佳 `best.pt`。
6. 启动前可运行 preflight：只取一个 batch 做前向/反向，不写 checkpoint，并输出显存峰值。

## 验证与停止边界

- 准备阶段只执行配置验证、数据读取、单 batch preflight 和 GPU 状态检查。
- 不执行 Group 1 的长时间训练，不执行诊断，不执行 MPC，不读取 Pick/Stack 数据。
- 直接启动命令由脚本输出，用户确认后再执行。
