# ManiSkill2 Dataset Adapter Implementation Plan

**Goal:** 将已有的 LiftCube-v0、PickCube-v0、StackCube-v0 compact demonstrations 接入当前 4D JEPA WorldModel，并完成数据到模型训练的一次闭环验证。

**Architecture:** 复用服务器上已有参考项目的 episode 级切分、NPZ 缓存和固定点数点云约定；新增通用 `ManiSkillSequenceDataset` 输出 `[T+1,K,3]` 与 `[T,7]`。smoke runner 按任务分开 batch，逐任务执行一次前向、反向和 AdamW 更新。

**Validation:** focused dataset tests、项目测试、三任务 smoke runner；所有输出和梯度必须有限，数据窗口不得跨 episode。
