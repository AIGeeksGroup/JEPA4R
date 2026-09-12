# Group 1 Scratch 训练流程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `LiftCube-v0` 的 90/10 episode 划分准备一个可恢复、可验证、默认不启动长训练的 scratch WorldModel 训练入口。

**Architecture:** 新增 Group 1 专用 YAML 和脚本，复用现有 `ManiSkillSequenceDataset` 与 `WorldModel`。脚本将数据契约检查、动作统计、preflight、训练、验证和 checkpoint 保存放在一个清晰入口中，避免改变现有合成数据训练入口。

**Tech Stack:** Python 3.12、PyTorch 2.7、CUDA、spconv、OmegaConf、pytest、ManiSkill compact NPZ。

## Global Constraints

- 只使用 `LiftCube-v0`。
- 只读取 manifest 中 90 个训练 episode 和 10 个验证 episode。
- 不读取 `PickCube-v0` 或 `StackCube-v0`。
- 训练模型必须随机初始化；恢复训练必须显式传 `--resume`。
- 准备阶段只执行 preflight，不启动长时间训练。
- 依赖不升级 PyTorch/CUDA 栈。

### Task 1: Group 1 配置与数据契约

**Files:**
- Create: `configs/maniskill_group1_scratch.yaml`
- Create: `tests/test_maniskill_group1.py`
- Create: `scripts/train_maniskill_group1.py`

**Interfaces:**
- `validate_group1_manifest(root: Path) -> dict`
- `load_group1_config(path: Path, overrides: list[str]) -> tuple[WorldModelConfig, Group1TrainConfig]`

- [ ] 测试配置路径、任务名和 90/10 划分。
- [ ] 运行 focused test，确认脚本缺失时失败。
- [ ] 添加配置和最小数据契约函数。
- [ ] 运行 focused test，确认通过。

### Task 2: Preflight 与训练/验证入口

**Files:**
- Modify: `scripts/train_maniskill_group1.py`
- Modify: `tests/test_maniskill_group1.py`

**Interfaces:**
- `run_preflight(...) -> dict`
- `main(argv: Sequence[str]) -> None`

- [ ] 增加训练动作统计、模型构造和单 batch preflight。
- [ ] 增加 checkpoint、metrics 和恢复逻辑。
- [ ] 运行远端 preflight，确认显存和梯度有限。
- [ ] 只在 preflight 后检查 GPU，不启动长训练。

### Task 3: 台账与提交

**Files:**
- Modify: `PROJECT_LOG.md`
- Modify: `ex_pyproject.md`

- [ ] 记录 Group 1 流程、preflight 和 GPU 状态。
- [ ] 运行全量 pytest、git diff --check、git status。
- [ ] 提交 Git commit。
