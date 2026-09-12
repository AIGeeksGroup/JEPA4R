# 4D JEPA 持久化项目布局实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** 将 4D JEPA 代码工作树迁移到可写持久化工作盘，保留旧路径兼容入口，并建立项目变更日志与依赖环境台账。

**Architecture:** `/root/autodl-tmp/projects/4d-jepa-architecture` 作为实际 Git 工作树，`/root/4d-jepa-architecture` 作为兼容符号链接；实验产物单独放在 `/root/autodl-tmp/runs/4d-jepa-architecture`。仓库根目录的 `PROJECT_LOG.md` 只记录代码、配置和实验脉络，`ex_pyproject.md` 单独记录依赖与运行环境。

**Tech Stack:** Linux shell、Git、Markdown、现有 Python/Conda/CUDA 运行环境。

## Global Constraints

- `/root` 位于约 30 GB 的容器 overlay 根分区，不承载项目实际工作树。
- `/root/autodl-tmp` 是约 50 GB 的可写持久化卷，用于项目代码和实验输出。
- `/autodl-pub` 是只读 NFS 共享盘，只用于读取共享数据。
- 不移动或清理 `/root/miniconda3`、`/root/.cache` 和 `/root/preload_cache`。
- 不修改模型代码、训练配置和依赖版本；本次只调整目录布局并新增记录文档。
- 不自动推送 GitHub；本地 Git 提交只用于保留服务器上的变更脉络。
- 迁移前必须确认目标路径不存在，避免覆盖已有数据。

## 文件与目录清单

- Move: `/root/4d-jepa-architecture` → `/root/autodl-tmp/projects/4d-jepa-architecture`
- Create: `/root/4d-jepa-architecture`（指向新项目目录的符号链接）
- Create: `/root/autodl-tmp/runs/4d-jepa-architecture/`
- Create: `/root/autodl-tmp/projects/4d-jepa-architecture/PROJECT_LOG.md`
- Create: `/root/autodl-tmp/projects/4d-jepa-architecture/ex_pyproject.md`
- Existing: `/root/autodl-tmp/projects/4d-jepa-architecture/docs/superpowers/specs/2026-09-03-storage-layout-design.md`

### Task 1: 将 Git 工作树迁移到持久化项目目录

**Files/Directories:**
- Move: `/root/4d-jepa-architecture`
- Create: `/root/autodl-tmp/projects/4d-jepa-architecture`
- Create: `/root/autodl-tmp/runs/4d-jepa-architecture`
- Create: `/root/4d-jepa-architecture` symlink

**Interfaces:**
- Consumes: 当前仓库 `/root/4d-jepa-architecture`，其 `main` 分支、`origin` 远端和工作区状态。
- Produces: 新路径上的同一 Git 工作树，以及旧路径的兼容入口。

- [ ] **Step 1: 迁移前读取并锁定仓库状态**

  ```bash
  set -eu
  src=/root/4d-jepa-architecture
  dst=/root/autodl-tmp/projects/4d-jepa-architecture
  test -d "$src/.git"
  test ! -e "$dst"
  git -C "$src" status --short --branch
  git -C "$src" remote -v
  git -C "$src" log -1 --oneline
  ```

  Expected: 源目录是 Git 仓库；目标目录不存在；当前分支为 `main`，远端为 `origin`，最新提交为迁移前的仓库提交。

- [ ] **Step 2: 创建持久化目录并移动工作树**

  ```bash
  set -eu
  src=/root/4d-jepa-architecture
  dst=/root/autodl-tmp/projects/4d-jepa-architecture
  mkdir -p /root/autodl-tmp/projects /root/autodl-tmp/runs/4d-jepa-architecture
  mv -- "$src" "$dst"
  ln -s -- "$dst" "$src"
  ```

  Expected: `readlink -f /root/4d-jepa-architecture` 输出 `/root/autodl-tmp/projects/4d-jepa-architecture`；目标目录包含 `.git`。

- [ ] **Step 3: 验证路径兼容和 Git 元数据未变**

  ```bash
  test -L /root/4d-jepa-architecture
  test -d /root/autodl-tmp/projects/4d-jepa-architecture/.git
  git -C /root/4d-jepa-architecture status --short --branch
  git -C /root/autodl-tmp/projects/4d-jepa-architecture remote get-url origin
  git -C /root/autodl-tmp/projects/4d-jepa-architecture log -1 --oneline
  ```

  Expected: 旧路径和新路径都能执行 Git 命令；分支、`origin` 地址和最新提交一致；没有意外工作区改动。

### Task 2: 创建项目脉络日志与依赖环境台账

**Files:**
- Create: `/root/autodl-tmp/projects/4d-jepa-architecture/PROJECT_LOG.md`
- Create: `/root/autodl-tmp/projects/4d-jepa-architecture/ex_pyproject.md`

**Interfaces:**
- Consumes: Task 1 迁移后的 Git 提交、目录布局和当前 Python/Conda/CUDA 环境信息。
- Produces: 可随仓库转移的项目脉络文件，以及独立的依赖/环境变更文件。

- [ ] **Step 1: 采集当前代码和环境基线**

  ```bash
  cd /root/autodl-tmp/projects/4d-jepa-architecture
  git rev-parse --short HEAD
  git branch --show-current
  python --version
  if command -v conda >/dev/null 2>&1; then conda env list; else echo 'conda: unavailable'; fi
  python -c "import importlib.util; print('torch:', bool(importlib.util.find_spec('torch'))); print('spconv:', bool(importlib.util.find_spec('spconv')))"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; else echo 'nvidia-smi: unavailable'; fi
  ```

  Expected: 命令输出作为台账的事实来源；不可用的组件明确记录为 unavailable，不凭空填写版本。

- [ ] **Step 2: 创建 `PROJECT_LOG.md` 初始记录**

  文件内容必须包含：

  ```markdown
  # 4D JEPA 项目变更脉络

  本文件只记录代码、配置和实验变更；每条记录包含日期、类型、变更、原因、验证和关联提交或实验目录。

  ## 2026-09-03 · 初始化基线

  - 类型：Code / Config / Experiment
  - 代码基线：记录当前 Git commit、分支和远端。
  - 配置基线：记录 `configs/default.yaml`、安装入口和当前未修改声明。
  - 实验基线：尚未运行本项目实验；实验输出预留于 `/root/autodl-tmp/runs/4d-jepa-architecture/`。
  - 验证：记录仓库状态、路径兼容检查和本次未执行训练的事实。
  - 关联：记录当前 Git commit。
  ```

  Expected: 文件不记录后续依赖版本变更；实验尚未运行时明确写明，而不是填写推测结果。

- [ ] **Step 3: 创建 `ex_pyproject.md` 依赖台账**

  文件内容必须包含：

  ```markdown
  # 4D JEPA 依赖与运行环境台账

  本文件只记录 Python、Conda、CUDA、PyTorch、spconv 及项目依赖文件的变更和验证。

  ## 2026-09-03 · 环境基线

  - Python：填入命令实测结果。
  - Conda：填入环境列表或 unavailable。
  - CUDA/GPU：填入 `nvidia-smi` 实测结果或 unavailable。
  - PyTorch/spconv：填入导入探测结果；未安装时明确记录。
  - 依赖声明：`requirements.txt`、`requirements-cu128.txt`、`environment.yml`、`pyproject.toml` 未因本次整理修改。
  - 验证命令：列出实际执行的命令和结果。
  ```

  Expected: 台账只使用实测信息；失败、缺失和未验证项目明确写出。

### Task 3: 完整性、空间和可转移性校验

**Files/Directories:**
- Verify: `/root/autodl-tmp/projects/4d-jepa-architecture`
- Verify: `/root/4d-jepa-architecture`
- Verify: `/root/autodl-tmp/runs/4d-jepa-architecture`
- Verify: `/root` and `/root/autodl-tmp` mount capacity

**Interfaces:**
- Consumes: Task 1 的路径布局和 Task 2 的两个 Markdown 文件。
- Produces: 迁移完成证据；必要时补充一条本地 Git 提交。

- [ ] **Step 1: 验证仓库、日志和兼容入口**

  ```bash
  set -eu
  repo=/root/autodl-tmp/projects/4d-jepa-architecture
  test -f "$repo/PROJECT_LOG.md"
  test -f "$repo/ex_pyproject.md"
  test -f "$repo/README.md"
  test -L /root/4d-jepa-architecture
  test "$(readlink -f /root/4d-jepa-architecture)" = "$repo"
  git -C "$repo" status --short --branch
  git -C "$repo" diff --check
  ```

  Expected: 两个台账、README 和 `.git` 均可访问；符号链接指向唯一实际工作树；Markdown 无空白错误。

- [ ] **Step 2: 验证空间分配与实验目录边界**

  ```bash
  df -hT / /root/autodl-tmp /autodl-pub
  findmnt -T /root
  findmnt -T /root/autodl-tmp
  findmnt -T /autodl-pub
  du -sh /root/autodl-tmp/projects/4d-jepa-architecture /root/autodl-tmp/runs/4d-jepa-architecture
  ```

  Expected: 实际工作树位于 `/root/autodl-tmp` 持久化挂载；实验目录为空但已创建；`/autodl-pub` 仍为只读挂载。

- [ ] **Step 3: 记录并提交仓库内台账**

  ```bash
  cd /root/autodl-tmp/projects/4d-jepa-architecture
  git add PROJECT_LOG.md ex_pyproject.md
  git -c user.name='Codex' -c user.email='codex@localhost' commit -m 'docs: add project and dependency ledgers'
  git status --short --branch
  ```

  Expected: 只提交两个台账文件；工作区干净；不推送远端。

## Completion Checklist

- [ ] 实际 Git 工作树位于 `/root/autodl-tmp/projects/4d-jepa-architecture`。
- [ ] `/root/4d-jepa-architecture` 保留为兼容符号链接。
- [ ] 实验输出目录位于 `/root/autodl-tmp/runs/4d-jepa-architecture`。
- [ ] `PROJECT_LOG.md` 只承担代码、配置和实验脉络记录。
- [ ] `ex_pyproject.md` 只承担依赖和环境变更记录。
- [ ] Git 元数据、远端配置和原始提交未被破坏。
- [ ] 校验命令通过，且没有推送 GitHub。
