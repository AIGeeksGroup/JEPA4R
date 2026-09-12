# 4D JEPA 项目存储布局与变更记录设计

日期：2026-09-03

## 目标

将项目代码从容量有限的容器根分区迁移到持久化工作盘，同时保留旧路径兼容性；在仓库内建立代码、配置和实验变更脉络，并将依赖与运行环境单独记录。

## 已确认的存储布局

```text
/root/autodl-tmp/
├── projects/
│   └── 4d-jepa-architecture/       # 实际 Git 工作树
└── runs/
    └── 4d-jepa-architecture/       # 实验输出、日志和 checkpoint

/root/4d-jepa-architecture           # 符号链接，指向实际 Git 工作树
/autodl-pub                          # 只读共享数据盘，仅用于读取数据
```

### 选择理由

- `/root` 位于约 30 GB 的容器 overlay 根分区，适合少量入口文件，不适合持续增长的实验数据。
- `/root/autodl-tmp` 是约 50 GB 的可写持久化卷，适合保存代码、实验输出和 checkpoint。
- `/autodl-pub` 是约 7.3 TB 的只读 NFS 共享盘，适合读取共享数据集，不作为项目写入目录。
- 代码与实验输出分离，便于转移代码时不被大型生成文件拖累。
- 保留 `/root/4d-jepa-architecture` 符号链接，避免已有命令、脚本或用户习惯失效。

## 项目脉络记录

仓库根目录使用 `PROJECT_LOG.md`，只追加记录以下内容：

- `Code`：源码结构和实现变更。
- `Config`：配置、超参数和运行入口变更。
- `Experiment`：实验目的、命令、数据来源、结果、验证状态和输出目录。

每条记录至少包含日期、类型、变更内容、原因、验证方式，以及关联 Git commit（提交）或实验目录。服务器目录迁移属于本次初始化记录的一部分，但后续基础设施信息不作为该文件的常规记录类别。

## 依赖与环境记录

仓库根目录使用 `ex_pyproject.md`，单独维护：

- Python 和 Conda 环境信息。
- CUDA、PyTorch、torchvision、spconv 等运行时版本。
- `requirements*.txt`、`environment.yml`、`pyproject.toml` 的依赖变化。
- 安装、升级、降级、兼容性验证和失败尝试。

该文件不记录源码、配置或实验过程，避免与 `PROJECT_LOG.md` 的项目脉络混杂。

## 迁移步骤

1. 在 `/root/autodl-tmp/projects` 和 `/root/autodl-tmp/runs` 下创建项目目录。
2. 将现有 Git 工作树完整移动到持久化项目目录，保留 `.git`、分支、远端和工作区状态。
3. 在旧路径创建指向新路径的符号链接。
4. 创建 `PROJECT_LOG.md` 和 `ex_pyproject.md`，写入初始记录。
5. 用 Git 状态、远端地址、最新提交、符号链接解析和磁盘挂载点进行校验。

## 边界与约束

- 不移动或清理 `/root/miniconda3`、`/root/.cache` 和 `/root/preload_cache`，它们不属于本次项目整理范围。
- 不将 `/autodl-pub` 改为可写或复制其数据。
- 不修改模型代码、训练配置和依赖版本；本次只调整目录布局并新增记录文档。
- 不自动推送 GitHub；本地提交只用于保留服务器上的变更脉络。

## 验证标准

- `/root/autodl-tmp/projects/4d-jepa-architecture/.git` 存在且仓库可正常执行 Git 命令。
- `/root/4d-jepa-architecture` 解析到新项目目录。
- `git status --short --branch` 能显示原分支和远端跟踪关系，最新提交不变。
- 两个 Markdown 文件位于仓库根目录且内容完整。
- `/root` 根分区不再承载该项目的实际工作树。
