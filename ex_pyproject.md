# 4D JEPA 依赖与运行环境台账

本文件只记录 Python、Conda、CUDA、PyTorch、spconv 及项目依赖文件的变更和验证，采用只追加方式。源码、配置和实验过程记录在 `PROJECT_LOG.md`。

## 2026-09-03 · 环境基线

- Python：`/root/miniconda3/bin/python` 为 Python 3.12.3；非交互 SSH shell 的 `PATH` 未包含 `python` 命令。
- Conda：`/root/miniconda3/bin/conda` 可用；当前仅检测到 `base` 环境（`/root/miniconda3`）。
- GPU/CUDA：NVIDIA GeForce RTX 5090，显存 32607 MiB，驱动 595.58.03；`nvidia-smi` 可用；`nvcc` 不在 `PATH`，未检测到 CUDA Toolkit 编译器。
- PyTorch：`torch==2.7.0+cu128`，`torch.version.cuda` 为 `12.8`，`torch.cuda.is_available()` 为 `True`。
- torchvision：`torchvision==0.22.0+cu128`。
- spconv：当前 base 环境中未检测到可导入的 `spconv`。
- 其他项目依赖：本次未安装、升级或降级依赖；未对环境做写操作。
- 依赖声明：`requirements.txt`、`requirements-cu128.txt`、`environment.yml`、`pyproject.toml` 均未因本次整理修改。
- 文件校验：
  - `requirements.txt` SHA-256：`701b0ac5f46232ec53e79ccd3dca8c100f2f96ff2378b4d67177086542c59042`
  - `requirements-cu128.txt` SHA-256：`d7421c5c93d3a02aa22c4e1f8d87d030de0438ea505eeb4f8042e2b6d2717a80`
  - `environment.yml` SHA-256：`8cce72bc4f0870fbded379f62d0b0c1429d8362b45d7caa732d05820350d83b3`
  - `pyproject.toml` SHA-256：`ef97f2bcdcdbe0c5362b9e88b9f67e1f5f00071c40d2217438b75e7a1263ff47`
- 验证命令：`/root/miniconda3/bin/python --version`、`/root/miniconda3/bin/conda env list`、`/root/miniconda3/bin/python -c 'import importlib.util ...'`、`nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`、`sha256sum requirements.txt requirements-cu128.txt environment.yml pyproject.toml`。
- 注意：README 的安装说明以 Python 3.11 为要求；当前 base 环境为 Python 3.12.3，尚未安装项目声明的完整依赖，也未运行测试或训练。

## 后续记录格式

### YYYY-MM-DD · 简短标题

- 变更：安装、升级、降级或移除的依赖/环境项。
- 原因：为什么改。
- 验证：版本、导入、GPU/CUDA 或测试验证结果。
- 关联：Git commit 或安装命令。

## 2026-09-03 · pytest 安装与测试验证

- 变更：在 `/root/miniconda3` base 环境安装项目声明的 `pytest==9.1.1`；其依赖 `iniconfig==2.3.0` 和 `pluggy==1.6.0` 一并安装/升级。
- 安装来源：PyPI 官方索引 `https://pypi.org/simple`。默认阿里云镜像未提供 `pytest==9.1.1`，因此未使用默认镜像完成安装。
- 验证：`/root/miniconda3/bin/python -m pytest -q` 已启动，但测试收集阶段因项目运行依赖 `omegaconf` 未安装而失败；未继续安装整套项目依赖。
- 阻塞：当前测试结果为 `1 error during collection`，错误为 `ModuleNotFoundError: No module named 'omegaconf'`。这不是 pytest 安装错误。
- 关联：服务器仓库分支 `codex/persistent-project-layout`；本次环境变更尚未推送 GitHub。

## 2026-09-03 · omegaconf 安装与测试复测

- 变更：在 `/root/miniconda3` base 环境安装依赖声明中的 `omegaconf==2.3.1`；其直接依赖 `antlr4-python3-runtime==4.9.3` 一并安装。
- 安装来源：PyPI 官方索引 `https://pypi.org/simple`。
- 验证：`/root/miniconda3/bin/python -c 'import omegaconf; print(omegaconf.__version__)'` 输出 `2.3.1`。
- 测试：`/root/miniconda3/bin/python -m pytest -q` 已越过 `omegaconf` 导入，现于测试收集阶段因项目依赖 `spconv.pytorch` 未安装而失败；结果为 `1 error during collection`。
- 范围：本次仅按授权安装 `omegaconf` 及其直接依赖，未安装 `spconv` 或完整项目依赖，未修改源码、配置和依赖声明。
- 关联：服务器仓库分支 `codex/persistent-project-layout`；待本次台账提交后仍不推送 GitHub。

## 2026-09-03 · ManiSkill2 smoke 运行依赖

- 变更：在 `/root/miniconda3` base 环境安装 `spconv-cu126==2.3.8`、`cumm-cu126==0.7.11`、`timm==1.0.29`、`addict==2.4.0`、`einops==0.8.2`；同时安装其依赖 `ccimport==0.4.4`、`pccm==0.4.16`、`pybind11==3.1.0`、`fire==0.7.1`、`ninja==1.13.2`、`lark==1.3.1`、`portalocker==4.3.0`、`huggingface_hub==1.29.0`、`safetensors==0.8.0` 等。
- 原因：当前 WorldModel 使用 `spconv.pytorch`，timm/addict/einops 为模型导入链所需依赖；缺失时无法执行真实数据到模型 smoke。
- 安装来源：清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`；官方 PyPI 下载曾长时间停滞，已停止无进展进程后改用带 `--timeout 60 --retries 2` 的镜像安装。
- 验证：`/root/miniconda3/bin/python -c 'import torch, spconv.pytorch, timm, addict, einops; ...'` 输出 `torch 2.7.0+cu128 cuda True`，`spconv` 导入路径为 `/root/miniconda3/lib/python3.12/site-packages/spconv/pytorch/__init__.py`；未升级现有 PyTorch/CUDA 栈。
- 关联：安装日志 `/root/autodl-tmp/pip_maniskill_install.log`；服务器仓库分支 `codex/persistent-project-layout`。

## 2026-09-03 · 原始轨迹检查依赖

- 变更：在 `/root/miniconda3` base 环境安装 `h5py==3.11.0`。
- 原因：读取并验证官方 ManiSkill2 `trajectory.h5` 文件结构；不改变现有 PyTorch/CUDA 栈。
- 安装来源：清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`。
- 验证：三个任务的 HDF5 均可读取 100/1000/1000 个 episode；抽查 `traj_0` 的 `actions`、`env_states`、`success` 形状和 dtype 正常。
- 关联：安装日志 `/root/autodl-tmp/pip_h5py_install.log`。

## 2026-09-03 · Group 1 训练 preflight 环境验证

- 变更：未新增依赖；复用已安装的 PyTorch 2.7.0+cu128、spconv-cu126 2.3.8、OmegaConf、h5py 等运行栈。
- 验证：`scripts/train_maniskill_group1.py --preflight` 在 RTX 5090 上完成真实 LiftCube batch 的前向/反向和有限性检查；CUDA 可用，峰值 allocated `250,983,424` bytes，reserved `358,612,992` bytes。
- 空闲状态：preflight 结束后 `nvidia-smi` 显示显存使用 2 MiB、GPU 利用率 0%，无 Group 1 训练进程。
- 关联：配置 `configs/maniskill_group1_scratch.yaml`；本次未启动长时间训练。

## 2026-09-03 · Group 1 训练与诊断复用环境

- 变更：本次 Group 1 训练、latent 诊断和离线评估未新增、升级或降级 Python 依赖；继续复用当前 base 环境中的 PyTorch 2.7.0+cu128、spconv-cu126 2.3.8、OmegaConf 和 h5py。
- 验证：训练在 RTX 5090 上完成 10000 steps；诊断与离线评估脚本均以退出码 0 完成，并生成运行目录下的 JSON 结果文件。
- 关联：`/root/autodl-tmp/runs/4d-jepa-architecture/group1_scratch_liftcube`。

## 2026-09-03 · Group 2 replay 运行时检查失败

- 变更：未新增或修改 Python 依赖；仅复用服务器上已有的 Python 3.11 runtime 与 ManiSkill2/SAPIEN site-packages 做单条 replay 冒烟。
- 验证：`PickCube-v0` 单条轨迹在 SAPIEN Vulkan device 创建阶段退出，错误为 `ErrorExtensionNotPresent`；无训练进程启动，GPU 显存回到约 2 MiB。
- 结论：当前运行时/硬件无法完成点云 replay，按任务要求停止 Group 2 数据生成与训练。
- 关联：临时 replay 目录已清理；未产生 Group 2 checkpoint。

## 2026-09-03 · A800 Nix 环境对齐

- 变更：不复制 RTX 5090 服务器的 Conda 环境；在 A800 现有 `/nix/lewm/4r-jepa-ms2/env` 中补齐项目模型依赖，并以 editable 模式安装当前项目。
- 核心版本：Python 3.11.9、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、spconv-cu120 2.3.6、timm 1.0.29、addict 2.4.0、einops 0.8.2、OmegaConf 2.3.1、NumPy 1.26.4、pytest 8.3.5。
- 声明同步：`pyproject.toml` 与 `requirements.txt` 已改为上述 A800 实际版本；RTX 50 系专用的 `requirements-cu128.txt` 保持不变。
- 路径：代码与运行产物保存在 `/workspace/users/lewm/`，Python 环境仅保存在 `/nix/lewm/4r-jepa-ms2`，未向 A800 root 文件系统复制环境。
- 验证：所有声明中的直接依赖均与已安装版本逐项匹配并可导入；`/nix/lewm/4r-jepa-ms2/env/bin/python -m pytest -q` 输出 `24 passed, 9 warnings`；真实 compact 数据读取和迁移 checkpoint 严格加载通过。
- 复现：使用 `/nix/lewm/4r-jepa-ms2/uv/bin/uv pip install --python /nix/lewm/4r-jepa-ms2/env/bin/python -r requirements.txt`，随后执行同一 uv 命令并加 `--no-deps -e .` 安装项目。
