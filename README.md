# JEPA4R

## 4R-JEPA — based on ManiSkill2-Learn and LeWM

**实现位于 `4r-jepa-based-on-maniskill2-lewm` 分支，项目子目录为
[`4r-jepa-maniskill2-lewm/`](4r-jepa-maniskill2-lewm/README.md)。浏览 GitHub 时请切换到该分支；`main` 没有本次实现。**

包含点云 PPO、连续帧世界模型训练、冻结世界模型表征后的 PPO 三个阶段。核心代码直接提交在本仓库中：

| 核心功能 | 源码 |
| --- | --- |
| WorldModel、逐帧 encoder 包装、LeWMLoss、forward/encode/rollout | [model.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/model.py) |
| 独立时间聚合 TemporalEncoder | [temporal.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/temporal.py)、[配置](4r-jepa-maniskill2-lewm/configs/temporal_encoder.yaml) |
| 官方 PointNet / critic 复用及末层 Identity | [rl_components.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/rl_components.py)、[pointnet_identity.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/pointnet_identity.py) |
| 冻结 encoder 接 PPO、连续帧历史 | [frozen_ppo.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/frozen_ppo.py)、[history.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/history.py) |
| 原 PPO / 世界模型训练 / 冻结 PPO 入口 | [run_ppo.py](4r-jepa-maniskill2-lewm/scripts/run_ppo.py)、[train_wm.py](4r-jepa-maniskill2-lewm/scripts/train_wm.py)、[run_frozen_ppo.py](4r-jepa-maniskill2-lewm/scripts/run_frozen_ppo.py) |
| 并行启发式采集及 PickCube 保持阶段 | [collect_wm.py](4r-jepa-maniskill2-lewm/scripts/collect_wm.py)、[process_env_pool.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/process_env_pool.py)、[pickcube_oracle.py](4r-jepa-maniskill2-lewm/src/ms2_lewm/pickcube_oracle.py) |
| 世界模型规划评测 | [evaluate_wm.py](4r-jepa-maniskill2-lewm/scripts/evaluate_wm.py) |

```bash
git clone --branch 4r-jepa-based-on-maniskill2-lewm --single-branch \
  git@github.com:AIGeeksGroup/JEPA4R.git
cd JEPA4R/4r-jepa-maniskill2-lewm
python3 scripts/bootstrap_sources.py
python3 scripts/bootstrap_sources.py --check
```

`vendor/` 不随 Git 提交：上述命令按 [sources.lock.json](4r-jepa-maniskill2-lewm/repro/sources.lock.json)
获取四个固定提交的上游仓库。上游源码不修改；本项目模型和环境适配均在上表的源码中。
安装环境与各阶段启动命令见 [项目 README](4r-jepa-maniskill2-lewm/README.md#复现入口)。

**复现范围是代码、固定依赖与启动流程。** PPO 正式训练与全量采集已有本机运行记录；真实数据世界模型训练、冻结表征 PPO 及完整论文结果尚未完成。
数据集、虚拟环境、运行日志和 checkpoint 不随仓库分发，不能将接口测试通过等同于论文指标复现。
