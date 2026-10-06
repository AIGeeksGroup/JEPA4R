# OGBench-IL-RL

JEPA4R 的 OGBench IL/RL 分支。当前下游接口是自采预训练的冻结 LeWM 表征，接官方 Cube 轨迹上的**单帧 MLP GCBC**。官方 state GCBC 保留为独立对照。

## 代码入口

```text
lewm_pcd/
├── jepa.py / module.py / train.py       原世界模型主干
├── encoders/                           2D ViT、DexArt 等 encoder
├── envs/                               原仿真与点云接口
├── config/                             原世界模型配置
├── scripts/data/                       上游自采配对协议
└── ogbench_il_rl/
    ├── models/frozen_lewm.py            冻结 encoder＋projector，保留 global192
    ├── models/policy.py                 current192＋goal192 → MLP512×3 → mean5
    │                                   unit Gaussian NLL、Adam3e-4
    ├── data/dataset.py                  官方 compact-boundary goals、单帧读取
    │                                   逻辑批次与预取 RNG 的精确恢复
    ├── data/transforms.py               原真实输入预处理，动作 identity
    ├── data/scene.py / rgb_renderer.py  原 Cube 场景与逐行 RGB 派生
    ├── data/rgb_shards.py               无损分片写入、提交和断点恢复
    ├── data/pointcloud.py               原裁剪采样＋仅空池的真实前景回退
    ├── data/view_geometry.py            相机、方块表面与遮挡验证
    ├── evaluation/native_single_frame.py 配对物理初态、真实 goal RGB、零 warmup
    ├── evaluation/official_state_*.py   官方 state 入口和独立 SR 验收
    ├── configs/gcbc.json               当前 MLP 配置与待验收项
    ├── configs/official_state.json     官方单 seed、100万步及评测配置
    ├── tests/                          数据边界、采样、冻结、恢复及 Flax 数值对照
    └── SOURCE_PROVENANCE.json          来源 SHA 与接口搬迁记录
docs/ogbench_il_rl/                     数据契约、API、依赖与验证范围
```

使用方式见 [接口说明](docs/ogbench_il_rl/README.md)，环境与源码锁见 [依赖说明](docs/ogbench_il_rl/DEPENDENCIES.md)。

## 当前结果与验证范围

| 路线 | 已验证 | 尚未验证 / 未完成 |
|---|---|---|
| 官方 Cube state GCBC | seed0、100万次更新；最终独立 SR **22/250 = 8.8%** | 论文8seed汇总未做 |
| 官方 Cube RGB 派生 | train1000 / val100完整轨迹；源动作、状态、边界和文件 SHA 独立验收 | 新 MLP 的完整真实数据训练尚未完成 |
| 2D LeWM → 单帧 MLP GCBC | CPU冻结、更新、严格恢复、有效1024与尾批576累积；与官方 Flax 头的输出/loss/梯度/同梯度Adam数值对照 | 新协议真实数据 CUDA/BF16、正式10epoch及最终 SR |
| 3D LeWM → GCBC | 同 Cube 轨迹配对 pilot、真实前景空裁剪回退 | 单帧 MLP 的真实3D数据门、独立精度验收、全量派生/训练/SR |
| 2D / DexArt LeWM＋CEM | 历史 custom fixed5，5任务各50，均 **0/250** | 不作为官方 Cube BC 对照结果 |

历史 history3 Transformer 的 CUDA/BF16 smoke 与旧自采 BC checkpoint 保留在实验记录中，不能作为当前单帧 MLP 的验收证据。原 `lewm_pcd/README.md` 的早期 fragment CEM 表也不与本分支 native 全 episode SR 混用。

本分支收最终接口、当前配置与必要正确性测试。worker 候选测速、失败重试脚本、过程日志、中间快照、数据和权重不上传。原世界模型目录保持兼容；发布整理在独立 checkout 完成，运行中的服务器工作树未改写。