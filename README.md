# OGBench-IL-RL

JEPA4R 的 OGBench IL/RL 分支。当前下游接口是自采预训练的冻结 LeWM 表征，接官方 Cube 轨迹上的**单帧 MLP GCBC**。官方 state GCBC 保留为独立对照。

## 实验目标与顺序

目标是比较：同一份官方 Cube 下游轨迹上，原始 state、2D LeWM 表征和3D LeWM 表征分别接 GCBC 后的成功率。

上游 LeWM 用自采配对数据预训练；下游 BC 用官方 Cube train1000 / val100轨迹训练。当前采用单帧 current192＋goal192 → MLP512×3 → action5，只训练动作头，冻结 encoder＋projector。

| 顺序 | 实验 | 计划 / 当前结果 |
|---|---|---|
| 1 | 自采数据预训练 2D、3D LeWM | 已有两组 epoch10 权重，重载与真实数据 encode/predict/loss 检查通过 |
| 2 | 两组 LeWM＋CEM | 已完成两组全量评测；各250条，均 **0%** |
| 3 | 官方 Cube state → GCBC | 已完成 seed0 / 100万步训练及250条最终评测；**SR 8.8%** |
| 4 | 官方 Cube 派生 RGB → 冻结2D LeWM → MLP GCBC | 全量RGB已完成；接下来依次做新版smoke、10epoch训练、最终250条SR |
| 5 | 同 Cube 轨迹派生 PCD → 冻结3D LeWM → MLP GCBC | 小样本PCD已完成；2D验收后推进全量PCD、独立smoke、10epoch训练、最终250条SR |
| 6 | RGB/DINO GCBC、RGB/DINO GCIQL及state方法对照 | 两组LeWM首轮训练和SR完成后推进；各组先过官方Cube数据上的smoke，再锁定正式预算 |

两组 LeWM 下游先各跑10epoch。得到独立验证集与最终SR结果后，再决定是否延长训练。新实验继续保留数据、权重、配置、日志和退出码；这里的顺序不代表所有阶段都已经执行。

## 主线进度

**状态快照：2026-10-06 16:37（北京时间）。当前在准备新版2D MLP的真实数据smoke，正式2D训练尚未开始。**

“未做”表示尚未执行；“完成”表示该阶段已按表中预算执行并验收。这里的 **smoke** 指少量真实数据的训练更新、保存/独立重载和环境动作接口检查；CPU合成数据及数值单测单独标注。训练结束与最终SR评测是两个阶段。

| 路线 | 数据准备 | 真实数据 / 环境 smoke | 正式训练 | 全量评测与结果 |
|---|---|---|---|---|
| 官方 Cube state → GCBC | **完成**：官方train1000 / val100 | **通过**：CPU、GPU更新/重载及动作接口 | **完成**：seed0，100万次更新 | **完成**：250条，**22/250＝8.8%** |
| 2D LeWM → 单帧MLP GCBC | **全量完成**：派生RGB，train1000 / val100 | **新版smoke未做**；MLP/采样器/恢复CPU单测已通过 | **未做**；计划10epoch | **未做**；计划250条，SR尚无结果 |
| 3D LeWM → 单帧MLP GCBC | **小样本完成**：train/val各1条；**全量PCD未做** | **新版3D smoke未做** | **未做**；计划10epoch | **未做**；计划250条，SR尚无结果 |

全量RGB实际完成1100条轨迹、1,101,100原始行；源动作、状态、行ID、轨迹边界与文件SHA独立检查通过，退出码0。新版单帧训练使用100万有效train transition及10万有效val transition。

新版2D的CPU单测已覆盖冻结状态、动作头更新、严格恢复、有效batch1024和尾批576梯度累积，以及官方Flax头的输出/loss/梯度/同梯度Adam数值对照。**这些结果不代表真实数据GPU smoke通过，也不代表正式训练或SR已完成。**

历史history3 Transformer的CUDA/BF16 smoke通过，但当前路线已改为单帧MLP，需重新做对应smoke。旧自采10k下游BC已停止，epoch1–5保留；不计入本表官方Cube轨迹上的新版MLP训练。

## 已完成的 LeWM 与 CEM 实验

| 实验 | 训练 / 权重 | Smoke | 全量评测 |
|---|---|---|---|
| 2D LeWM 世界模型 | epoch10权重已验收 | **通过**：独立重载、真实数据encode/predict/loss有限 | 世界模型本身不单列策略SR；对应CEM结果见下一行 |
| 2D LeWM＋CEM | 使用上述epoch10权重 | **通过**：最小规划/环境接口 | **完成**：task1–5各50条，共250条；**0/250＝0%** |
| DexArt 3D LeWM 世界模型 | epoch10权重已验收 | **通过**：独立重载、真实数据encode/predict/loss有限 | 世界模型本身不单列策略SR；对应CEM结果见下一行 |
| DexArt 3D LeWM＋CEM | 使用上述epoch10权重 | **通过**：最小规划/环境接口 | **完成**：task1–5各50条，共250条；**0/250＝0%** |

CEM使用历史custom fixed5数据上预训练的世界模型。每episode最多200个实际环境步，每次规划25步后重新规划。官方state GCBC与后续LeWM BC使用官方Cube下游轨迹，两种结果的数据和方法范围分别标注。官方state的8.8%对应单训练seed，不是论文8seed平均值。

## 接下来执行什么

1. **新版2D smoke**：把审计通过的RGB分片接入新版单帧loader/MLP，验证真实train/val输入、FP32/BF16、非零梯度、冻结状态、完整恢复和短环境动作接口。
2. **2D正式训练**：smoke通过后训练10个完整epoch；按独立验证集选择best，保留last，并独立严格重载两者。
3. **2D全量SR**：先做最小评测接口检查，再做task1–5各50条、每条最多200步，与已完成的state基线核对相同物理初态和goal后比较。
4. **3D完整流程**：确认全量PCD存储预算，完成与同源RGB配对的PCD派生；独立做3D smoke、10epoch训练、best/last重载及250条SR。2D的精度结论不代替3D检查。
5. **后续对照与延长训练**：汇总2D/3D首10epoch结果，决定是否延长；随后按各自数据与输入契约推进下表方法。

| 后续方法 | 当前smoke状态 | 正式训练 / 最终SR |
|---|---|---|
| RGB/DINO → GCBC | 历史接线smoke通过；**官方Cube数据上的smoke未做** | **未做**；预算在对应smoke后锁定 |
| RGB/DINO → GCIQL | 历史更新/重载/控制接口诊断已做；**官方Cube数据上的smoke未做** | **未做**；预算在对应smoke后锁定 |
| 官方state → GCIQL对照 | 历史接口诊断已做；**官方Cube数据上的smoke未做** | **未做**；预算待定 |

Visual Cube保留为后续独立数据对照。GCIQL的RGB/DINO方法仍只使用对应RGB与goal pixels输入，PCD方法需作为独立扩展记录。

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

本分支收最终接口、当前配置与必要正确性测试。worker 候选测速、失败重试脚本、过程日志、中间快照、数据和权重不上传。原世界模型目录保持兼容；发布整理在独立 checkout 完成，运行中的服务器工作树未改写。
