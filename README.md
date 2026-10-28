# CortexBench Adroit：LeWM 与视觉 baseline 结果

本页汇总 **Reorient-Pen（Pen）与 Relocate** 的官方论文参照、我们的BC和单独记录的片段CEM。2026-10-26新增Pen的四条RGB baseline，2026-10-28补齐原生Relocate对应结果并合入下方主矩阵；RH56 Relocate的2D／3D结果与PCD权重入口单独保留。各实验保留各自版本与协议。

## Adroit BC 参考矩阵

SR与均值单位为%。[Table 8](https://arxiv.org/html/2303.18240v2#A1.T8)为原论文公开结果参照，本项目未正式提交官方榜。官方任务列取Table 8，官方均值沿用Table 2／4。新增四条RGB baseline在单元格中分别列 **Best／Final** 的三seed均值±样本SD（`ddof=1`）；Adroit列分别对两个任务的Best均值、Final均值取算术平均。既有LeWM两行仍为Best-of-20及总体std（`ddof=0`），保留历史口径。输入列为视觉模态，BC另用proprio。

| 模型／版本 | 输入 | 来源 | Pen SR | Relocate SR | Adroit 均值 |
|---|---|---|---:|---:|---:|
| **LeWM：可见表面PCD** | PCD† | 本项目当前 | **81.33±4.99** | **88.00±5.66** | **84.67†** |
| R3M (RN-50) | RGB | 官方论文 | 81.3 | 65.3 | 73.3 |
| VC-1 (ViT-L; Ego4D+MNI) | RGB | 官方论文 | 80.0 | 38.7 | 59.3 |
| MAE Ego4D+MNI (ViT-B) | RGB | 官方论文 | 76.0 | 32.0 | 54.0 |
| MAE Ego4D+N (ViT-L) | RGB | 官方论文 | 76.0 | 32.0 | 54.0 |
| VIP (RN-50) | RGB | 官方论文 | 81.3 | 26.7 | 54.0 |
| MVP (ViT-L) | RGB | 官方论文 | 76.0 | 30.7 | 53.3 |
| MAE Ego4D+MN (ViT-L) | RGB | 官方论文 | 78.7 | 26.7 | 52.7 |
| MAE Ego4D+M (ViT-L) | RGB | 官方论文 | 74.7 | 29.3 | 52.0 |
| V-JEPA 2.1／ViT-B/16，384 | RGB | Meta发布权重，冻结 | Best **66.67±4.62**<br>Final **48.00±4.00** | Best **36.00±4.00**<br>Final **18.67±2.31** | Best **51.33**<br>Final **33.33** |
| MAE Ego4D+M (ViT-B) | RGB | 官方论文 | 77.3 | 25.3 | 51.3 |
| MAE Ego4D (ViT-L) | RGB | 官方论文 | 76.0 | 24.0 | 50.0 |
| MAE Ego4D+N (ViT-B) | RGB | 官方论文 | 77.3 | 22.7 | 50.0 |
| MAE Ego4D (ViT-B) | RGB | 官方论文 | 74.7 | 22.7 | 48.7 |
| MAE Ego4D+MN (ViT-B) | RGB | 官方论文 | 73.3 | 24.0 | 48.7 |
| MVP (ViT-B) | RGB | 官方论文 | 69.3 | 26.7 | 48.0 |
| CLIP (ViT-B) | RGB | 官方论文 | 72.0 | 22.7 | 47.3 |
| VISReg／ViT-B/16，224 | RGB | 作者发布权重，冻结 | Best **52.00±0.00**<br>Final **25.33±18.04** | Best **40.00±10.58**<br>Final **30.67±8.33** | Best **46.00**<br>Final **28.00** |
| Random Fine-tuned (ViT-B) | RGB | 官方论文 | 73.3 | 14.7 | 44.0 |
| LeJEPA／ViT-Tiny/14＋512D head | RGB | 各任务原生10轮预训练，冻结 | Best **66.67±2.31**<br>Final **57.33±6.11** | Best **16.00±0.00**<br>Final **6.67±2.31** | Best **41.33**<br>Final **32.00** |
| PLDM／ViT-Tiny/14 | RGB | 各任务原生10轮预训练，冻结 | Best **70.67±6.11**<br>Final **64.00±12.00** | Best **10.67±2.31**<br>Final **1.33±2.31** | Best **40.67**<br>Final **32.67** |
| **LeWM：RGB（Pen最新／Relocate v6）** | RGB | 本项目当前 | **37.33±4.99** | **8.00±0.00** | **22.67** |
| Random Frozen (ViT-L) | RGB | 官方论文 | 5.3 | 0.0 | 2.7 |
| Random Frozen (ViT-B) | RGB | 官方论文 | 4.0 | 0.0 | 2.0 |

† PCD／目标信息接口为扩展，不参与官方RGB排名。LeWM、PLDM、LeJEPA各任务使用独立WM权重，预训练还使用约10,000条任务内轨迹；VISReg／V-JEPA使用各自同一发布encoder、分别训练任务BC head。100demos仅指BC动作头预算。两任务均值为描述性汇总，模型容量、预训练数据及预算不同，不代表官方榜成绩或匹配预算的因果对照；不为两任务均值构造合并轨迹SR或置信区间。

LeWM RGB的Relocate **8.00%** 沿用原v6 checkpoint的BC结果，本轮没有重训LeWM 2D；新增四条RGB baseline单独列于主表。LeWM v6与下方86%片段CEM使用不同策略和成功定义。官方Table 8未给逐任务std；既有LeWM两行及下方历史明细的±保持总体std（`ddof=0`），新增四条baseline使用样本SD（`ddof=1`）。

## Pen：LeWM 2D／3D 与四条 baseline

下表集中展示既有LeWM 2D／3D与2026-10-26新增的四条baseline。BC预算均为 **100条官方示范／每轮25条在线轨迹／100epochs／seeds100、200、300**，评测索引 `5、10、…、95、99` 共20轮。每seed为 **Best-of-20／Final（epoch99）**；本表全部按三seed原值统一计算样本标准差（`ddof=1`，单位百分点），下方历史表保留原总体标准差（`ddof=0`）。

| 方法／编码器 | 表征来源 | Seed100 best／final | Seed200 best／final | Seed300 best／final | Best均值±样本SD | Final均值±样本SD |
|---|---|---:|---:|---:|---:|---:|
| LeWM 2D／RGB | 2026-10-06 Pen epoch10，任务内预训练 | 36／24 | 44／32 | 32／24 | **37.33±6.11** | **26.67±4.62** |
| LeWM 3D／可见表面PCD† | 2026-10-06 Pen epoch10，任务内预训练 | 80／76 | 76／76 | 88／68 | **81.33±6.11** | **73.33±4.62** |
| PLDM／ViT-Tiny/14 | Pen任务内原PLDM，10epochs | 72／52 | 76／76 | 64／64 | **70.67±6.11** | **64.00±12.00** |
| LeJEPA／ViT-Tiny/14＋512D head | Pen任务内仓库四视图配方，10epochs | 68／56 | 68／52 | 64／64 | **66.67±2.31** | **57.33±6.11** |
| V-JEPA 2.1／ViT-B/16，384 | Meta发布的distilled ViT-G权重，冻结 | 64／52 | 64／44 | 72／48 | **66.67±4.62** | **48.00±4.00** |
| VISReg／ViT-B/16，224 | 作者提供的ImageNet-1K权重，冻结 | 52／8 | 52／24 | 52／44 | **52.00±0.00** | **25.33±18.04** |

新增四条baseline的12组BC均有100轮有限loss、20次完整在线评测及实际退出0；保存的reward-selected policy通过CPU重载检查。LeWM两行引用[既有逐seed记录](docs/results/adroit_pen_20261006_paired_bc.json)，其中RGB seed300评分完整，但原保存条件未导出policy。Best-of-20与Final来自评测曲线，不能直接赋给按reward选择的保存policy。`train/*`日志复用评测轨迹，不代表独立的训练初态评测。

† LeWM 3D为PCD模态扩展，输入及目标信息接口与RGB不同；集中展示用于查看已有结果，不将其视为同模态、同信息预算的官方RGB排名。

PLDM／LeJEPA保留各自原方法；VISReg／V-JEPA 2.1直接使用发布权重。本节保留2026-10-26的Pen结果，Relocate逐seed结果见下方，两任务描述汇总见上方主表。模型容量、预训练数据、原生分辨率及训练预算不同，三seed结果仅作描述性比较。

复现使用现有 `embedding=pldm_pen|lejepa_pen|visreg_vitb|vjepa21_vitb` 开关；特征宽度保留192／512／768／768，三帧FLARE加24D proprio后为600／1560／2328／2328D，未做PCA或统一维度。数据、权重、外部运行时和许可证继续外置。

[最薄入口、权重来源与复现命令](docs/experiments/adroit_pen_baselines_20261026.md) · [逐seed结果、协议与SHA证据](docs/results/adroit_pen_baselines_20261026.json)

## Relocate：四条 RGB baseline（2026-10-28）

原生 `relocate-v0` 使用 **100条官方示范／每轮25条在线轨迹／100epochs／seeds100、200、300**；Best-of-20、Final（eval99）和样本SD口径同上。PLDM／LeJEPA在已验收的Relocate RGB数据上各完成原生10轮预训练后冻结；VISReg／V-JEPA 2.1复用发布权重冻结读出。本轮不重训LeWM 2D，不混入RH56、PCD或CEM结果。

| 方法／编码器 | Seed100 best／final | Seed200 best／final | Seed300 best／final | Best均值±样本SD | Final均值±样本SD |
|---|---:|---:|---:|---:|---:|
| PLDM／ViT-Tiny/14 | 12／0 | 8／0 | 12／4 | **10.67±2.31** | **1.33±2.31** |
| LeJEPA／ViT-Tiny/14＋512D head | 16／8 | 16／8 | 16／4 | **16.00±0.00** | **6.67±2.31** |
| VISReg／ViT-B/16，224 | 48／40 | 44／28 | 28／24 | **40.00±10.58** | **30.67±8.33** |
| V-JEPA 2.1／ViT-B/16，384 | 40／16 | 32／20 | 36／20 | **36.00±4.00** | **18.67±2.31** |

12组BC均完成100条有限loss、20轮完整评测并实际退出0，保存的reward-selected policy通过两次CPU重载；曲线Best／Final不自动等于该保存文件的SR。官方100条示范的RGB由保留的官方初态与动作配对重渲，非视觉字段一致；没有声称与已缺失的历史RGB逐像素相同。动作／proprio保留30D，原192／512／768／768D读出对应606／1566／2334／2334D策略输入，未统一宽度或重写BC。

仅在现有薄入口显式增加 `--task adroit_relocate`，BC沿用 `embedding=pldm_relocate|lejepa_relocate|visreg_vitb|vjepa21_vitb`；数据、权重、运行环境和许可证继续外置。[接口与复现条件](docs/experiments/adroit_relocate_baselines_20261028.md) · [逐seed曲线、真实生产版本与SHA](docs/results/adroit_relocate_baselines_20261028.json)

## BC 三 seed 明细

协议为 **100 demos／每轮25条在线轨迹／100 epochs**，评测 `5、10、…、95、99` 共20轮。单元格为 **Best-of-20／Final（epoch99）**；最后两列均为均值±总体std，单位%。[官方协议 Appendix A.2](https://arxiv.org/html/2303.18240v2#A1.SS2)

| 任务／视觉输入 | WM版本与来源 | Seed100 best／final | Seed200 best／final | Seed300 best／final | Best均值±std | Final均值±std |
|---|---|---:|---:|---:|---:|---:|
| Relocate／RGB | [原v6 epoch10](docs/results/adroit_relocate_100demo_20261004.json) | 8／0 | 8／0 | 8／4 | **8.00±0.00** | 1.33±1.89 |
| Pen／RGB | [2026-10-06 epoch10](docs/results/adroit_pen_20261006_paired_bc.json) | 36／24 | 44／32 | 32／24 | **37.33±4.99** | 26.67±3.77 |
| Pen／PCD扩展 | [2026-10-06 epoch10](docs/results/adroit_pen_20261006_paired_bc.json) | 80／76 | 76／76 | 88／68 | **81.33±4.99** | 73.33±3.77 |
| Relocate／PCD扩展 | [2026-10-07 epoch10](docs/results/adroit_relocate_20261007.json) | 84／84 | 96／92 | 84／68 | **88.00±5.66** | 81.33±9.98 |

Relocate RGB v6权重记录SHA：`4ea58e8e…5519b`。Relocate新版PCD的最高reward保存策略SR为84／96／84%；单seed54321的best88%／final68%独立记录，不纳入三seed均值。Pen新RGB seed300有完整评分，但没有保存策略文件。

## CEM 配对50片段（不纳入BC矩阵）

两路使用相同起点：seed42、goal `t+25`、budget50、horizon5／action_block5／receding5、300候选／top30／10次迭代、history1。均从独立test150抽样；Pen涉及40个episode，Relocate44个。

| 任务／epoch10输入 | 全部50片段 | 初始未达标子集 | 初始已达标起点 |
|---|---:|---:|---:|
| Pen／RGB | **34/50（68.00%）** | 11/27（40.74%） | 23/50 |
| Pen／PCD扩展 | **39/50（78.00%）** | 16/27（59.26%） | 23/50 |
| Relocate／RGB v6 | **43/50（86.00%）** | 12/19（63.16%） | 31/50 |
| Relocate／PCD扩展 | **46/50（92.00%）** | 15/19（78.95%） | 31/50 |

Pen目标为未来笔位置／方向，距离<0.075m且dot>0.95，掉落失败；Relocate为未来物体世界位置，距离<0.1m。初始达标起点全部成功；无随机／无动作基线。Relocate完整任务另要求>25帧达标，与本片段判据不同；CEM不纳入BC矩阵，不能据此归因规划收益。

[Pen逐段结果与判据](docs/results/adroit_pen_20261006_fragment_cem.json) · [Relocate逐段结果、checkpoint与协议绑定](docs/results/adroit_relocate_20261007.json)。两路真实子进程退出码0；归一化保留Pen CEM的train-only StandardScaler，与训练Torch的std约定有区别。

## 输入、复现与历史

| 任务 | 原生观测 | 动作／proprio | 三帧BC输入→动作 | WM动作块 |
|---|---:|---:|---:|---:|
| Pen | 45D | 24D／24D | 600D→24D | 120D |
| Relocate | 39D | 30D／30D | 606D→30D | 150D |

WM编码器输入RGB或1200×6 XYZRGB，原生状态只用于数据、恢复和成功检查。此处192D表征输入对应LeWM；新增baseline的各自宽度见上方。LeWM最新Pen两路WM完成10epochs；Relocate的LeWM复用RGB v6，新PCD完成10epochs／127,260steps并严格重载通过。

- [Pen配置、依赖和复现入口](docs/experiments/adroit_pen_paired_20261006.md)
- [Relocate重放、WM、BC与CEM入口](docs/experiments/adroit_relocate_replay_20261007.md)
- [完整历史：旧PCD合成64点、随机对照、旧协议与后续论文参照](docs/experiments/adroit_history.md)

MuJoCo200／license、外部运行时、数据与checkpoint需另行提供。合并后未重跑完整训练或其他任务GPU回归；当前PCD接口缺匹配随机表征对照，不作预训练、模态优势或sim2real结论。

## RH56 Relocate：原生 RM75-B／RH56 的 2D／3D 结果

[RH56 数据／接口与显式开关](docs/experiments/rh56_native_20261025.md)接入原生 RM75-B/RH56 的56D专家观测、13D proprio和12D动作。通过独立 `lewm_pcd/scripts/run_rh56.py` 选择阶段及 `--route rgb|pcd`；原Adroit入口和共享学习核心保持不变，数据／专家资产／权重作为外部输入。

以下为 **2026-10-25完成的自定义RH56仿真结果**，不并入上方官方Adroit本体矩阵。BC使用100条新采RH56示范，仅迁移官方Relocate示范的物体／目标初始位置；RH56本体初态、专家、控制与实际12D动作保持原生。

两路使用各自的WM epoch10冻结encoder＋projector，BC预算为 **100 demos／每轮25条在线轨迹／100 epochs／seeds100、200、300**，评测索引 `5、10、…、95、99` 共20轮。单元格为 **Best-of-20／Final（epoch99）**；均值取三seed算术平均，样本SD按 `ddof=1` 计算、单位为百分点。

| 视觉输入 | Seed100 best／final | Seed200 best／final | Seed300 best／final | Best均值±样本SD | Final均值±样本SD |
|---|---:|---:|---:|---:|---:|
| LeWM 2D／RGB | 4／0 | 8／0 | 4／0 | **5.33±2.31** | **0.00±0.00** |
| LeWM 3D／PCD | 60／60 | 44／24 | 36／20 | **46.67±12.22** | **34.67±22.03** |

两路WM均完成10轮、125,480次优化器更新及13,950个验证batch；六组BC均完成100轮、20次评测、7,900次Adam更新与有限梯度检查，冻结表征未变，策略重载通过，实际退出码为0。完整证据见[原生产结果与版本绑定](docs/results/rh56_relocate_20261025.json)；这些是训练完整性和重载验收，不代表真机效果或本次文档更新重跑了实验。

**PCD上游与下游权重：** 已迁入 [AIGeeksGroup/JEPA4R 的 `rh56-relocate-pcd` 分支](https://huggingface.co/AIGeeksGroup/JEPA4R/tree/rh56-relocate-pcd)，经用户授权公开，`main` 分支保持不变。包含PCD epoch10完整WM、训练checkpoint、三个种子的 `best_policy`（原pickle及可独立加载的safetensors）、配置、动作归一化和接口说明；模型权重未改，首次迁移时153个文件均完成SHA核验，[迁移记录与文件SHA](docs/results/rh56_pcd_hub_migration_20261027.json)保留当时的版本绑定。

**Checkpoint＋训练／推理代码交接：** 同一Hub分支补齐了 `code/rh56_inference.py`、`examples/infer_sequence.py` 和 `code/load_world_model.py`：支持点云→预处理→192D表征→原FLARE＋13D proprio→12D动作均值，以及完整WM的111项state严格重建。补充版本的417个文件已完成SHA核验，原8个模型／策略权重文件未改。5项CPU检查和示例的synthetic self-check／NPZ输入两条CLI路径均实际退出0；这些只验证离线加载与数据流，不是新训练、仿真SR或真机结果。[交接记录](docs/results/rh56_pcd_handoff_20261027.json)绑定补充版本与证据。

原训练入口及依赖源码快照、[训练说明](https://huggingface.co/AIGeeksGroup/JEPA4R/blob/11d2802382e0aa5a66897676d7754d58d8f6c65f/docs/TRAINING.md)和[仿真依赖清单](https://huggingface.co/AIGeeksGroup/JEPA4R/blob/11d2802382e0aa5a66897676d7754d58d8f6c65f/docs/SIMULATION_DEPENDENCIES.md)已随包提供。原生仿真控制源码可供接入参考，训练数据、机器人资产／mesh、专家actor及完整legacy BC运行时仍需另行准备；包内没有一键新数据微调CLI。

保存的 `best_policy` 按**最高平均reward**选择，PCD三个种子对应保存轮次的SR为 **48／40／32%**，主推seed100；这些数值与上表的曲线Best-of-20及Final分别记录。RGB权重不在本次PCD发布包中。

两路保留各自预处理；RGB完整模型CPU／GPU数值等价门槛仍未通过，集成入口的uint8/GPU开关默认关闭。配对抽样未见明显整体表征坍缩，但不能据此保证控制效果，或将两路SR差异归因于模态本身。**真机传感器、坐标标定和控制接口尚未验证**；下载权重可用于后续微调初始化与BC接口测试。
