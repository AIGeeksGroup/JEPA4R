# CortexBench Adroit：LeWM 与视觉 baseline 结果

本页汇总 **Reorient-Pen（Pen）与 Relocate** 的官方论文参照、我们的BC和单独记录的片段CEM。2026-10-26新增Pen的四条RGB baseline；既有LeWM、CEM和RH56记录保留各自版本与协议。

## Pen：四条 RGB baseline（2026-10-26）

四条均复用原BC学习循环：**100条官方示范／每轮25条在线轨迹／100epochs／seeds100、200、300**，评测索引 `5、10、…、95、99` 共20轮。下表每seed为 **Best-of-20／Final（epoch99）**，均值的±为三seed样本标准差（`ddof=1`，单位百分点）。本节与下面历史LeWM表沿用的总体标准差（`ddof=0`）分别标注。

| 方法／编码器 | 表征来源 | Seed100 best／final | Seed200 best／final | Seed300 best／final | Best均值±样本SD | Final均值±样本SD |
|---|---|---:|---:|---:|---:|---:|
| PLDM／ViT-Tiny/14 | Pen任务内原PLDM，10epochs | 72／52 | 76／76 | 64／64 | **70.67±6.11** | **64.00±12.00** |
| LeJEPA／ViT-Tiny/14＋512D head | Pen任务内仓库四视图配方，10epochs | 68／56 | 68／52 | 64／64 | **66.67±2.31** | **57.33±6.11** |
| V-JEPA 2.1／ViT-B/16，384 | Meta发布的distilled ViT-G权重，冻结 | 64／52 | 64／44 | 72／48 | **66.67±4.62** | **48.00±4.00** |
| VISReg／ViT-B/16，224 | 作者提供的ImageNet-1K权重，冻结 | 52／8 | 52／24 | 52／44 | **52.00±0.00** | **25.33±18.04** |

全部12组BC均有100轮有限loss、20次完整在线评测及实际退出0；保存的reward-selected policy通过CPU重载检查。Best-of-20与Final来自评测曲线，不能直接赋给按reward选择的保存policy。`train/*`日志复用评测轨迹，不代表独立的训练初态评测。

PLDM／LeJEPA保留各自原方法；VISReg／V-JEPA 2.1直接使用发布权重。本节只测Pen，未给这四条新增Relocate成绩或双任务Adroit均值。模型容量、预训练数据、原生分辨率及训练预算不同，三seed结果仅作描述性比较。

复现使用现有 `embedding=pldm_pen|lejepa_pen|visreg_vitb|vjepa21_vitb` 开关；特征宽度保留192／512／768／768，三帧FLARE加24D proprio后为600／1560／2328／2328D，未做PCA或统一维度。数据、权重、外部运行时和许可证继续外置。

[最薄入口、权重来源与复现命令](docs/experiments/adroit_pen_baselines_20261026.md) · [逐seed结果、协议与SHA证据](docs/results/adroit_pen_baselines_20261026.json)

## Adroit BC 参考矩阵

SR与均值单位为%。[Table 8](https://arxiv.org/html/2303.18240v2#A1.T8)为原论文公开结果参照，本项目未正式提交官方榜。官方任务列取Table 8，官方均值沿用Table 2／4；Ours为三seed Best-of-20，均值按两任务算术平均。输入列为视觉模态，BC另用proprio。

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
| MAE Ego4D+M (ViT-B) | RGB | 官方论文 | 77.3 | 25.3 | 51.3 |
| MAE Ego4D (ViT-L) | RGB | 官方论文 | 76.0 | 24.0 | 50.0 |
| MAE Ego4D+N (ViT-B) | RGB | 官方论文 | 77.3 | 22.7 | 50.0 |
| MAE Ego4D (ViT-B) | RGB | 官方论文 | 74.7 | 22.7 | 48.7 |
| MAE Ego4D+MN (ViT-B) | RGB | 官方论文 | 73.3 | 24.0 | 48.7 |
| MVP (ViT-B) | RGB | 官方论文 | 69.3 | 26.7 | 48.0 |
| CLIP (ViT-B) | RGB | 官方论文 | 72.0 | 22.7 | 47.3 |
| Random Fine-tuned (ViT-B) | RGB | 官方论文 | 73.3 | 14.7 | 44.0 |
| **LeWM：RGB（Pen最新／Relocate v6）** | RGB | 本项目当前 | **37.33±4.99** | **8.00±0.00** | **22.67** |
| Random Frozen (ViT-L) | RGB | 官方论文 | 5.3 | 0.0 | 2.7 |
| Random Frozen (ViT-B) | RGB | 官方论文 | 4.0 | 0.0 | 2.0 |

† PCD／目标信息接口为扩展，不参与官方RGB排名。Ours各任务使用独立WM权重，WM预训练还使用约10,000条任务内轨迹，100demos仅指BC动作头预算；该均值是描述性汇总，不能视为同一通用encoder的成绩或完全匹配预算的对照。

新版RGB的Relocate **8.00%** 沿用原v6 checkpoint的BC结果，本轮没有新增RGB BC；它与下方86%片段CEM使用不同策略和成功定义。官方Table 8未给逐任务std，Ours的±为三BCseed总体std（ddof=0）。

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

WM编码器输入RGB或1200×6 XYZRGB，原生状态只用于数据、恢复和成功检查。最新Pen两路WM完成10epochs；Relocate复用RGB v6，新PCD完成10epochs／127,260steps并严格重载通过。

- [Pen配置、依赖和复现入口](docs/experiments/adroit_pen_paired_20261006.md)
- [Relocate重放、WM、BC与CEM入口](docs/experiments/adroit_relocate_replay_20261007.md)
- [完整历史：旧PCD合成64点、随机对照、旧协议与后续论文参照](docs/experiments/adroit_history.md)

MuJoCo200／license、外部运行时、数据与checkpoint需另行提供。合并后未重跑完整训练或其他任务GPU回归；当前PCD接口缺匹配随机表征对照，不作预训练、模态优势或sim2real结论。

## 可选：原生 RH56 Relocate

[RH56 数据／接口与显式开关](docs/experiments/rh56_native_20261025.md)接入原生 RM75-B/RH56 的56D专家观测、13D proprio和12D动作。通过独立 `lewm_pcd/scripts/run_rh56.py` 选择阶段及 `--route rgb|pcd`；原Adroit入口和共享学习核心保持不变，数据／专家资产／权重作为外部输入。

[2026-10-25历史生产结果及版本绑定](docs/results/rh56_relocate_20261025.json)单独记录RH56本体，不并入上方官方Adroit矩阵。该记录不是本次集成提交的完整重训验收；RGB uint8/GPU开关默认关闭，原完整模型数值等价失败记录保留。
