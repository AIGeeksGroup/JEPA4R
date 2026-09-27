# MetaWorld（CortexBench 的 MetaWorld 任务）结果

每个任务一节，按做完的顺序追加。

## 做法

- **数据**：三部分采样，主干、分叉、随机三类回合各占 50%、30%、20%，每回合 200 步。全量是 1 万条（201 万帧），与 OGBench cube 的全量规模相同。2D 数据集与 3D 数据集来自同一批 rollout，逐行配对。采集用 `scripts/data/run_mw_collection.sh`，配置在 `scripts/data/config/mw_<任务>.yaml`，环境和采集策略在 `envs/metaworld/`。
- **模型**：2D 是官方 LeWM（ViT-tiny），3D 是 DexArt PointNet。两者都训练 10 个 epoch，取第 10 个 epoch 的权重。流程见 `scripts/run_mw_pipeline.sh`。
- **规划成功率**：LeWM 论文协议，在留出的测试集（150 条）上评测。
  - 起点取测试集中的一行，目标取同一回合 25 步之后的那一行；执行预算 50 步；CEM 用 300 个样本、取前 30、迭代 10 次。
  - 手到目标位置 4 cm 以内、物体到目标位置 3 cm 以内，算成功。
  - 种子 42–45，每个种子 50 个起点。"随机策略"一行用的是同一批起点。
  - 配置见 `config/eval/mw_<任务>_{pix,pcd}.yaml`。
- **CortexBench**：官方模仿学习协议。编码器冻结，用 25 条官方示范训练行为克隆；每个种子取 20 次评测中的最高成功率，报 3 个种子的均值。3D 编码器的输入是同一台相机的点云，做法见 `bench/cortexbench/README.md`。
  - 图像输入的评测结果每次重跑会有波动，因为渲染不是逐位确定的；点云输入没有这个问题。

## bin-picking（2026-09-26）

| | 规划成功率 | CortexBench bin-picking |
|---|---|---|
| LeWM 3D，全量 | 77.5（76 / 82 / 74 / 78） | **100.0**（100 / 100 / 100） |
| LeWM 2D，全量 | **83.5**（86 / 82 / 84 / 82） | 77.3（76 / 68 / 88） |
| LeWM 3D，小批量（600 条 × 250 步，15 万帧） | 83.5 | 98.7 |
| LeWM 2D，小批量 | 68.0 | 78.7 |
| 随机策略 | 44.0（全量测试集）；小批量测试集上是 48.0 | — |

同机复现的基线（CortexBench bin-picking）：R3M 93.3，VC-1 ViT-L 81.3，VC-1 ViT-B 80.0，CLIP 74.7，VIP 69.3，随机初始化 ViT-B 0.0。

- **全量与小批量的比较。** 小批量有自己的测试集（120 条），与全量用的起点不同。
  - 3D：两项都没有明显变化。
  - 2D：CortexBench 不变，规划成功率从 68.0 升到 83.5。在同一批起点上，随机策略只有 44.0。
- **2D 与 3D 的比较。**
  - 小批量时，3D 在两项上都强于或持平 2D。
  - 全量后，规划上 2D 高于 3D（83.5 对 77.5），CortexBench 上 3D 仍明显更高（100 对 77.3）。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 规划评测日志：`logs/eval/mw_binpick_{pix,pcd}_e10_s*.log`、`logs/eval/mw_binpick_random_s*.log`。
  - CortexBench 汇总表：`logs/cortexbench/results/scores.md`；每次运行的记录：`logs/cortexbench/runs/`。
  - 模型：`checkpoints/mw_binpick_{pix,pcd_dexart}/`。

## assembly（2026-09-27）

| | 规划成功率 | CortexBench assembly |
|---|---|---|
| LeWM 3D，全量 | **94.5**（98 / 94 / 92 / 94） | **96.0**（96 / 100 / 92） |
| LeWM 2D，全量 | 84.5（90 / 82 / 82 / 84） | 72.0（72 / 72 / 72） |
| 随机策略 | 27.5（同一批起点） | — |

同机复现的基线（CortexBench assembly）：VC-1 ViT-B 89.3（论文 88.0）。

- **数据**：全量 1 万条 × 200 步。主干循环做"把螺母套上柱子 → 取下 → 放回桌面"，每条回合平均放置 0.99 次，其中套上柱子 0.71 次。点云前景包含柱子，它是每回合重新摆放的固定件。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 评测日志：`logs/eval/mw_assembly_{pix,pcd}_e10_s*.log`、`logs/eval/mw_assembly_random_s*.log`。
  - 模型：`checkpoints/mw_assembly_{pix,pcd_dexart}/`。

## 与其他工作的对比（CortexBench MetaWorld，截至 assembly）

- "论文"行取自 VC-1 论文（arXiv 2303.18240）附录 Table 8 的 MetaWorld 各列。
- "本机复现"行由 `bench/cortexbench` 在本机用各自的官方权重跑出，3 个种子。
- 按两个任务的均值排序。LeWM 目前只在训练过的任务上有分数。

| 模型 | 来源 | assembly | bin-picking | 两任务均值 |
|---|---|---|---|---|
| **LeWM 3D（我们，点云）** | 本机 | 96.0 | 100.0 | **98.0** |
| R3M (RN-50) | 论文 | 97.3 | 93.3 | 95.3 |
| MAE Ego4D (ViT-L) | 论文 | 98.0 | 84.0 | 91.0 |
| VC-1 ViT-L | 论文 | 88.0 | 84.0 | 86.0 |
| VC-1 ViT-B | 本机复现 | 89.3 | 80.0 | 84.7 |
| VIP (RN-50) | 论文 | 93.3 | 76.0 | 84.7 |
| MVP (ViT-L) | 论文 | 89.3 | 78.7 | 84.0 |
| VC-1 ViT-B | 论文 | 88.0 | 78.7 | 83.3 |
| MVP (ViT-B) | 论文 | 92.0 | 73.3 | 82.7 |
| MAE Ego4D+MN (ViT-L) | 论文 | 93.3 | 70.7 | 82.0 |
| MAE Ego4D+N (ViT-B) | 论文 | 86.7 | 76.0 | 81.3 |
| MAE Ego4D+M (ViT-L) | 论文 | 89.3 | 73.3 | 81.3 |
| MAE Ego4D+N (ViT-L) | 论文 | 89.3 | 73.3 | 81.3 |
| MAE Ego4D+MN (ViT-B) | 论文 | 82.7 | 74.7 | 78.7 |
| MAE Ego4D (ViT-B) | 论文 | 81.3 | 76.0 | 78.7 |
| **LeWM 2D（我们，图像）** | 本机 | 72.0 | 77.3 | **74.7** |
| CLIP (ViT-B) | 论文 | 70.7 | 68.0 | 69.3 |
| MAE Ego4D+M (ViT-B) | 论文 | 76.0 | 58.7 | 67.3 |
| Random (ViT-B) 端到端微调 | 论文 | 61.3 | 34.7 | 48.0 |
| Random (ViT-B) 冻结 | 论文 | 0.0 | 0.0 | 0.0 |
| Random (ViT-L) 冻结 | 论文 | 0.0 | 0.0 | 0.0 |

只在本机复现了 bin-picking 的基线：R3M 93.3、VC-1 ViT-L 81.3、CLIP 74.7、VIP 69.3、随机初始化 ViT-B 0.0。LeWM 小批量（仅 bin-picking）：3D 98.7、2D 78.7。

**可比性。**
- 其他工作都是通用预训练的视觉表征，没有见过 MetaWorld。LeWM 用 bin-picking 同域的游玩数据训练过（没用 CortexBench 的示范），有同域优势。
- 3D 用了模拟器分割来去掉背景，相当于多了一份前景信息。
- 本机复现与论文在 bin-picking 上相差 0 到 6.7 分：R3M 相同，VC-1 L −2.7，VC-1 B +1.3，CLIP +6.7，VIP −6.7。
