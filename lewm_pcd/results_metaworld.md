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

## button-press-topdown（2026-09-27）

| | 规划成功率 | CortexBench button-press |
|---|---|---|
| LeWM 3D，全量 | **87.5**（90 / 84 / 86 / 90） | **98.7**（100 / 100 / 96） |
| LeWM 2D，全量 | 57.5（60 / 58 / 54 / 58） | 29.3（32 / 32 / 24） |
| 随机策略 | 32.0（同一批起点） | — |

同机复现的基线（CortexBench button-press）：VC-1 ViT-B 74.7（论文 82.7）。

- **数据**：全量 1 万条 × 200 步。主干循环做"按下按钮 → 去空中路点"。按钮的回弹弹簧太弱，一回合内按下后不会弹起，所以每回合只有第一次是真按下。有按下的回合：主干 99.9%，分叉 81.5%，随机 0%。点云前景包含按钮盒，它是每回合重新摆放的固定件。
- **2D 偏弱**：CortexBench 上行为克隆的训练损失和 3D、VC-1 在同一量级（能拟合示范），但评测成功率一直在 20% 上下，20 次评测中最高 32。规划成功率也明显低于 3D（57.5 对 87.5）。原因是数据覆盖：官方示范约 87% 的帧是手在按钮上方按着（手臂挡住大半个相机），我们的数据里手在按钮上方的行只有 29%。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 评测日志：`logs/eval/mw_button_{pix,pcd}_e10_s*.log`、`logs/eval/mw_button_random_s*.log`。
  - 模型：`checkpoints/mw_button_{pix,pcd_dexart}/`。

## drawer-open（2026-09-28）

| | 规划成功率 | CortexBench drawer-open |
|---|---|---|
| LeWM 3D，全量 | **87.5**（86 / 88 / 92 / 84） | **100.0**（100 / 100 / 100） |
| LeWM 2D，全量 | 71.5（74 / 68 / 70 / 74） | 52.0（60 / 56 / 40） |
| 随机策略 | 34.0（同一批起点） | — |

同机复现的基线（CortexBench drawer-open）：VC-1 ViT-B 100.0（论文 100.0）。

- **数据**：全量 1 万条 × 200 步。主干循环做"把抽屉拉开或推回到随机开度（0–16 cm）→ 去空中路点"。拉开超过 5 cm 的回合：主干 98.6%，分叉 89.6%，随机 0%。点云前景包含抽屉柜，它是每回合重新摆放的固定件。
- **2D 偏弱**：原因与 button-press 相同，是数据覆盖。官方示范约 83% 的帧是"抽屉全开、手握着把手"，我们的数据里这类状态只有 3.5%。
- **点云**：训练数据用的是点云修复（分割改用单采样渲染）之前的代码，3D 规划评测用修复后的代码现场渲染。drawer 的假点比例为 0，两者在同一状态下几何一致（新点到最近旧点的距离中位数 1.9 mm）。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 评测日志：`logs/eval/mw_drawer_{pix,pcd}_e10_s*.log`、`logs/eval/mw_drawer_random_s*.log`。
  - 模型：`checkpoints/mw_drawer_{pix,pcd_dexart}/`。

## hammer（2026-09-29）

| | 规划成功率 | CortexBench hammer |
|---|---|---|
| LeWM 3D，全量 | **91.0**（88 / 98 / 88 / 90） | **97.3**（96 / 100 / 96） |
| LeWM 2D，全量 | 87.5（84 / 96 / 84 / 86） | 89.3（92 / 92 / 84） |
| 随机策略 | 48.5（同一批起点） | — |

同机复现的基线（CortexBench hammer）：VC-1 ViT-B 100.0（论文 98.7）。

- **数据**：全量 1 万条 × 200 步。主干先抓起锤子，之后在三种目标之间串联：把钉子推到随机深度并顶住停 20–70 步、拿着锤子去空中路点、放回桌上再抓。主干回合 94% 抓起过锤子，89% 推动过钉子，63% 结束时钉子已推到底。约 3% 的回合里，回撤时锤子的羊角把钉子带出来一段（中位数 1 cm）。钉子箱每回合不动，是背景；锤子和钉子在点云前景里。
- **2D 正常**：按 button-press 的教训设计了主干，"拿着锤子站在击打线上"占主干时间的 39%（官方示范约 85%）。
- **随机策略偏高**：主干里有大量举着锤子顶住钉子不动的停留段，这些起点和 25 步后的目标几乎相同。
- **点云**：第一个用修复后点云代码（分割改用单采样渲染）采集的任务，训练和评测一致。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 评测日志：`logs/eval/mw_hammer_{pix,pcd}_e10_s*.log`、`logs/eval/mw_hammer_random_s*.log`。
  - 模型：`checkpoints/mw_hammer_{pix,pcd_dexart}/`。

## bin-picking，RM75 机械臂（2026-10-01）

把 Sawyer 换成实验室的 RM75-B 机械臂 + EG2-4C2 夹爪，其余按 bin-picking 一节的做法。环境和采集策略在 `envs/metaworld/rm75/`，配置是 `mw_binpick_rm75*`；流程用 `scripts/run_mw_pipeline.sh`，参数 `TASK=mw_binpick_rm75 CB_TASK=rm75-bin-picking`。

| | 规划成功率 | CortexBench bin-picking（RM75） |
|---|---|---|
| LeWM 3D，全量 | **80.5**（82 / 82 / 76 / 82） | **88.0**（88 / 84 / 92） |
| LeWM 2D，全量 | 75.5（78 / 78 / 68 / 78） | 41.3（40 / 48 / 36） |
| 随机策略 | 44.5（38 / 52 / 42 / 46，同一批起点） | — |

同为全量的 Sawyer bin-picking：规划 3D 77.5、2D 83.5、随机 44.0；CortexBench 3D 100.0、2D 77.3。

- **机器人**：RM75-B 七轴机械臂 + EG2-4C2 二指夹爪，模型与 OGBench cube 的 RM75 版相同，装在桌子 +x 一侧 20 cm 高的底座上。
  - 动作接口不变：4 维动作，每步把手的目标点移动 a[:3] / 100。底层换成逆运动学 + 关节伺服，关节限速，每步新设的目标点离手不超过 3 cm。
  - 观测仍是 metaworld 的 39 维。夹爪读数是两指间距 / 66.2 mm。
  - 夹持力约 20 N（Sawyer 约 400 N），所以采集策略和专家的抓取都按 RM75 改过，见 `envs/metaworld/rm75/policy.py`、`expert.py`。
- **数据**：全量 1 万条 × 200 步，三类回合实际占 50.3%、30.3%、19.5%。主干回合 97.5% 抓起过方块。
  - 第 4 个关节有 80 行（7 个回合）超出模型的关节范围（±2.356 rad），最多超出 0.25°。MuJoCo 的关节限位是软约束，会有这种小幅超出。
  - 有 76 帧（6 个回合）手和方块都不在相机视野里，点云为空。
- **CortexBench 示范**：Sawyer 的官方示范不能用在 RM75 上。用改成 RM75 版的 metaworld 专家脚本重新录了 25 条，格式和种子规则与官方相同（`bench/cortexbench/make_rm75_demos.py`）。
  - 专家在 metaworld 自带的布局上成功 1000/1000，Sawyer 用原脚本也是 1000/1000；原脚本的规则直接用在 RM75 上只有 49.7%。
  - 在示范和评测实际用的布局上，专家成功 159/160。
  - 机器人和示范都与 Sawyer 不同，所以这一列不能和 Sawyer 的 CortexBench 分数直接比，也不放进下面的五任务对比表。
- **2D 在 CortexBench 上明显偏低**（41.3，Sawyer 77.3），原因还没查明。代码检查没有发现采集、示范、评测与 Sawyer 的做法不一致的地方。
- **迁移**：这批数据、模型和分数是代码迁入仓库之前，在 TMP 里以 `rm75_binpick_full*` 的名字跑出来的，迁移时改成了现在的名字；日志和模型目录里的 `config.yaml` 保留原名。迁入后的代码与原代码核对过：
  - 同一种子采集的数据逐位相同；2D 图像有个别像素差 1 个灰度，原代码自己重跑两次也有，是渲染噪声。
  - 25 条示范能重新生成：3D 逐位相同，2D 只有同样的渲染噪声。
  - 种子 42 的规划评测结果相同：2D 78、3D 82、随机 38。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 采集日志：`logs/collect_mw_binpick_rm75.log`；流程日志：`logs/mw_binpick_rm75_pipeline.log`。
  - 评测日志：`logs/eval/mw_binpick_rm75_{pix,pcd}_e10_s*.log`、`logs/eval/mw_binpick_rm75_random_s*.log`。
  - CortexBench 每次运行的记录：`logs/cortexbench/runs/lewm_mw_binpick_rm75_{pix,pcd}/`；示范：`datasets/cortexbench/metaworld-rm75-expert{,-pcd}/`。
  - 模型：`checkpoints/mw_binpick_rm75_{pix,pcd_dexart}/`。

## button-press-topdown，RM75 机械臂（2026-10-03）

把 Sawyer 换成 RM75-B + EG2-4C2，其余按 button-press-topdown 一节的做法。环境是 `envs/metaworld/rm75/` 的 `RM75ButtonPressEnv`，配置是 `mw_button_rm75*`；流程用 `scripts/run_mw_pipeline.sh`，参数 `TASK=mw_button_rm75 CB_TASK=rm75-button-press-topdown`。

| | 规划成功率 | CortexBench button-press（RM75） |
|---|---|---|
| LeWM 3D，全量 | **92.5**（88 / 94 / 90 / 98） | **100.0**（100 / 100 / 100） |
| LeWM 2D，全量 | 41.0（42 / 46 / 46 / 30） | 49.3（56 / 36 / 56） |
| 随机策略 | 32.0（32 / 30 / 34 / 32，同一批起点） | — |

同为全量的 Sawyer button-press：规划 3D 87.5、2D 57.5、随机 32.0；CortexBench 3D 98.7、2D 29.3。

- **相对 Sawyer 版改了什么**：只有和机器人有关的两处。
  - 按钮的滑动关节在 RM75 模型里不在 `qpos[9]`（RM75 的关节排在最后），按它自己的地址读。
  - 评测的 `set_state` 恢复 RM75 控制器的状态（手的目标点、关节指令），按钮盒照旧恢复。
  - 采集策略（`ButtonThreePartPolicy`）和 CortexBench 的专家脚本（metaworld 的 `SawyerButtonPressTopdownV3Policy`）都用原版，没改。
- **数据**：全量 1 万条 × 200 步，三类回合实际占 51.0%、30.2%、18.8%。
  - 有按下的回合：主干 98.7%，分叉 82.2%，随机 0%（Sawyer：99.9%、81.5%、0%）。
  - 关节没有超出模型的范围，点云没有空帧。
- **CortexBench 示范**：metaworld 原版按钮脚本在 RM75 上录的 25 条，种子 12345–12369，没有跳过。
  - 专家成功率：示范和评测实际用的布局上 984/1000，metaworld 自带的布局上 989/1000（Sawyer 1000/1000）。
  - 失败局结束时，手都偏离按钮中心 17–21 mm，按钮压不到成功线的深度。抽查的失败局里，手是在下压过程中横向滑开的；滑开的原因还没查明（`bench/cortexbench/README.md`）。采集时主干回合按下的比例比 Sawyer 低 1.2 个百分点，应该也是同一个原因。
- **与 Sawyer 比较**：
  - 3D 两项都持平或略高：规划 92.5 对 87.5，CortexBench 100 对 98.7。
  - 2D 规划低于 Sawyer（41.0 对 57.5），CortexBench 反而更高（49.3 对 29.3）。
  - 机器人和示范都不同，所以 CortexBench 这一列不能和 Sawyer 的直接比。
- **原始记录**（在 `$STABLEWM_HOME` 下）：
  - 采集日志：`logs/collect_mw_button_rm75.log`；流程日志：`logs/mw_button_rm75_pipeline.log`；示范：`logs/cortexbench/make_rm75_demos_button-press-topdown.log`。
  - 评测日志：`logs/eval/mw_button_rm75_{pix,pcd}_e10_s*.log`、`logs/eval/mw_button_rm75_random_s*.log`。
  - CortexBench 每次运行的记录：`logs/cortexbench/runs/lewm_mw_button_rm75_{pix,pcd}/`；示范：`datasets/cortexbench/metaworld-rm75-expert{,-pcd}/`。
  - 模型：`checkpoints/mw_button_rm75_{pix,pcd_dexart}/`。

## 与其他工作的对比（CortexBench MetaWorld 五任务）

- "论文"行取自 VC-1 论文（arXiv 2303.18240）附录 Table 8 的 MetaWorld 各列。
- "本机复现"行由 `bench/cortexbench` 在本机用各自的官方权重跑出，3 个种子。
- 按五个任务的均值排序，即 VC-1 论文中 MetaWorld 一项的分数。

| 模型 | 来源 | assembly | bin-picking | button-press | drawer-open | hammer | 五任务均值 |
|---|---|---|---|---|---|---|---|
| **LeWM 3D（我们，点云）** | 本机 | 96.0 | 100.0 | 98.7 | 100.0 | 97.3 | **98.4** |
| R3M (RN-50) | 论文 | 97.3 | 93.3 | 89.3 | 100.0 | 100.0 | 96.0 |
| MAE Ego4D (ViT-L) | 论文 | 98.0 | 84.0 | 84.0 | 100.0 | 98.7 | 92.9 |
| MVP (ViT-B) | 论文 | 92.0 | 73.3 | 92.0 | 100.0 | 98.7 | 91.2 |
| VIP (RN-50) | 论文 | 93.3 | 76.0 | 88.0 | 100.0 | 93.3 | 90.1 |
| VC-1 ViT-B | 论文 | 88.0 | 78.7 | 82.7 | 100.0 | 98.7 | 89.6 |
| MAE Ego4D+N (ViT-L) | 论文 | 89.3 | 73.3 | 89.3 | 100.0 | 93.3 | 89.0 |
| VC-1 ViT-B | 本机复现 | 89.3 | 80.0 | 74.7 | 100.0 | 100.0 | 88.8 |
| VC-1 ViT-L | 论文 | 88.0 | 84.0 | 80.0 | 100.0 | 92.0 | 88.8 |
| MAE Ego4D+M (ViT-L) | 论文 | 89.3 | 73.3 | 84.0 | 100.0 | 94.7 | 88.3 |
| MVP (ViT-L) | 论文 | 89.3 | 78.7 | 70.7 | 100.0 | 98.7 | 87.5 |
| MAE Ego4D+MN (ViT-L) | 论文 | 93.3 | 70.7 | 74.7 | 100.0 | 94.7 | 86.7 |
| MAE Ego4D+N (ViT-B) | 论文 | 86.7 | 76.0 | 73.3 | 100.0 | 96.0 | 86.4 |
| MAE Ego4D (ViT-B) | 论文 | 81.3 | 76.0 | 80.0 | 100.0 | 93.3 | 86.1 |
| MAE Ego4D+MN (ViT-B) | 论文 | 82.7 | 74.7 | 77.3 | 100.0 | 92.0 | 85.3 |
| MAE Ego4D+M (ViT-B) | 论文 | 76.0 | 58.7 | 84.0 | 100.0 | 98.7 | 83.5 |
| CLIP (ViT-B) | 论文 | 70.7 | 68.0 | 48.0 | 100.0 | 90.7 | 75.5 |
| **LeWM 2D（我们，图像）** | 本机 | 72.0 | 77.3 | 29.3 | 52.0 | 89.3 | **64.0** |
| Random (ViT-B) 端到端微调 | 论文 | 61.3 | 34.7 | 20.0 | 40.0 | 93.3 | 49.9 |
| Random (ViT-B) 冻结 | 论文 | 0.0 | 0.0 | 2.7 | 0.0 | 0.0 | 0.5 |
| Random (ViT-L) 冻结 | 论文 | 0.0 | 0.0 | 0.0 | 0.0 | 2.7 | 0.5 |

只在本机复现了 bin-picking 的基线：R3M 93.3、VC-1 ViT-L 81.3、CLIP 74.7、VIP 69.3、随机初始化 ViT-B 0.0。LeWM 小批量（仅 bin-picking）：3D 98.7、2D 78.7。

**可比性。**
- 其他工作都是通用预训练的视觉表征，没有见过 MetaWorld。LeWM 用 bin-picking 同域的游玩数据训练过（没用 CortexBench 的示范），有同域优势。
- 3D 用了模拟器分割来去掉背景，相当于多了一份前景信息。
- 本机复现与论文在 bin-picking 上相差 0 到 6.7 分：R3M 相同，VC-1 L −2.7，VC-1 B +1.3，CLIP +6.7，VIP −6.7。
