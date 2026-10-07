# TriFinger（CortexBench 的 TriFinger 任务）结果

CortexBench 的 TriFinger 有两个任务：reach_cube（一根手指去碰方块）和 move_cube（三根手指把方块推到目标位置）。

## 做法

- **环境**：直接用 CortexBench 自己的 TriFinger 环境，即 eai-vc `trifinger_vc` 的 `MoveCubeEnv` / `CubeReachEnv`，设置与它评测时用的 `sim_nn.Task` 相同。它们被封装成 stable-worldmodel 的环境 `swm/TriFinger-v0`（`envs/trifinger/`）。
  - 回合按 CortexBench 示范的设定：初始关节角固定，方块位姿由 `move_cube.sample_goal(-1)` 采样，move 的目标由 `sample_goal(1)` 采样。move 10 步，reach 5 步，每步 0.4 s；reach 里只有第一根手指动。
  - 2D 是平台 camera60 的画面（270×270，双三次缩放到 224×224）。3D 是同一台 camera60 的深度反投影成的点云：手指的活动连杆加方块，1200 点，5 mm 体素，方块占 25%。
  - 两处运行时补丁，都不改变观测：
    - 平台复位时挪方块的那一行补上 pybullet 客户端编号，这样一个进程里可以开多个环境。
    - 每步 0.4 s 内相机触发 4 次，只有最后一次会被观测到，前 3 次不渲染。`scripts/checks/check_tf_frames.py` 逐位验证过，观测到的图像、状态、点云都不变。
- **专家**：CortexBench 示范的专家代码没有公开，按示范里记录的计划复现（`envs/trifinger/expert.py`）。目标点用官方 `cube_utils` 计算，各阶段的时间从示范拟合。
  - 从 125 条 move 示范的起点在官方环境里回放：计划与示范相差 0.13 cm 以内；指尖、方块轨迹分别相差 0.14 cm、0.09 cm（每条示范最大误差的中位数）；最终成功率 91.2，示范是 91.9（`scripts/checks/check_tf_expert.py`）。
- **数据**：三部分采样，与 MetaWorld 相同。
  - 主干（专家 + 最大 3 mm 的高斯噪声）、分叉（专家执行到某一步，之后随机到回合结束）、随机，各占 50%、30%、20%。
  - 随机动作是每轴 ±2 cm 的指尖位移，快要出相机视野时反向。
  - move 和 reach 各占一半回合，回合长度与示范相同。2D、3D 数据集来自同一批 rollout，逐行配对。
- **模型**：2D 是官方 LeWM（ViT-tiny），3D 是 DexArt PointNet。配置在 `config/train/data/tf_cube{,_pcd}.yaml`，frameskip 为 1，因为一行已经是 0.4 s。
- **规划成功率**：LeWM 论文协议，只在 move 上评。
  - 起点取留出测试集（150 条 move 回合，种子 2000000）中的一行，目标取同一回合 5 行之后的那一行；执行预算 10 步；CEM 用 300 个样本、取前 30、迭代 10 次。
  - 三根指尖都在目标行的 4 cm 以内、方块在 3 cm 以内，算成功。
  - 种子 42–45，每个种子 50 个起点；随机策略和零动作用同一批起点。配置在 `config/eval/tf_cube_{pix,pcd}.yaml`。
  - 零动作（`scripts/eval/eval_hold.py`，指尖目标原地不动）是这个协议的地板。目标在 5 行之后，回合里常有停在原地的段落（接近、接触、松开，以及 reach 里不动的两根手指），这时不动就能满足目标。随机策略每步每轴最多动 ±8 cm，会马上离开这些状态，所以它的分数不是地板。
  - 3D 的规划成功率每次重跑会变。原因是评测时点云抽样没有设种子：测试集没有 `seed` 列，评测复位环境时不带种子，`envs/pcd_wrapper.py` 的抽样随机数就用系统熵。种子 42 重跑三次，结果为 48、52、50（下表为 52）。2D 和零动作逐位可复现。
- **CortexBench**：官方模仿学习协议，100 条训练示范。reach 训练 100 个 epoch、评测 21 次，move 训练 1000 个 epoch、评测 41 次；每次评测在 25 条测试示范上取平均。每个种子取所有评测中的最高，报 3 个种子（100 / 200 / 300）的均值，与 VC-1 论文表 8 同一口径。做法见 `bench/cortexbench/README.md` 的 TriFinger 一节。

流程在 `scripts/run_tf_pipeline.sh`：采集 → 训练 → 规划评测 → CortexBench。默认是全量规模：move、reach 各 12 万回合（约 204 万行，与 MetaWorld 的 201 万行相当），训练 10 个 epoch。

## 小规模（2026-10-07）

move、reach 各 2000 回合，共 3.4 万行；测试集 150 回合、1650 行。两个模型各训练 100 个 epoch，评测用第 100 个 epoch。命令是 `NUM_TRAJ=2000 SHARDS=14 EPOCHS=100 bash scripts/run_tf_pipeline.sh`。全量规模在这一版环境和专家上还没有跑。

| | 规划成功率（move） | CortexBench reach | CortexBench move |
|---|---|---|---|
| LeWM 3D | **55.0**（52 / 50 / 56 / 62） | 86.3（86.5 / 86.2 / 86.3） | 41.8\*（40.2 / 40.6 / 44.5） |
| LeWM 2D | 26.5（20 / 16 / 34 / 36） | **87.1**（87.5 / 87.2 / 86.8） | 42.6\*（43.9 / 44.0 / 39.8） |
| 零动作 | 7.0（10 / 6 / 8 / 4） | — | — |
| 随机策略 | 0.5（0 / 0 / 0 / 2） | — | — |

\* move 是 CortexBench 中唯一一个策略以目标图像为条件的任务（目标是示范最后一帧的编码）。

- **原始记录**：这次运行的 `STABLEWM_HOME` 是 `lewm_pcd/TMP/trifinger/home`，以下路径都在它下面。
  - 运行日志：`logs/tf_cube_pipeline.log`。
  - 数据：`datasets/trifinger/tf_cube{,_pcd,_test,_test_pcd}.h5`。
  - 模型：`checkpoints/tf_cube_{pix,pcd_dexart}/`。
  - 规划评测日志：`logs/eval/tf_cube_{pix,pcd}_e100_s*.log`、`logs/eval/tf_cube_{random,hold}_s*.log`。
  - CortexBench 每次运行的记录：`logs/cortexbench/trifinger/lewm_tf_{pix,pcd}/{reach_cube,move_cube}/s*/`；示范和 3D 示范点云的缓存：`datasets/cortexbench/trifinger/assets/`。
- **已知问题（没有修）**：CortexBench 示范每一行记录的方块位姿来自 10 fps 相机的画面，可能比同一行的关节角早最多 0.1 s。3D 示范点云按这两者渲染，所以 move 的 3D 示范里方块可能落后手指最多 0.1 s。reach 的方块不动，不受影响。

## 与其他工作的对比（CortexBench TriFinger）

- "论文"行取自 VC-1 论文（arXiv 2303.18240）附录 Table 8 的 reach_cube、move_cube 两列。
- 按两个任务的均值排序，即 VC-1 论文中 TriFinger 一项的分数。

| 模型 | 来源 | reach | move |
|---|---|---|---|
| MVP (ViT-L) | 论文 | 84.8 | 63.4 |
| MAE Ego4D+MN (ViT-L) | 论文 | 82.4 | 62.4 |
| VC-1 ViT-B | 论文 | 83.9 | 60.6 |
| VC-1 ViT-L | 论文 | 83.3 | 60.2 |
| MAE Ego4D+MN (ViT-B) | 论文 | 78.5 | 62.1 |
| MAE Ego4D (ViT-L) | 论文 | 82.4 | 57.0 |
| R3M (RN-50) | 论文 | 86.5 | 51.9 |
| MAE Ego4D+M (ViT-B) | 论文 | 80.7 | 57.5 |
| MAE Ego4D (ViT-B) | 论文 | 82.6 | 54.0 |
| MAE Ego4D+N (ViT-B) | 论文 | 81.6 | 54.1 |
| MAE Ego4D+N (ViT-L) | 论文 | 81.1 | 52.7 |
| VIP (RN-50) | 论文 | 86.2 | 47.2 |
| **LeWM 2D（我们，图像）** | 本机 | 87.1 | 42.6\* |
| MAE Ego4D+M (ViT-L) | 论文 | 82.1 | 47.2 |
| **LeWM 3D（我们，点云）** | 本机 | 86.3 | 41.8\* |
| CLIP (ViT-B) | 论文 | 83.8 | 40.1 |
| MVP (ViT-B) | 论文 | 75.0 | 44.3 |
| Random (ViT-B) 冻结 | 论文 | 74.3 | 41.3 |
| Random (ViT-L) 冻结 | 论文 | 74.9 | 39.4 |
| Random (ViT-B) 端到端微调 | 论文 | 74.5 | 37.6 |

**可比性。**
- 其他工作都是通用预训练的视觉表征，没有见过 TriFinger。LeWM 用 TriFinger 同域的游玩数据训练过（没用 CortexBench 的示范），有同域优势。
- 3D 用模拟器的分割去掉了背景，只留手指的活动连杆和方块，相当于多了一份前景信息。
- 训练规模与 MetaWorld 不同：这里是 3.4 万行、100 个 epoch，MetaWorld 是 201 万行、10 个 epoch。
