# LeWM-PCD：点云版 LeWorldModel（OGBench Cube）

## Adroit Relocate（2026-10-07）：真实 PCD replay、BC 与 epoch-10 CEM

原 RGB v6 epoch-10 WM 保留；新 DexArt PCD WM 已完成 **10 epochs／127,260 optimizer steps**，严格重载与有限性检查通过。训练／独立test为10,000／150条200步轨迹，新PCD为真实可见1200×6 XYZRGB，区别于历史64个合成target点。下列3D BC与CEM共享同一PCD epoch-10权重；PCD是独立模态扩展。

BC沿用100条官方demos／每轮25条在线轨迹／100epochs，共20轮评测；冻结三帧192D表征加30D proprio，606D输入→30D动作。三seed均已完成，真实训练与策略重载验证退出码0。

| BC seed | 3D PCD Best-of-20 | Final（epoch99） |
|---|---:|---:|
| 100 | 84% | 84% |
| 200 | 96% | 92% |
| 300 | 84% | 68% |
| 均值±总体std（ddof=0） | **88.00±5.66%** | **81.33±9.98%** |

最高reward保存策略的对应success为84／96／84%；单seed54321的best88%／final68%另存，未混入三seed均值。Best-of-20为同评测集20轮选最大值，且当前PCD接口缺匹配随机表征对照。

| epoch-10世界模型 | 配对50片段CEM | 初始未达标的19段 |
|---|---:|---:|
| 2D RGB v6 | **43/50（86.00%）** | **12/19（63.16%）** |
| 3D DexArt PCD扩展 | **46/50（92.00%）** | **15/19（78.95%）** |

seed42按原采样器从150条test轨迹取相同50起点，覆盖44个来源episode；goal为`t+25`物体世界位置，距离严格<0.1m。保持horizon5／action_block5／receding5、budget50、300候选／top30／10次迭代、history1与原latent MSE cost；两路实际子进程退出码0。256原生渲染、224策略处理、MSAA0；归一化沿用Pen CEM的train-only StandardScaler。

**31/50（62%）起点已在目标内，两路这31段均成功。** 这是单次到达未来位置的片段扩展，不等同原生完整任务的>25帧保持成功；未做完整150episode评测或随机／无动作基线。CEM不能作为官方RGB BC分数，也不能与历史8% RGB BC直接比较或用于预训练／规划收益归因。

[本轮最终结果与绑定](../docs/results/adroit_relocate_20261007.json) · [代码／依赖／数据契约与复现入口](../docs/experiments/adroit_relocate_replay_20261007.md) · [仓库README与历史结果](../README.md)。以下Pen、OGBench和换臂结果按原协议保留。

**Adroit Pen 2026-10-06 更新：** 新 paired 数据已完成两路WM及100demo／25eval的三seed BC。RGB Best-of-20为37.3±5.0%，PCD扩展为81.3±5.0%；片段CEM和旧Relocate结果分别记录。见[仓库README](../README.md)和[本轮代码／外部前置／复现命令](../docs/experiments/adroit_pen_paired_20261006.md)。RH56仍为独立未通过任务门禁的候选，没有作为已调通代码混入。

## Adroit Pen 25步片段 CEM（2026-10-06，独立扩展）

起点抽样seed=42，两路使用相同50起点，来自独立test集中的40个episode；目标为同轨迹`t+25`原始环境步的笔位置／方向。保留原CEM的`horizon=5`、`action_block=5`（25步规划）、`receding_horizon=5`、50步预算、300候选／top30／10次迭代。两路正式运行均完成50段，真实退出码0。

| 世界模型 | 全部50段成功率 | 初始未达标的27段成功率 |
|---|---:|---:|
| 2D RGB | **34/50（68.0%）** | **11/27（40.7%）** |
| 3D PCD扩展 | **39/50（78.0%）** | **16/27（59.3%）** |

成功判据：位置距离<0.075m、方向dot>0.95，掉落失败。**23/50（46%）起点初始已达标**，两路这23段均成功；原采样和正式结果未过滤。上述为片段规划扩展，未覆盖150条完整episode，也没有随机／无动作基线；不能作为官方CortexBench BC分数或规划器收益归因证据。[逐段结果](../docs/results/adroit_pen_20261006_fragment_cem.json) · [配对与判据审计](../docs/results/adroit_pen_20261006_cem_audit.json)。

在 stable-worldmodel 的 `swm/OGBCube-v0` 环境上，把 LeWM 的像素输入换成点云输入，并加一个时间聚合的 4D 变体；采集、模型、训练、评测都沿用 LeWM 的做法，只改必要的部分。
同一套流程还跑了官方 2D 复现、像素 4D，以及框架自带的 PLDM / GCBC 基线。基于 le-wm `8edfeb3` 与 stable-worldmodel `6f1e499`。

## 结果

协议：官方 `config/eval/cube.yaml`（50 个起点、seed 42、目标 25 步后、预算 50 步、CEM 300 候选 / top 30），CEM 迭代次数按论文附录 D 取 10
（在 `cube_pix.yaml` / `cube_pcd.yaml` 里覆盖，官方文件不动）；每个 run 10 个 epoch 的 checkpoint 全部在 GPU/EGL 上评，
原始表 `logs/eval/success_rates_cem10.csv`（仓库里的副本 `results_success_rates_cem10.csv`）。
参照：随机策略在这 50 个起点上实测 5 次为 44 / 48 / 38 / 38 / 46，均值 42.8（论文报 48；随机策略的动作采样不固定种子）；作者发布的 checkpoint 在同一评测集、同一协议下 72 到 74；论文报 74。单个数字的统计误差约 ±6 分（50 个 episode），比较看均值。
下面两张表的点云行（3D / 4D）用的是之前的采集设定：6 台相机融合的点云，2D、3D 数据集分两次采、轨迹不对应。现在的采集是单台深度相机、2D 与 3D 出自同一次 rollout（见「数据采集」），两者不能直接比。

| run | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | 均值 | 最大 | 最终 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 官方 2D（ViT-tiny） (`cube_pix`) | 70 | 64 | 68 | 72 | 70 | 70 | 68 | 72 | 70 | 70 | 69.4 | 72 | 70 |
| 4D 像素（ViT-tiny + 聚合器） (`cube_pix_agg`) | 66 | 78 | 72 | 72 | 68 | 74 | 68 | 70 | 70 | 68 | 70.6 | 78 | 68 |
| 3D 点云 ppt (`cube_pcd_ppt`) | 68 | 72 | 70 | 66 | 66 | 64 | 64 | 66 | 66 | 64 | 66.6 | 72 | 64 |
| 3D 点云 dexart (`cube_pcd_dexart`) | 68 | 66 | 66 | 64 | 64 | 68 | 62 | 64 | 64 | 66 | 65.2 | 68 | 66 |
| 3D 点云 ms2 (`cube_pcd_ms2`) | 70 | 72 | 68 | 66 | 72 | 68 | 70 | 70 | 68 | 64 | 68.8 | 72 | 64 |
| 4D 点云 ppt (`cube_pcd_agg_ppt`) | 66 | 66 | 68 | 68 | 68 | 72 | 66 | 70 | 66 | 68 | 67.8 | 72 | 68 |
| 4D 点云 dexart (`cube_pcd_agg_dexart`) | 66 | 72 | 68 | 76 | 68 | 72 | 70 | 70 | 68 | 74 | 70.4 | 76 | 74 |
| 4D 点云 ms2 (`cube_pcd_agg_ms2`) | 66 | 66 | 68 | 62 | 64 | 62 | 64 | 66 | 66 | 64 | 64.8 | 68 | 64 |
| GCBC，框架基线：冻结 DINOv2-small + 行为克隆，逐步出动作、无规划 (`gcbc_cube`) | 92 | 90 | 96 | 94 | 92 | 88 | 96 | 96 | 96 | 98 | 93.8 | 98 | 98 |
| PLDM，框架基线：世界模型 + CEM 10 次，发布配置原样 (`pldm_cube`) | 54 | 50 | 50 | 52 | 52 | 48 | 54 | 52 | 52 | 56 | 52.0 | 56 | 56 |

- 官方 2D 复现均值 69.4，与作者 checkpoint 的 72 到 74 同一水平。
- 4D 对同 encoder 的 3D：dexart +5.2、像素 +1.2、ppt +1.2、ms2 −4.0，只有 dexart 这一对超出噪声。
- 两个框架基线和世界模型那几行不是一类，只作参照。GCBC 均值 93.8（论文 84）：评测的起点和目标都取自专家数据集（与训练同分布），目标条件行为克隆复现专家动作即可成功，
  用的又是预训练 DINOv2-small 特征、每个环境步都闭环出动作，不靠规划。PLDM 均值 52.0（论文 65）接近随机：损失系数就是论文附录 C.2 表 2 那组（发布的 `pldm.yaml` 与之一致），
  10 个 epoch 也相同；评测路径核查过没有问题（作者的 checkpoint 走同一路径得 74），是训出来的模型代价面太尖，离开专家动作序列代价立刻跳到随机序列的水平，CEM 没有坡可下。
  GCIQL / GCIVL 未评（见"框架基线"）。
- 与论文 74 的差距来自评测集（自采数据的另一组 50 道题，其中 17 个起点时方块已在目标 4 cm 内）和框架版本。
- 论文图 6 的 cube 对照数：DINO-WM 86、GCBC 84、LeWM 74、PLDM 65、GCIQL 64、GCIVL 56、随机 48。

### RM75-B + EG2-4C2 上的三个 run（换臂，见「换臂」一节）

同一协议（50 个起点、seed 42、CEM 10 次、GPU/EGL），数据是 RM-75 环境上新采的 10000 集（`cube_rm75_single_expert{,_pcd}.h5`），起点从这份数据里抽。
参照：随机策略在这 50 个起点上点云 50、像素 44。工作区缩到 0.5 倍、专家搬运高度受限，这张表和上面 UR5e 的表不能直接比，三行之间可比。

| run | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | 均值 | 最大 | 最终 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 官方 2D（ViT-tiny） (`cube_rm75_pix`) | 76 | 68 | 72 | 74 | 74 | 76 | 74 | 74 | 74 | 76 | 73.8 | 76 | 76 |
| 3D 点云 dexart (`cube_rm75_pcd_dexart`) | 82 | 84 | 80 | 90 | 80 | 74 | 76 | 80 | 78 | 74 | 79.8 | 90 | 74 |
| 4D 点云 dexart (`cube_rm75_pcd_agg_dexart`) | 82 | 82 | 90 | 86 | 82 | 82 | 80 | 78 | 78 | 78 | 81.8 | 90 | 78 |

- 点云比像素高 6 到 8 个点，4D 比 3D 高 2 个点，都在单次 ±6 的噪声边缘；三条曲线都在第 3、4 个 epoch 附近最高、之后略降。

## 与 LeWM 的对应

| 环节 | LeWM 原件 | 本仓库 |
|---|---|---|
| 采集 | stable-worldmodel `scripts/data/collect_cube.py` + `ExpertPolicy(policy_type='plan_oracle')` | `scripts/data/collect_cube_pcd.py`：一次 rollout 同时写 2D（像素）和 3D（点云）两个数据集，逐行对应；点云由 `envs/pcd_wrapper.py` 从同一台相机的深度反投影；纯像素版 `scripts/data/collect_cube_pix.py` |
| 模型 | `jepa.py`（ViT encoder + ARPredictor）、`module.py` | 同名文件就地改（encoder 可插拔）；4D 在 `jepa_agg.py` |
| 训练 | `train.py` + `config/train/lewm.yaml` + `config/train/data/ogb.yaml` | `train.py`（预处理器由配置实例化）+ `config/train/data/ogb_pcd.yaml` + `scripts/train/*.sh` |
| 评测 | `eval.py` + `config/eval/cube.yaml` | `eval.py`（点云策略、4D 的历史与目标窗口）+ `config/eval/cube_{pix,pcd}[_agg].yaml` |

论文那份 cube 数据的采集设定（stable-worldmodel commit 5223ba5 的 `collect_cube.py`）：single、pixels、非 multiview、224×224、`visualize_info=False`、
`terminate_at_goal=False`、`mode='data_collection'`、plan_oracle、10 个并行 env、每回合 200 步、10000 回合、seed 3072。配对采集沿用同一套，一次 rollout 把像素写进 2D 数据集、点云写进 3D 数据集；纯像素采集就是这一套。

## 目录

```
lewm_pcd/
├── jepa.py module.py utils.py train.py eval.py   ← 来自 le-wm@8edfeb3，改动见各节
├── jepa_agg.py aggregators/   4D 模型（时间聚合）
├── encoders/                  可插拔 encoder：vit（像素基线）、ms2、dexart、ppt、sonata_cls、pointnet（参考）
├── envs/                      点云环境、包装器、专家策略、评测策略
├── config/{train,eval}/       训练 / 评测配置（launcher/local.yaml 里加了 hydra 输出目录）
├── scripts/data|train|eval|checks/   采集 / 训练 / 评测 / 检查脚本；scripts/baselines/ 是框架基线补丁的记录；scripts/run_baselines.sh 跑基线
├── third_party/stable-worldmodel/   固定在 6f1e499 的本地 clone（不进 git）
├── env.sh                     环境变量，跑任何脚本前先 source
├── TMP/                       杂物，不进 git
└── LICENSE-le-wm              le-wm 的 MIT 许可
```

## 环境与数据盘

| 组件 | 版本 |
|---|---|
| le-wm | github.com/lucas-maes/le-wm @ 8edfeb3 |
| stable-worldmodel | github.com/galilai-group/stable-worldmodel @ 6f1e499（`third_party/`） |
| ogbench | 1.2.0 |
| mujoco | 3.12.0（EGL 无头渲染） |
| torch | 2.8.0+cu128，Python 3.10；`pip install -e third_party/stable-worldmodel[train,env]`，`ogbench` |

`env.sh` 把 `STABLEWM_HOME` 指到数据盘的 `lewm_pcd_data/`，LeWM 和 stable-worldmodel 的路径约定全部落在那里：

| 内容 | 位置 | 由谁写 |
|---|---|---|
| 数据集 | `datasets/ogbench/cube_single_expert_pcd.h5`（点云）、`cube_single_expert.h5`（像素） | 采集脚本（`swm.World.collect`） |
| 模型 | `checkpoints/<run>/`：`config.yaml`、`weights_epoch_N.pt`、`<run>_weights.ckpt` | `train.py` 的 `SaveCkptCallback` |
| Lightning 运行目录（含 last.ckpt） | `checkpoints/_spt_cache/`（`SPT_CACHE_DIR`） | stable-pretraining |
| 训练 / 评测 / 采集日志 | `logs/train/`、`logs/eval/`、`logs/collect/` | hydra、各脚本 |

## 数据采集

- 环境 `swm/OGBCubePCD-v0` = `swm/OGBCube-v0` + 几台非物理相机（`envs/cube_pcd_env.py`，默认不用，只是单台深度相机的备选视角）。动力学、RNG、专家、seed、步数、回合数与论文采集一致；
  `envs/expert_policy.py` 只放宽了上游 `ExpertPolicy` 对 env id 的白名单。
- 2D、3D 同时采：`envs/pcd_wrapper.py` 作为 `pre_wrapper` 每步往 info 加 `pcd_xyz (1200,3) f32` 和 `pcd_rgb (1200,3) u8`，同一步里 swm 自己的 `AddPixelsWrapper` 渲染官方的 224×224 `pixels`；
  `collect_cube_pcd.py` 的 `PairedWriter` 把每个回合写两份：点云以外的列写进 2D 数据集（`pix_dataset_name`，编码同 `collect_cube_pix.py`），`pixels` / `render_time` 以外的列写进 3D 数据集（`dataset_name`）。
  两边的状态、动作等列出自同一批回合、逐行相同，2D 与 3D 一一对应（训练的切分、评测抽到的 50 个起点也就相同）；分两次采做不到这一点（见下面的等价性）。
- 3D 只用一台 eye-to-hand 深度相机，就是 2D 用的官方 `front_pixels`：按 ToF 相机处理，直接用深度图，不用双目、不做多相机融合（wrapper 拒绝多于一台相机）。每帧流程：渲染 RGB / 深度 / 分割 → 反投影到世界系 →
  去掉桌面、底面和目标 marker → 只保留 `crop_box` 内的点 → 5 mm 体素去重 → 分层随机采样（cube 最多 25%，其余机械臂）。单视角 224×224 下（试采几回合）cube 可见面中位约 145 个体素，
  所以 cube 实占约 12%；机械臂中位 1500 到 3000 个体素，约 5–10% 的帧不够凑满 1200 点，按有放回抽。
- 等价性：`scripts/checks/check_equivalence.py` 把三处随机源（env 的 `np_random`、变化空间 RNG、oracle 用的全局 `np.random`）钉成相同 seed 后，
  论文配置与点云配置在 4 条回合上 26 列 qpos / qvel / action / info 逐位一致，只有 `pixels` / `render_time` 换成 `pcd_xyz` / `pcd_rgb`。
  上游 `CubeEnv.reset` 不把 seed 传给底层环境，起点 / 目标由变化空间用系统熵采样，oracle 噪声用未 seed 的全局 `np.random`，所以采集本身不可逐位复现，各分片自然互相独立，
  分开跑的 2D、3D 两次采集轨迹也不同（所以要上面的配对采集）。
- 全量采集：`scripts/data/collect_cube_pcd_sharded.py --shards N --out-pix ogbench/cube_single_expert.h5`（每片 10 env × 200 步，同时写 3D、2D 两个分片；
  每片要渲染像素、内存约 5 GB，用 `--max-parallel` 限制同时跑的片数，`run_collection_detached.sh` 默认 8），结束后两套分片按同一顺序用 `merge_h5_shards.py` 各合并成一个 h5，
  再抽查两边的 `ep_len` / `ep_offset` / `id` / `qpos` / `action` 逐行一致（上游 `HDF5Writer` 的 append 校验对嵌套列有 bug，合并脚本自己按列追加）。
  纯像素（官方复现）用 `--script collect_cube_pix.py`，不带 `--out-pix`，它和任何 3D 数据都不对应。
- 合并后跑一次 `scripts/data/add_eval_columns.py`：官方 `eval.py` 按论文时代的 h5 布局工作（每行 `ep_idx`、扁平列名），新写入器只给 `ep_len/ep_offset` 和嵌套分组，
  脚本加 `ep_idx`、用硬链接把嵌套列扁平化，不复制数据。`scripts/data/rechunk_h5.py` 把每步一个 chunk 的文件按 20 行一个 chunk 重写（否则每个训练进程仅读一次 `action` 列就占 12 GB 内存）。
- 像素数据集用 `GzipHDF5Writer`（pixels gzip-4、5 帧一个 chunk），值与官方写入器相同。

## 模型与 encoder

契约见 `encoders/__init__.py`：输入 `(B, N, C)` 点云（C=3 或 6，由 `pcd_in_dim` / `preprocessor.use_rgb` 决定），输出 `(B, embed_dim=192)`。
接一个新架构：把类放进 `encoders/`，在 `config/train/model/encoder/` 加一个 yaml，训练时 `model/encoder=<名字>`。所有 encoder 只调输出维度到 192，其余架构不动。

- `jepa.py` 改动 6 行：encoder 调用改为通用的 `self.encoder(x)`，观测键改为构造参数 `obs_key`（像素 `pixels`，点云 `pcd`）。
- `encoders/vit.py`：像素基线适配器，`scripts/checks/check_model_equivalence.py` 验证与上游 `jepa.py` 的嵌入 / 预测逐位相同（state_dict 键只多一层 `encoder.vit.` 前缀）。
- `utils.py` 新增 `PointCloudPreprocessor`：xyz 减工作区中心 `(0.425, 0, 0.2)` 乘 4，rgb/255，拼成 `(T, 1200, 6)` 放到 `pcd`。固定仿射，不用数据集统计量（对无序点集按点序做 z-score 无意义）。
- `train.py` 改动 2 行：预处理器由 `data.preprocessor` 实例化；归一化跳过 `pcd*` 列。像素数据配置各自声明原来的 `get_img_preprocessor`，行为不变。

| encoder | 结构 | 参数量 |
|---|---|---|
| `encoders.ManiSkillPointNet`（`ms2`，`model=lewm_pcd` 默认） | ManiSkill2-Learn `maniskill2_pn.py` 的 PointNet 原样复制：逐点 Conv1d(1×1, 无 bias) → LayerNorm(eps 1e-6) → ReLU ×3，沿点 max-pool，无 T-Net；一层 Linear 调到 192 | 原宽度 `[64,128,512]` 主干 75,520；配置默认 `[512,1024,4096]` 共 5.52M，对齐像素 ViT-tiny 的 5.50M |
| `encoders.DexArtPointNet`（`dexart`） | DexArt `pretrain_nets.py` 的 PointNet 原样复制：逐点 Linear → GELU → Linear，无归一化，末层无激活，沿点 max-pool，trunc_normal(0.02) 初始化；一层 Linear 调到 192 | smallpn 原宽度 `[64,256]` 主干 17K；配置默认 `[1080,4320]` 共 5.51M |
| `encoders.PointPatchTransformer`（`ppt`） | PPRL `conf/model/ppt.yaml` 的 PointPatchTransformer 用纯 PyTorch 重写：FPS(0.0625 → 75 个中心) + kNN(32) 分组，组内 mini-PointNet 两级 MLP 出 token，正弦位置编码，3 层 6 头 Transformer（mlp_ratio 3），注意力池化；末尾 Linear 调到 192。eval 模式下 FPS 起点固定，输出确定 | 原 embed_dim 384 共 5.01M；配置默认 embed_dim 402 共 5.45M |

显存对策（不改架构、不改 batch）：两个 PointNet 在训练时沿点维分块并用 activation checkpointing 重算（`chunk_points=300`），逐点 LayerNorm 与 max-pool 保证结果与整体前向数值等价；
checkpoint 的 config 记录了 `chunk_points`，评测用同一值；ManiSkill PointNet 的 Conv1d 1×1 改为等价的逐点 Linear 布局以去掉转置。

### Sonata encoder（来自分支 `lewm-pcd-sonata`；预训练冻结骨干，不属于参数量对齐的那一组）

- **是什么**：`encoders/sonata_cls.py` 的 `SonataCLSEncoder`。冻结的 Sonata 骨干（PTv3，预训练权重 `sonata_small`，`eval` + `no_grad`）
  提取每个 2 cm 体素的特征（拼最后 3 级），后接可训练的读出头：LayerNorm–Linear–LayerNorm 投影到 192 维、三维正弦位置编码、
  带 CLS token 的 2 层 8 头 Transformer，输出 CLS 向量 `(B, 192)`，接现有的 projector / predictor，3D 和 4D 都能用。只训读出头（约 1M 参数）。
  输入是 `(B, N, 6)`：米单位的世界坐标 xyz + `[0, 1]` 的 rgb（编码器自己做 CenterShift 和体素化）。
- **和上面三个的区别**：ms2 / dexart / ppt 是从头训练、参数量对齐 ViT-tiny 5.50M 的一组；Sonata 是外部预训练骨干，比较时单独标注。
- **配置**：`config/train/model/encoder/sonata_cls.yaml`（骨干权重路径、sha256、读出头尺寸）。因为要米单位坐标，
  `config/train/data/ogb_pcd_sonata.yaml` 和评测配置 `config/eval/cube_pcd_sonata.yaml` / `cube_pcd_agg_sonata.yaml` 把点云仿射改为恒等（center 0、scale 1，rgb 仍 /255）；其他 encoder 的配置不受影响。
- **依赖（本机未装、未跑过）**：`pip install git+https://github.com/facebookresearch/sonata@18c09ff8d713494f78a8213792262b910977a65d`（及其 README 里的 spconv / torch-scatter 等依赖）；
  权重 `sonata_small.pth` 放 `$STABLEWM_HOME/checkpoints/sonata/`（sha256 `ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1`，加载时校验），权重不进 git。
  没装 `sonata` 包时只是这个 encoder 不能实例化，`import encoders` 和其他模型不受影响（包在构造时才导入）。
- **怎么跑**：3D `DATA=ogb_pcd_sonata bash scripts/train/run_train_detached.sh lewm_pcd cube_pcd_sonata model/encoder=sonata_cls`；
  4D `DATA=ogb_pcd_sonata bash scripts/train/run_train_detached.sh lewm_pcd_agg cube_pcd_agg_sonata --config-name=lewm_agg model/encoder=sonata_cls`；
  逐 epoch 评测循环按 run 名里的 `sonata` 自动选对应评测配置。单元测试：`scripts/checks/test_sonata_cls_encoder.py`（假骨干，CPU）。
  分支自带的对照脚本没有收进来（它给 Sonata 传的是非恒等仿射的配置，与这个 encoder 的输入约定冲突）。

## 4D 世界模型（时间聚合）

- 结构：encoder → projector → latent1（每帧，与 3D 模型完全相同）→ 聚合器（前 3 帧 latent1 → latent2，同形状向量）→ predictor(latent2, 动作块) → 下一帧 latent1。
  规划时世界模型的状态是 latent2，CEM 在 latent2 空间算 cost，目标是目标帧及其前两帧聚合出的 latent2。
- 训练：同上游数据（4 帧片段，frameskip 5）。一个样本 = 前 3 帧聚合、用第 3 个动作块预测第 4 帧 latent1；loss = MSE(pred, e4) + 0.09 × SIGReg([latent2, e4])，
  SIGReg 一次调用对两个位置取平均（与上游对 4 个位置取平均同形式），λ 不动。
- 代码：`jepa_agg.py`（`AggJEPA(JEPA)`，复用 encode / projector / action_encoder / criterion，重写 predict / rollout / get_cost；`rollout` 是框架 `LeWM.rollout` 的逐字拷贝，只改观测键和单步预测；
  `agg_forward` 为训练前向）；`aggregators/`（`temporal` 默认：ARPredictor 的位置编码 + 因果 block，不带动作，取最后 token；`mlp`、`mean` 对照）；predictor 复用 ARPredictor 但 `num_frames=1`。
- 配置：`model/lewm_pcd_agg.yaml`、`train/lewm_agg.yaml`、`eval/cube_pcd_agg.yaml`（= cube_pcd + `plan_config.history_len 3` + `eval.goal_window 3`）；像素版 `model/lewm_pix_agg.yaml`、`eval/cube_pix_agg.yaml`。
  上游文件改动：`train.py` 1 行（前向按配置选）、`eval.py` 的历史 / 目标窗口分支。
- 评测时的历史：`envs/eval_policy.py` 的 `WindowedWorldModelPolicy` 用数据集里每个起点之前的 10 步填满历史缓冲，第一次规划就像训练片段一样看到 start−10、start−5、start 三帧；
  起点离回合开头不足 10 步时只取整块的真实步，其余用框架自己的补齐（最旧真实帧的拷贝、零动作）。目标是 goal−10、goal−5、goal 三帧。
- 命令：训练 `python train.py --config-name=lewm_agg data=ogb_pcd model/encoder=ppt subdir=<run> output_model_name=<run>`；评测 `python eval.py --config-name=cube_pcd_agg.yaml policy=<run>/weights_epoch_N.pt`。
- 参数量：聚合器 3.15M（2 层，block 形状与 predictor 相同：16 头 × 64、MLP 2048），整体 ≈ encoder 5.5M + 聚合器 3.2M + predictor 10.8M + 投影头 0.9M ≈ 20.4M（3D 模型 17.2M）。
- CEM 求解器按 `solver.batch_size=1` 逐个 env 调用 `get_cost`，上游 `criterion` 的广播只对 B=1 成立，4D 模型沿用同一 criterion。
- 检查脚本：`scripts/checks/test_agg_rollout.py`（rollout / get_cost 单测）、`scripts/checks/check_history_order.py`（评测时历史与目标窗口的顺序）。

## 训练

- 超参按论文：batch 128、10 epoch（论文附录 E 四个环境均为 10 epoch，发布的配置文件写 100，以论文为准）、AdamW 5e-5、weight decay 1e-3、bf16；
  学习率调度是 stable-pretraining 的 `LinearWarmupCosineAnnealingLR` 默认：按 step 计，前 1% 步从 0 线性升到 5e-5，之后余弦退火到 0。每 epoch 12,796 步。
- 启动：`bash scripts/train/run_train_detached.sh <model 配置> <run 名> [hydra 覆盖]`（`DATA=ogb` 为像素；`--config-name=lewm_agg` 为 4D），脱离会话；
  本地跑 `train.py` 一定要传 `subdir=<run 名>`，否则 checkpoint 互相覆盖。崩溃续训：`scripts/train/resume_train_detached.sh`。
- 官方 2D 与像素 4D 的完整流程（采集 → 合并 → 评测列 → 训练 → 逐 epoch 评测）：`scripts/run_pix_pipeline.sh`、`scripts/run_pix_agg_followup.sh`。
- 容器内存上限 90 GB：像素数据集的页缓存加两个训练会顶到上限并被平台杀掉进程，训练时要控制并行数（脚本里的并行度和显存门槛是本机参数）。

## 评测

官方 `eval.py` + `config/eval/cube.yaml` 的流程原样保留（50 个起点、`set_state(qpos,qvel)` 复位、25 步后的 cube 位姿为目标、CEM 300 样本 top-30、预算 50 步、seed 42、`terminate_at_goal` 判成功）。

- 配置：`cube_pix.yaml`（= 官方 `cube.yaml` + 设备 / 视频开关 + CEM 10 次）、`cube_pcd.yaml`（再加环境 `swm/OGBCubePCD-v0` + 采集用的同一个 wrapper、数据集名、`obs: pcd`、点云仿射参数、`video: false`）、
  `cube_pix_agg.yaml` / `cube_pcd_agg.yaml`（4D：`history_len 3`、`goal_window 3`）。
- 命令：`python eval.py --config-name=<配置>.yaml policy=<run>/weights_epoch_N.pt [device=cpu solver.device=cpu]`。
- `eval.py` 的改动：wrapper 从配置实例化；点云时用训练同一仿射走 policy 的 `process` 通道并校验与 checkpoint 的训练配置一致；策略用 `envs/eval_policy.py`
  的 `PCDWorldModelPolicy`（把 `pcd_xyz/pcd_rgb` 拼成 `pcd`、`goal_pcd_*` 拼成 `goal`，丢掉只用于视频的 `pixels`）；4D 的历史与目标窗口见上节；
  `MUJOCO_GL` 与模型设备可配置（CPU 上用 osmesa 渲染）。
- 框架接口对齐（像素和点云共用）：求解器在 `stable_worldmodel.planning.solver`，世界模型通过 `cost=` 传入（`config/eval/solver/*.yaml`、`eval.py` 各改一处）。
- 结果文件按官方逻辑写到 `$STABLEWM_HOME/checkpoints/<run>/ogb_cube_*_results.txt`；hydra 日志在 `logs/eval/`。

### 逐 epoch 自动评测

`scripts/eval/auto_eval_checkpoints.sh`（脱离会话运行）盯着 `RUNS` 里各 run 的 checkpoint 目录，每出现一个 `weights_epoch_N.pt` 就按 run 名选配置评一次
（`*pix_agg*` → `cube_pix_agg`，`*pix*` → `cube_pix`，`*sonata*` → Sonata 配置，`*agg*` → `cube_pcd_agg`，其余 `cube_pcd`；`pldm_*` / `gcbc_*` 等走框架自己的 `eval_wm.py` / `eval_ff.py`）。
训练优先：同一时刻只跑一个评测且 `nice -n 10`；显存不足时整套评测在 CPU 上跑（约 40 分钟，GPU 约 3 分钟）；`GPU_ONLY=1` 时不落 CPU，等显存够再评；
内存守护只看 cgroup 的 anon 内存（`memory.current` 里有可回收的页缓存），超限杀掉评测稍后重试，失败同样重试，最多 3 次。门槛都是环境变量，默认值是本机参数。
结果：`logs/eval/success_rates_cem10.csv`（run, epoch, success_rate, 秒数, 渲染后端, 时间），完整日志在 `logs/eval/auto/`。

## 换臂：RealMan RM75-B + 因时 EG2-4C2（`envs/cube_rm75_env.py`）

把 OGBench 的机器人层从 UR5e + Robotiq 2F-85 换成睿尔曼 RM75-B 七轴臂加因时 EG2-4C2 二指夹爪（睿尔曼官方模型库 `rm_models` 里的配套夹爪）；任务层、专家策略、点云管线、模型、训练、评测全部复用。

- **改了什么**：只有 `ManipSpaceEnv` 的机器人层，在子类 `RM75ManipSpaceEnv` 里覆盖三个方法：`__init__`（7 关节 home 姿态、用新臂建 IK、工作区四个盒子等比缩放）、
  `build_mjcf_model`（照抄上游，只换两个模型文件路径）、`post_compilation`（照抄上游，PD 增益 6 个改 7 个）。`RM75CubeEnv(CubeEnv, RM75ManipSpaceEnv)` 和
  `RM75CubePCDEnv(CubePCDEnv, RM75ManipSpaceEnv)` 用多重继承拼出像素版和点云版，注册为 `swm/OGBCubeRM75-v0`、`swm/OGBCubeRM75PCD-v0`。
  新模型 XML 沿用上游的槽位名 `ur5e` / `robotiq` 和实体名（`attachment_site`、`pinch`、`right_driver_joint`、`right_pad`、材质名），上游按名字找机器人的代码一处不动。
- **模型文件** `envs/descriptions/realman_rm75b/`、`inspire_eg2_4c2/`：在 ogbench 自带的 `ur5e.xml` / `2f85.xml` 模板上把机器人数据换成厂商 URDF 的连杆位姿、惯量、限位、力矩上限（STL 用厂商的）；
  夹爪的四连杆用关节等式约束实现 mimic，驱动关节量程 0 到 0.8，所以上游 `qpos / 0.8` 的开合度和 0 到 255 的指令原样成立。厂商原始 URDF / STEP / 尺寸图放数据盘 `vendor/realman/`。
- **三个几何决定**：RM75 臂展 610 mm（UR5e 850），夹爪竖直向下时底座放哪都够不全 UR5e 的工作区。底座放桌面 (0.15, 0, 0)，四个盒子绕原中心 (0.425, 0) 等比缩到 0.5 倍
  （0.55 以上上游 IK 在远角不收敛）；中心不动，相机、点云仿射、评测起点抽样沿用 UR5e 的，代价是任务区域只占画幅和归一化范围的约一半（方块的像素大小不变，能出现的范围小了）。`CubeEnv` 写死的两处 UR5e 坐标（variation 默认起止点、task 模式的方块坐标）按同比例缩放（`scale_cube_task`）。
  点云裁剪框 x 下限 0.10 改 0.21，因为底座前移进了原框。这三个值改了就要重新验收（`scripts/checks/verify_rm75.py` 的 D 节）。
- **工作区高度**：等比缩放后 z 上限 0.185 m，可达性扫图证实这就是物理上限：夹爪竖直向下时腕心比夹点高 29 cm，在整个 x、y 盒子里全覆盖的最高高度是 0.19 m，0.24 m 起碰关节限位，0.34 m 以上到不了（底座垫高 10 cm 可到约 0.29 m）。专家脚本写死的抬举高度（方块上方 0.1 到 0.2 m 加余量）因此被环境压到天花板，约 18% 的步竖直分量被裁，抓放和平移照常，专家成功率 99%；数据里的搬运高度比 UR5e 低，是两边不可直接比的原因之一。
- **已知差异**：夹爪夹持力按数据表 20 N，`proprio/gripper_contact` 夹住时约 0.42，swm 的 markov oracle（判夹住的阈值 0.5）在这台机器人上会卡住；采集用的 plan oracle 不受影响。
  夹爪型号若实为睿尔曼自研 RMG24（130 N），只需重做夹爪 XML，接口名不变。工作区缩到 0.5 倍后任务比 UR5e 的小，随机策略成功率约 60（UR5e 数据 43），两张表不能直接比。
- **验收** `scripts/checks/verify_rm75.py`：臂和夹爪逐项对厂商 URDF（关节限位、质量、惯量、2000 组随机关节角的正运动学、联动、开口、夹持力）、接口对原环境（动作空间、控制周期、info 键、相机、variation）、
  plan oracle 50 回合的成功率 / 限位 / 自碰撞 / 关节速度，以及数据集的点云统计；`CUDA_VISIBLE_DEVICES= python scripts/checks/verify_rm75.py`，报告写到 `logs/verify_rm75.txt`。
- **怎么跑** `scripts/run_rm75_pipeline.sh`：配对采集（`scripts/data/config/ogb_rm75_pcd.yaml`，一次写 3D、2D 两个数据集）→ 两个数据集上评随机策略 → 三个训练：
  官方 2D（`data=ogb_rm75 model=lewm`）、3D 和 4D 各一个 encoder（默认 dexart）→ 逐 epoch 评测（`cube_rm75_{pix,pcd,pcd_agg}.yaml`）。数据集 `cube_rm75_single_expert{,_pcd}.h5`，run 名 `cube_rm75_*`。
- **动到的旧文件**：`envs/__init__.py` 两条 register；`scripts/data/collect_cube_pcd.py`、`collect_cube_pix.py` 的 env id 改从配置读（默认值不变，官方路径原样；像素版在 RM-75 时用仓库里的 `PCDExpertPolicy`，同一个 oracle 去掉 env id 白名单）；`scripts/eval/auto_eval_checkpoints.sh` 四行 case。

## 框架基线（PLDM、GCBC）

用我们采的 `cube_single_expert.h5`，框架 `scripts/train/{pldm,gcbc}.py` 原样运行（只传数据名、`action_dim=5`、10 epoch、`+wandb.enable=false`），
评测用框架自己的 `scripts/plan/eval_wm.py`（PLDM，CEM 10 次）和 `eval_ff.py`（GCBC）；训练命令在 `scripts/run_baselines.sh`。DINO-WM 不在框架里，未做。
框架在 6f1e499 上有四处脚本级小错，各改了一处，另加一处按需改动；都是原地改、带 `# patched` 注释，diff 存在 `scripts/baselines/*.patch`，不涉及 LeWM、点云、4D 的任何代码路径：
1. `scripts/train/pldm.py`：`logger` 在函数里被重新赋值，打日志那行必报 UnboundLocalError，改用 `logging.getLogger(__name__)`。
2. `scripts/train/gcbc.py`：`ckpt_path` 无条件指向不存在的续训文件，首次运行必崩，加 `exists()` 判断（`gciql.py` / `gcivl.py` 本来就有）。
3. `stable_worldmodel/wm/gcrl/gcrl.py`：向 transformers 4.50 的 `Dinov2Model` 传了不存在的 `interpolate_pos_encoding` 参数，默认值改为 `False`（数值不变）。
4. `scripts/plan/eval_ff.py`：调用的 `World.evaluate_from_dataset(..., goal_offset_steps=, video_path=)` 在库里已改名为 `World.evaluate(..., goal_offset=, video=)`，改调用处。
5. `pldm.py` / `gcbc.py` 存 checkpoint 的间隔从每 5 / 每 3 个 epoch 改为每个 epoch（逐 epoch 评测）。
GCIQL / GCIVL 能训不能评：训练脚本把整份 hydra 配置存成 `config.json`，框架的 `load_pretrained` 重建不了模型，`eval_ff.py` 读不了，要评得自写加载代码。

## 与论文的已知差异

- 数据是同一采集程序的另一次采集，不是作者发布的那份；评测的 50 个起点随之不同（随机策略 42.8 对论文 48）。
- CEM 迭代次数论文前后矛盾（附录 B 写 30，附录 D 写 10），这里取附录 D 的 10；发布的 `solver/cem.yaml` 写 30，用 30 时作者 checkpoint 在本评测集上为 66。
- 框架版本：le-wm `8edfeb3` 面向 stable-worldmodel `abdced4`，这里用 `6f1e499`；两者在 CEM 上只差精英样本标准差的自由度修正（n−1 对 n），作者 checkpoint 两种算法各得 72 / 74。
- 基线：GCBC 的 epoch 数论文没写，这里取 10；PLDM 只训了一个种子。
