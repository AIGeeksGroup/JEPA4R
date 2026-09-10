# LeWM-PCD：点云版 LeWorldModel（OGBench Cube）

目标：在 stable-worldmodel 的 `swm/OGBCube-v0` 环境上，把 LeWM 的像素输入换成点云输入，
采集、模型、训练三步都沿用 LeWM 的做法，只改必要的部分。

## 三项工作与落点

| 工作 | LeWM 原件 | 本仓库落点 |
|---|---|---|
| 1. 按它的方法采集点云样本 | stable-worldmodel `scripts/data/collect_cube.py` + `ExpertPolicy(policy_type='plan_oracle')` | `envs/cube_pcd_env.py`（继承 swm `CubeEnv`，在 info 里加点云）+ `scripts/data/collect_cube_pcd.py` |
| 2. 按它的方法微调模型架构 | `jepa.py`（ViT encoder + ARPredictor）、`module.py` | 同名文件就地改；新增 `config/train/model/lewm_pcd.yaml` |
| 3. 按它的方法训练 | `train.py` + `config/train/lewm.yaml` + `config/train/data/ogb.yaml` | `train.py`（把 `get_img_preprocessor` 换成点云预处理）+ `config/train/data/ogb_pcd.yaml` + `scripts/train/*.sh` |

论文那份 cube 数据的采集设定（stable-worldmodel commit 5223ba5）：single、pixels、非 multiview、224×224、
`visualize_info=False`、`terminate_at_goal=False`、`mode='data_collection'`、plan_oracle、
10 个并行 env、每回合 200 步、10000 回合、seed 3072。点云采集沿用同一套，只是多存一列点云。

## 目录

```
lewm_pcd/                      代码（git）
├── jepa.py module.py utils.py train.py eval.py   ← le-wm@8edfeb3 原样拷贝，作为基线
├── config/{train,eval}/       ← 同上；launcher/local.yaml 里加了 hydra 输出目录
├── envs/                      点云环境包装（工作 1）
├── scripts/data|train|eval/   采集 / 训练 / 评估的启动脚本
├── third_party/stable-worldmodel/   固定在 6f1e499 的本地 clone（不进 git）
├── env.sh                     环境变量，跑任何脚本前先 source
├── TMP/                       杂物，不进 git
└── LICENSE-le-wm              le-wm 的 MIT 许可
```

## 存储规划（全部在数据盘 /root/autodl-tmp，系统盘只有 30G）

`env.sh` 把 `STABLEWM_HOME` 指到 `/root/autodl-tmp/lewm234d/lewm_pcd_data`，LeWM 和 stable-worldmodel 的路径约定就全部落到这里：

| 内容 | 位置 | 由谁写 |
|---|---|---|
| 训练数据 | `lewm_pcd_data/datasets/ogbench/cube_single_expert_pcd.{h5,lance}` | `scripts/data/collect_cube_pcd.py`（`swm.World.collect`） |
| 模型 | `lewm_pcd_data/checkpoints/<subdir>/`：`config.yaml`、`weights_epoch_N.pt`、`<name>_weights.ckpt` | `train.py` 的 `SaveCkptCallback` |
| 训练日志 | `lewm_pcd_data/logs/train/<日期>/<时间>_<模型名>/`（hydra 输出、`train.log`）；wandb 目录也在 `logs/train/` | hydra / WandbLogger |
| 评估结果 | `lewm_pcd_data/logs/eval/<日期>/<时间>/`（含 `*_results.txt`） | `eval.py` |
| 采集日志 | `lewm_pcd_data/logs/collect/` | 采集脚本 |
| Lightning 运行目录（含 last.ckpt） | `lewm_pcd_data/checkpoints/_spt_cache/`（`SPT_CACHE_DIR`） | stable-pretraining |

注意：
- 本地跑 `train.py` 一定要传 `subdir=<run_name>`。默认 `subdir=${hydra:job.id}`，本地 launcher 下每次都一样，checkpoint 会互相覆盖。
- 数据盘目前剩 55 GB。2M 帧（10000 回合 × 200 步）按不同点云规格的占用：

| 每帧格式 | 每帧 | 2M 帧 |
|---|---|---|
| 像素 224×224×3 uint8（官方） | 150 KB | 301 GB（官方 zst 后 46 GB） |
| 2048 点 × (xyz f32 + rgb u8) | 30 KB | 61 GB |
| 1024 点 × (xyz f16 + rgb u8) | 9 KB | 18 GB |

  点数、精度、回合数三者要在实现采集脚本时一起定。

## 采集方案（已定，2026-09-09）

- 环境 `swm/OGBCubePCD-v0` = `swm/OGBCube-v0` + 4 台非物理相机（`envs/cube_pcd_env.py`）。动力学、RNG、专家、seed、步数、回合数与论文采集完全一致；
  `envs/expert_policy.py` 只放宽了上游 `ExpertPolicy` 对 env id 的白名单。
- `envs/pcd_wrapper.py` 作为 `pre_wrapper` 每步往 info 加 `pcd_xyz (1200,3) f32` 和 `pcd_rgb (1200,3) u8`，其余列照旧；不再渲染/存 224×224 图像。
- 每帧流程：6 台相机（官方 `front_pixels` + 4 台 45° 斜向环形 + 1 台俯视）各渲染 RGB/深度/分割 → 反投影到世界系 →
  去掉桌面/底面和目标 marker → 只保留工作区附近 `crop_box` 内的点 → 5 mm 体素去重 → 分层随机采样（cube 保底 25%，其余机械臂）。
- pilot 实测：cube 每帧 253～346 点，cube 质心误差均值 0.56 cm；单进程约 35 ms/步，全量 10000 回合约 19 h，需要分片并行。
- 试采与可视化脚本：`scripts/checks/check_pilot.py`、`scripts/checks/render_pcd_multiview.py`。
- **等价性证明** `scripts/checks/check_equivalence.py`：把三处随机源（env 的 `np_random`、变化空间 RNG、oracle 用的全局 `np.random`）在测试里钉成相同 seed 后，
  论文配置（`swm/OGBCube-v0` + 像素 + 原版 `ExpertPolicy`）与点云配置（新相机 + wrapper + `ob_type='states'` + `add_pixels=False`）
  在 4 条回合上 26 列 qpos/qvel/action/info 逐位一致；唯一不同的列是 `pixels`/`render_time` 换成 `pcd_xyz`/`pcd_rgb`。
- 上游行为说明（两种配置相同，未改动）：swm `CubeEnv.reset` 不把 seed 传给底层环境，cube 起点/目标由变化空间用系统熵采样，
  oracle 噪声用未 seed 的全局 `np.random`。所以采集本身不可逐位复现，各分片自然互相独立。
- 全量采集：`scripts/data/collect_cube_pcd_sharded.py --shards N`（每片 10 env × 200 步，seed 派生方式同上游分片版 93b1a86），
  结束后自动用 `merge_h5_shards.py` 合并成一个 h5，格式与单次长采集相同（上游 `HDF5Writer` 的 append 校验对嵌套列有 bug，合并脚本自己按列追加）。

## 可插拔 encoder（工作 2，已完成）

契约见 `encoders/__init__.py`：输入 `(B, N, C)` 点云（C=3 或 6，由 `pcd_in_dim` / `preprocessor.use_rgb` 决定），输出 `(B, embed_dim=192)`。
接入一个新架构只需两步：把类放进 `encoders/`（或任何可 import 的路径），在 `config/train/model/lewm_pcd.yaml` 的 `encoder._target_` 指向它。

- `jepa.py` 改动 6 行：encoder 调用改为通用的 `self.encoder(x)`，观测键改为构造参数 `obs_key`（像素 `pixels`，点云 `pcd`）。
- `encoders/vit.py`：像素基线适配器，`scripts/checks/check_model_equivalence.py` 验证与上游 `jepa.py` 的嵌入/预测逐位相同（state_dict 键只多一层 `encoder.vit.` 前缀）。
- `encoders/pointnet.py`：最初的链路验证用参考实现（Linear + LayerNorm + GELU + max-pool）。
- **约定：所有 encoder 的输出维度固定为 LeWM 的 `embed_dim`=192，只调输出维度，架构其他部分不动。**

| # | encoder | 结构 | 参数量 | 状态 |
|---|---|---|---|---|
| 1 | `encoders.ManiSkillPointNet` | ManiSkill2-Learn `maniskill2_pn.py` 的 PointNet 原样复制：逐点 Conv1d(1×1, 无 bias) → LayerNorm(eps 1e-6) → ReLU ×3，沿点 max-pool，无 T-Net；输出用一层 Linear 调到 192 | 原宽度 `[64,128,512]` 主干 75,520（与上游一致）；配置默认 `[512,1024,4096]` 共 5.52M，对齐像素 ViT-tiny 的 5.50M | 冒烟训练通过 |
| 2 | `encoders.DexArtPointNet`（`model=lewm_pcd_dexart`） | DexArt `pretrain_nets.py` 的 PointNet 原样复制：逐点 Linear → GELU → Linear，无归一化，末层无激活，沿点 max-pool，trunc_normal(0.02) 初始化；输出用一层 Linear 调到 192 | smallpn 原宽度 `[64,256]` 主干 17K；配置默认 `[1080,4320]` 共 5.51M；medium `[672,672,1344,2688]`、large `[408,408,816,816,1632,1632]` 也已对齐 | 冒烟训练通过 |
| 3 | `encoders.PointPatchTransformer`（`model=lewm_pcd_ppt`） | PPRL `conf/model/ppt.yaml` 的 PointPatchTransformer 用纯 PyTorch 重写（不依赖 parllel / torch_geometric）：FPS(0.0625→75 个中心) + kNN(32) 分组，组内 mini-PointNet 两级 MLP 出 token，正弦位置编码，3 层 6 头 Transformer（mlp_ratio 3），注意力池化；末尾 Linear 调到 192。PPRL 原本在池化后拼接的状态向量不在 encoder 内 | 原 embed_dim 384 共 5.01M；配置默认 embed_dim 402 共 5.45M（0.99×ViT-tiny） | 冒烟训练通过；eval 模式下 FPS 起点固定，输出确定 |
- `utils.py` 新增 `PointCloudPreprocessor`：xyz 减工作区中心 `(0.425, 0, 0.2)` 乘 4，rgb/255，拼成 `(T, 1200, 6)` 放到 `pcd`。固定仿射，不用数据集统计量（对无序点集按点序做 z-score 无意义，且要读 29 GB）。
- `train.py` 改动 2 行：预处理器由 `data.preprocessor` 实例化；归一化跳过 `pcd*` 列。各像素数据配置各自声明原来的 `get_img_preprocessor`，行为不变。
- 训练命令：`python train.py data=ogb_pcd model=lewm_pcd subdir=<run_name>`。

## 评测（工作 3 的评测部分，已完成）

官方 `eval.py` + `config/eval/cube.yaml` 的流程原样保留（50 个起点、`set_state(qpos,qvel)` 复位、25 步后的 cube 位姿为目标、CEM 300 样本 30 轮 top-30、预算 50 步、seed 42、`terminate_at_goal` 判成功）。

- 点云版：`python eval.py --config-name=cube_pcd.yaml policy=<checkpoint 相对 checkpoints/ 的路径>`。`cube_pcd.yaml` 与 `cube.yaml` 只差：环境 `swm/OGBCubePCD-v0` + 采集用的同一个 wrapper、数据集名、`obs: pcd`、点云仿射参数、`video: false`（三联视频需要数据集里的目标图像）。
- `eval.py` 改动约 10 行：wrapper 从配置实例化；点云时用训练同一仿射走 policy 的 `process` 通道；策略用 `envs/eval_policy.py` 的 `PCDWorldModelPolicy`（把 `pcd_xyz/pcd_rgb` 拼成 `pcd`、`goal_pcd_*` 拼成 `goal`，丢掉只用于视频的 `pixels`，其余交给父类）。
- **数据布局**：官方 `eval.py` 按论文时代的 h5 布局工作（每行 `ep_idx`、扁平列名 `privileged_block_0_pos`），HEAD 写入器只给 `ep_len/ep_offset` 和嵌套分组。合并后的数据集跑一次 `scripts/data/add_eval_columns.py`：加 `ep_idx`，用硬链接把嵌套列扁平化并删掉分组，不复制数据。训练配置只用扁平键，不受影响。
- **swm HEAD 接口对齐**（像素和点云共用）：求解器在 `stable_worldmodel.planning.solver`，世界模型通过 `cost=` 传入（`config/eval/solver/*.yaml`、`eval.py` 各改一处）。
- 已验证：随机策略与真实 checkpoint（GPU 上的 CEM）在点云试采数据上跑通；像素基线的训练与评测路径同样跑通。
- 结果文件按官方逻辑写到 `$STABLEWM_HOME/<policy 的上级目录>/ogb_cube_pcd_results.txt`；hydra 日志在 `logs/eval/`。

## 全量训练（工作 3）

- 数据：`lewm_pcd_data/datasets/ogbench/cube_single_expert_pcd.h5`，10000 回合 × 201 帧，39.6 GB，已加评测布局；抽检：cube 每帧 ≥204 点，坐标全在工作区内。
- 启动：`bash scripts/train/run_train_detached.sh <model 配置> <run 名>`，脱离会话；checkpoint 在 `checkpoints/<run>/weights_epoch_N.pt`，评测时 `policy=<run>/weights_epoch_N.pt`。
- 2026-09-10 02:56 按 10 epoch 重启三个 run（22:28 那次因内存事故作废，02:38 那次因 epoch 数改正作废）。最初 2026-09-09 22:28 同时启动 `cube_pcd_ms2`（lewm_pcd）、`cube_pcd_dexart`、`cube_pcd_ppt`，配置按论文（batch 128、**10 epoch**：论文附录 E 四个环境均为 10 epoch，发布的配置文件写 100，以论文为准；AdamW 5e-5、weight decay 1e-3、bf16）。学习率调度是 stable-pretraining 的 `LinearWarmupCosineAnnealingLR` 默认：按 step 计，前 1% 步（1,279 步）从 0 线性升到 5e-5，之后余弦退火到 0（第 10 个 epoch 末），已用日志里记录的学习率核对。每 epoch 12,796 步。
- 单跑吞吐（100 步实测）：ManiSkill 2.8 it/s / 14 GB，DexArt 6.0 it/s / 3.4 GB，PPT 9.5 it/s / 9.8 GB。三个并行时约 1.6 / 2.5 / 3.5 it/s，显存合计 27 GB。
- **内存事故与修复（2026-09-10）**：容器内存上限 90 GB（cgroup），三个训练并行时被打满、页缓存被挤掉、吞吐掉到 0.2～0.7 it/s，3.5 小时没跑完一个 epoch，手动停止。根因：上游 `HDF5Writer` 每步一个 chunk，200 万步的文件每列 200 万个 chunk，HDF5 的 chunk 索引让每个进程仅读一次 `action` 列就占 12 GB，DataLoader 子进程同样增长。修复：`scripts/data/rechunk_h5.py` 把同一份数据按 20 行一个 chunk 重写（33 列逐列比对一致），单个训练进程组从 20 GB 降到 5 GB；02:38 重新启动三个训练。上游现在默认采集成 Lance，选 HDF5 是为了对齐 cube 配置的 `.h5` 名字。
- 显存对策（不改架构、不改 batch）：两个 PointNet 在训练时沿点维分块并用 activation checkpointing 重算（`chunk_points=300`），逐点 LayerNorm 与 max-pool 保证结果与整体前向数值等价（浮点分块次序不同，差异约 1e-7；点数 1200、分块 300 时实测差 0）；checkpoint 的 config 记录了 `chunk_points`，评测用同一值；ManiSkill PointNet 的 Conv1d 1×1 改为等价的逐点 Linear 布局以去掉转置（已验证与 Conv1d/LN1d 参考实现输出相同）。

## 4D 世界模型（工作 3：时间聚合）

- 结构：encoder → projector → latent1（每帧，与 3D 模型完全相同，三个 encoder 共用）→ 聚合器（前 3 帧 latent1 → latent2，同形状向量）→ predictor(latent2, 动作块) → 下一帧 latent1。规划时世界模型的状态是 latent2，CEM 在 latent2 空间算 cost，目标是目标帧及其前两帧聚合出的 latent2。
- 训练：同上游数据（4 帧片段，frameskip 5）。一个样本 = 前 3 帧聚合、用第 3 个动作块预测第 4 帧 latent1；loss = MSE(pred, e4) + 0.09 × SIGReg([latent2, e4])，SIGReg 一次调用对两个位置取平均（与上游对 4 个位置取平均同形式），λ 不动，MSE 不加系数。
- 代码：`jepa_agg.py`（`AggJEPA(JEPA)`，复用 encode / projector / action_encoder / criterion，重写 predict / rollout / get_cost；`agg_forward` 为训练前向）；`aggregators/`（`temporal` 默认：抄 ARPredictor 的位置编码 + 因果 block，不带动作，取最后 token，1～3 帧都可；`mlp`、`mean` 对照）；predictor 复用 ARPredictor 但 `num_frames=1`。
- 配置：encoder 拆成配置组 `config/train/model/encoder/{ms2,dexart,ppt}.yaml`，3D 与 4D 共用（`model/encoder=xxx`）；`model/lewm_pcd_agg.yaml`、`train/lewm_agg.yaml`、`eval/cube_pcd_agg.yaml`（= cube_pcd + `plan_config.history_len 3` + `eval.goal_window 3`）。上游文件改动：`train.py` 1 行（前向按配置选）、`eval.py` 约 10 行（取目标窗口）。
- 命令：训练 `python train.py --config-name=lewm_agg data=ogb_pcd model/encoder=ppt subdir=<run> output_model_name=<run>`；评测 `python eval.py --config-name=cube_pcd_agg.yaml policy=<run>/weights_epoch_N.pt`。`scripts/train/queue_agg_runs.sh` 在显存空出 12 GB 时依次拉起 `cube_pcd_agg_{ppt,dexart,ms2}`。
- 参数量：聚合器 3.15M（2 层，block 形状与 predictor 相同：16 头 × 64、MLP 2048），整体 ≈ encoder 5.5M + 聚合器 3.2M + predictor 10.8M + 投影头 0.9M ≈ 20.4M（3D 模型 17.2M）。
- 注意：CEM 求解器按 `solver.batch_size=1` 逐个 env 调用 `get_cost`，上游 `criterion` 的广播只对 B=1 成立，4D 模型沿用同一 criterion，行为一致。
- 验证：配置组重构后三个 3D 配置与运行中任务的配置逐项相同；4D 模型在试采数据上 CPU 冒烟训练、`cube_pcd_agg` 评测（history_len 3、目标窗口 3、CEM）均跑通。

## 逐 epoch 自动评测

`scripts/eval/auto_eval_checkpoints.sh`（脱离会话运行）盯着三个 run 的 checkpoint 目录，每出现一个 `weights_epoch_N.pt` 就用官方评测管线跑一次（`cube_pcd.yaml`：50 个起点、seed 42、CEM、预算 50；起点由 `np.random.default_rng(42)` 抽样，三个模型比的是同一批起点和目标）。
**训练优先**：同一时刻只跑一个评测且 `nice -n 10`；GPU 空闲显存不足 12 GB 时整套评测在 CPU 上跑（`MUJOCO_GL=osmesa` 渲染 + `device=cpu solver.device=cpu`，为此 `eval.py` 的 `MUJOCO_GL` 和模型设备改为可配置），完全不碰训练的 CUDA 内存，一次约 40 分钟；显存够时用 GPU，约 3 分钟。启动前要求 cgroup 内存 < 60 GB 且 anon < 40 GB，运行中若 cgroup 超过 78 GB 或评测进程 RSS 超过 20 GB 立即杀掉评测、稍后重试（最多 3 次）。
结果：`lewm_pcd_data/logs/eval/success_rates.csv`（run, epoch, success_rate, 秒数, 渲染后端, 时间），完整日志在 `logs/eval/auto/`。
官方对照（论文图 6，cube，同一评测协议）：DINO-WM 86、GCBC 84、LeWM 74、PLDM 65、GCIQL 64、GCIVL 56、随机 48。

## 代码审查（2026-09-10，两个 Opus 审查代理，只读）

结论：没有会让当前训练或已采数据出错的 bug；数据集经端到端核验干净（1 万条回合互不重复、点云与 cube 位姿时序对齐、无越界点）。已修复的防护缺失：
- 采集改为 `HDF5Writer(mode='error')`，分片启动器拒绝已存在的输出、检查分片 seed 区间不重叠、加 `--max-parallel`、关闭日志句柄；合并脚本递归枚举硬链接别名、拒绝已带 `ep_idx` 的分片；`add_eval_columns` 删组前校验同名对象；`rechunk_h5` 保留 `maxshape`、按对象地址重建别名。
- wrapper 用模型引用而非 `id()` 判断是否重建相机/geom 缓存（环境每次 reset 都重编译模型）；`PCDExpertPolicy` 对上游 `set_env` 源码做 md5 钉死，上游一变就报错。
- 评测：`use_rgb: false`（xyz-only）路径原来会在 `_prepare_info` 的 reshape 处崩，已改为按 `use_rgb` 拼接；`eval.py` 顶部显式 `import envs`；评测前把 `pcd_process` 与 checkpoint 里训练配置的 center/scale/use_rgb 做一致性断言；训练启动脚本拒绝重名 run（`FORCE=1` 覆盖），`DATA=` 可选数据配置。

结论解读时的注意事项（上游行为或设计选择，未改）：
- 训练/验证按片段随机切分，同一回合的片段两边都有，`validate/*` 是拟合指标不是泛化指标；比较 encoder 以规划成功率为准。
- PPT 训练时 FPS 起点随机（PPRL 默认），预测目标 `emb` 因而带噪声，其 pred_loss 有另外两个 encoder 没有的下界；评测时是确定的。
- **`cube_fraction: 0.25` 用渲染器的分割信息给 cube 保底 25% 的点**，这是像素基线没有的特权信息；均匀采样时 cube 自然占比约 5.7%（约 70 点）。是否为公平对比改为 0 并重采，待定。
- 深度与分割两次渲染在轮廓像素上不一致，约 0.16% 的点是"飞点"，落在物体与地面之间；重采时可加 1 像素腐蚀。
- DexArt 分支用 trunc_normal(0.02) 初始化加宽后的层，初始输出尺度比另外两个小一到两个数量级，projector 的 BatchNorm 会归一化掉，但有效学习速度可能偏低。

## 环境

- 需要一个带 CUDA torch 的专用 venv（建议 `lewm_pcd/.venv`，Python 3.10，`torch==2.8.0+cu128`，
  `pip install -e third_party/stable-worldmodel[train,env]`，`ogbench`）。GPU 是 RTX 5090，
  驱动 595.71，`mujoco 3.12.0` + EGL 无头渲染已验证可用。
- 建好之前可临时用 `/root/autodl-tmp/ms2_lewm_model_only_20260906/.venv-worldmodel`：
  依赖齐全，但 torch 是 CPU 版，只能采集和调试。

## 版本来源

| 组件 | 版本 |
|---|---|
| le-wm | github.com/lucas-maes/le-wm @ 8edfeb3（本机 `/root/autodl-tmp/lewm234d`） |
| stable-worldmodel | github.com/galilai-group/stable-worldmodel @ 6f1e499（`third_party/`） |
| ogbench | 1.2.0 |
| mujoco | 3.12.0 |

## 进度

- [x] 目录与存储规划
- [ ] 专用 venv
- [x] 1. 点云采集：`envs/cube_pcd_env.py`、`envs/pcd_wrapper.py`、`scripts/data/collect_cube_pcd.py`（2 回合 pilot 已验证，全量未跑）
- [x] 2. 模型：可插拔 encoder 接口（`encoders/`），像素基线逐位等价，PointNet 参考实现在 pilot 数据上冒烟训练通过
- [x] 3. 训练：三个候选的全量训练已于 2026-09-09 22:28 启动（`scripts/train/run_train_detached.sh`）；评测管线已完成并验证

### 2026-09-10 19:20 — 4D evaluation fix (the 54% epoch-1 number was invalid)

Two defects in the 4D *evaluation* path (training was never affected; the checkpoints are reused as is):

1. `AggJEPA.rollout` had copied the le-wm `jepa.py` action split `[H, T-H]`, which assumes the executed history
   actions are prepended to the candidates. The stable-worldmodel policy at HEAD passes strictly-future candidates
   (T = horizon) and the H-1 executed blocks separately in `info['action_history']`. With `history_len 3` the
   candidates were misaligned and only 3 blocks (15 steps) were imagined instead of 5 (25 steps, the goal).
   Fix: `rollout` is now a verbatim copy of the framework's own `stable_worldmodel.wm.lewm.LeWM.rollout` with
   two marked edits (`obs_key`; per-step prediction = window -> aggregator -> latent2 -> predictor -> next latent1);
   `get_cost` aggregates the last `window` rollout frames into latent2 before the inherited criterion.
2. The 3-frame goal window built by `eval.py` never reached the model: the branch in
   `PCDWorldModelPolicy._prepare_info` had not been written to the file. Added (4 lines).

Verification (`scripts/checks/test_agg_rollout.py`, `scripts/checks/check_history_order.py`): rollout equals a plain
reference loop for H = 1/2/3 (5 predictions each, cost = MSE of latent2 vs the goal window's latent2); in the 2-env
CPU eval the goal arrives as 3 frames whose last frame equals the World's goal frame, the first plan sees H = 1,
later plans H = 3 with 2 history blocks, predicted frames = H + 5, newest history frame = current observation.
The auto-eval loop was restarted with the 4D runs first and the `cube_pcd_agg_*` rows removed from
`success_rates.csv`. Open point: the first plan of every episode runs with a 1-frame window (framework warm-up
behaviour), which the aggregator never produced outputs for in training.

### 2026-09-10 19:25 — 4D evaluation: history seeded from the dataset (user decision)

The framework starts every evaluation episode with an empty history buffer, so the first of the two plans per
episode saw a single frame (the aggregator only ever produced outputs for 3-frame windows in training). Now
`eval.py` also reads the `(history_len - 1) * action_block = 10` dataset steps preceding each start (the same
episode's true past, frames + the actions that led into them, following `HistoryBuffer`'s conventions) and
`PCDWorldModelPolicy.get_action` fills the buffer with them before the first plan, so both plans see
start-10 / start-5 / start (then live frames). Starts closer than 10 steps to their episode's first step (2 of
the 50: step 1 and step 6) get the framework's own warm-up padding for the missing steps (copies of the oldest
real frame, zero actions). The 50 starts and the 3D evaluations are unchanged (`history_len 1`, no seed).
Verified with `scripts/checks/check_history_order.py`: first plan H = 3 with 2 history blocks, oldest frame ==
scaled dataset frame start-10, newest == current observation, 8 = 3 + 5 predicted frames.

### 2026-09-10 20:30 — first valid 4D number

`cube_pcd_agg_ppt` epoch 1 with the corrected evaluation (framework rollout + 3-frame goal + dataset-seeded
history): **72 %** success (50 starts, seed 42, budget 50), versus 54 % before the fixes, 48 % random, 72 % for the
3D `cube_pcd_ppt` at epoch 1 and 74 % for the paper's pixel LeWM. Training metrics of the 4D run are healthy
(train pred_loss 0.028 → 0.008 and SIGReg 2.31 → 0.97 over epochs 1–7; val pred_loss 0.0087 → 0.0027, val SIGReg
1.89 → 1.48, all monotone). Later epochs are evaluated automatically (`logs/eval/success_rates.csv`).
