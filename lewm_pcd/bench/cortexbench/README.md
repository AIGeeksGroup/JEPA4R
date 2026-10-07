# CortexBench 跑分（MuJoCo 模仿学习部分、TriFinger）

用 CortexBench 的原协议（eai-vc 官方代码，commit 76fe35e）给 LeWM 编码器跑分，分数与 VC-1 论文表 8 同一含义：冻结的编码器 + 行为克隆策略，在专家示范上训练，看任务成功率。
LeWM 只在它的世界模型训练过的任务上评测（MetaWorld 的五个任务，换成 RM75 机械臂的 bin-picking、button-press-topdown、drawer-open 和 assembly，见“RM75 上的任务”一节，以及 TriFinger 的 reach_cube 和 move_cube，见“TriFinger”一节）；2D 编码器输入图像，3D 编码器输入同一台相机的点云。

## 协议（官方代码原样执行）

- 编码器冻结，每帧输出一个向量；策略是 mjrl `BatchNormMLP`（256, 256, 256），输入 = 3 帧历史的 FLARE 融合 + 本体感受（`obs[:4]`）。
- BC + MSE，Adam lr 1e-3，batch 256，100 个 epoch；第 5、10、…、95、99 个 epoch 各评测一次，每次 25 个回合（MetaWorld），评测种子固定。
- 示范：MetaWorld 每个任务 25 条官方示范（256×256 渲染，相机 `top_cap2`）。
- 单次运行的分数 = 20 次评测中的最高成功率；每个任务跑 3 个种子（100/200/300）取平均。`summarize.py` 同时列出最后一次评测的值（不做 20 选 1）。

## 与官方的差异

1. **运行环境。** RTX 5090（sm_120）跑不了官方的 py3.8 + torch 1.13，换成 Python 3.10 + torch 2.8 cu128；其余依赖按官方版本钉住：MuJoCo 2.0 + mujoco-py 2.0.2.13（GPU/EGL 构建）、gym 0.23.1、numpy 1.23.4、timm 0.6.11、dm-control 1.0.5 + mujoco 2.2.1、pyparsing 3.1.4。
2. **`run_bc.py` 的三个运行时补丁**（不改变报告的分数）：
   - 编码器配置先在本目录 `conf/model/<名字>.yaml` 找（LeWM、R3M/VIP/CLIP、随机 ViT-B），找不到再用 vc_models 自带的（VC-1）；加载仍走官方的 `load_pretrained_model`。
   - 不用 wandb，`eval/*` 等记录写进运行目录的 `metrics.jsonl`。
   - 官方每次评测后会再做一遍只用于记录 `train/*` 的 rollout，这里直接复用刚算好的评测路径。评测回合每回合都重新设种子，BC 训练不消耗 numpy 的随机数，所以评测回合、训练出的策略和 `eval/*` 都不变。
3. **LeWM 2D 的预处理。** 256 的帧双三次缩放到 224 再做 ImageNet 归一化（LeWM 训练时看的就是完整的 224 画面；官方 VC-1 的中心裁剪会缩小视野）。编码器输出 = encoder + projector 的 192 维潜变量，也就是 LeWM 预测器和规划器用的那个空间。
4. **LeWM 3D：点云代替图像**（`run_bc3d.py`），其余不变：
   - 示范：官方示范逐条重放（见下节），每一步换成同一台 `top_cap2` 相机的点云，(500, 1200, 6) = 世界坐标 xyz（米）+ rgb。
   - 评测：环境每一步返回同一台相机的点云（`pcd_fork.MWPointClouds`，与训练数据的设置相同：单深度相机反投影、裁剪框、5 mm 体素去重、1200 点）。点云的随机抽样按回合种子设种子，评测确定。
   - 前景用模拟器的分割（带标签的特权信息），这一点与图像输入不完全可比。

**图像输入的评测不是逐位可复现的。** CortexBench 的 RGB 渲染（mujoco-py 2.0、多重采样、EGL）对同一个状态偶尔会有几个像素差 1 个灰度（实测：同一回合两个进程，第 9 步 4 个像素差 1），闭环里放大成不同的轨迹。所以同一代码、同一配置重跑，图像模型的单次评测结果会变（bin-picking 第 5 轮评测，25 个回合成功 8 或 12 个），而 BC 训练逐位相同。所有图像输入的模型（VC-1、R3M、LeWM 2D 等）都带这份噪声。3D 的点云用关掉多重采样的深度 + 分割渲染，逐位可复现。

## 3D 示范（`convert_demos_3d.py`）

- 第 i 条官方示范（i = 0..24）的种子是 `12345 + i`（从数据反搜得到；eai-vc 注释里的 54321 不对）。每条示范新建环境（`goal-observable`、`_freeze_rand_vec=False`，与 CortexBench 相同），`seed` 后 `reset`，复位后第一个动作之前先渲染一帧，再依次执行记录的动作。
- 点云需要关掉多重采样（否则边缘像素的分割标签会被解成别的零件），而 mujoco-py 2.0 同一进程里第二个渲染上下文会弄坏第一个，所以点云单独一个进程渲染。物理不依赖渲染，两遍的 qpos/qvel 逐位相同。
- 前景 = 不与世界固连的物体（机械臂、夹爪、物体）+ 每回合复位时重新摆放的固定件（assembly 的柱子、button-press 的按钮盒、drawer-open 的抽屉柜）；桌面、围墙、箱子、机器人底座是背景。
- 已验证（bin-picking，25 条 × 500 步）：动作与官方逐位相同；RGB 重放与官方图像最大差 1/255；RGB 与点云两遍的状态逐位相同；点云文件与评测时环境给出的点云逐位相同。

## RM75 上的任务（`run_bc_rm75.py`）

MetaWorld 的 bin-picking、button-press-topdown、drawer-open 和 assembly 换成实验室的 RM75-B 机械臂 + EG2-4C2 夹爪（`envs/metaworld/rm75`），按同一协议评测。任务名 `rm75-<任务>-v2-goal-observable`，在 `grid.py` 里属于 suite `metaworld_rm75`，任务简称 `rm75-bin-picking`、`rm75-button-press-topdown`、`rm75-drawer-open`、`rm75-assembly`。

- **环境。** RM75 环境需要 MuJoCo 3 + metaworld 3.0（`lewm_mw` 环境），评测框架用的是 MuJoCo 2.0。所以 `run_bc_rm75.py` 把环境放在子进程里运行（`rm75_server.py`，用 `MW_PY` 指定的 python 启动）；框架这边是一个代理环境，转发 reset、step、seed 和每一帧。其余都是官方流程：256×256 的 `top_cap2` 画面（3D 是同一台相机的点云，设置与训练数据相同，按 224×224 渲染，与训练数据和 Sawyer 的 3D 评测一致；2026-10-04 之前跑的 bin-picking、button-press-topdown、drawer-open 的 3D 结果用的是按 256×256 渲染的点云，见 `results_metaworld.md` 的“三处修复”）、FLARE、BC、20 次评测 × 25 回合、第 500 步的 `info['success']`。2D 和 3D 用同一个入口。
- **布局。** 环境复位用 `MWTaskEnv.reset(seed)`（`rm75_env.py`）：每个种子仍是一个随机布局，但同一个种子对应的布局与 metaworld 自带的 `seed(s)` + `reset()` 不同。
- **示范**（`make_rm75_demos.py <任务>`）。Sawyer 的官方示范不能用在 RM75 上，所以用 metaworld 的专家脚本在 RM75 上重新录 25 条：bin-picking、drawer-open 和 assembly 用改成 RM75 版的脚本（`envs/metaworld/rm75/expert.py`），button-press-topdown 用原脚本、不改。
  - 种子 12345 + i，与官方相同；只收第 500 步成功的回合。四个任务现有的 25 条都是种子 12345–12369，没有跳过。bin-picking、button-press-topdown、drawer-open 的 3D 示范在 2026-10-04 按 224 的点云重录过（观测和动作与原示范逐位相同），2D 示范没动。
  - 动作按脚本输出原样记录，不截断（环境执行时才截断到 [-1, 1]），与官方示范相同。
  - 2D 和 3D 示范来自同一批回合，格式与官方相同。
- **专家成功率**（第 500 步成功）：
  - bin-picking：metaworld 自带的布局（种子 12345–13344）1000/1000，Sawyer 用原脚本也是 1000/1000，原脚本的规则不改直接用在 RM75 上只有 49.7%；示范和评测实际用的布局（种子 12345–12504）159/160，种子 12404 失败。
  - button-press-topdown（原脚本）：metaworld 自带的布局（种子 12345–13344）989/1000，Sawyer 1000/1000；示范和评测实际用的布局（同样的种子）984/1000。16 个失败局结束时手都偏离按钮中心 17–21 mm，按钮停在差一点到成功线的深度。抽查的失败局里，手是在下压过程中横向滑开的；两指对按钮的压力约 130 N，成功局约 190 N（只碰按钮，不碰箱子）。先对准到 1 cm 以内再下压不能减少失败（实测），横向滑开的根本原因还没查明。
  - drawer-open（原脚本只把手的高度从把手下方 2 cm 改到上方 2.8 cm）：示范和评测实际用的布局 998/1000，metaworld 自带的布局 994/1000；原脚本在 RM75 上 70/200（每一局都中途成功过，之后指垫从杆上滑脱、抽屉离开止挡），Sawyer 200/200。
  - assembly（原脚本改两处：夹爪只在抓取高度 5 mm 以内才合拢，原脚本从把手上方 12 cm 就开始合拢，3.35 cm 宽的指垫会落在把手上面；指垫夹紧之后才抬起，原脚本手一到把手 5 cm 以内就抬，20 N 的 EG2-4C2 这时还没合拢）：示范和评测实际用的布局（种子 12345–13344）1000/1000，metaworld 自带的布局 1000/1000；原脚本在 RM75 上两种布局都是 0/200，Sawyer 200/200。

## TriFinger（`run_tf_bc.py`）

CortexBench 的 TriFinger 两个任务（reach_cube、move_cube）用 eai-vc 的 `cortexbench/trifinger_vc`，和上面的 MuJoCo 部分是两套代码。LeWM 的 TriFinger 编码器用 `envs/trifinger` 采的数据训练（`scripts/run_tf_pipeline.sh`），结果见 `results_trifinger.md`。

**协议（官方代码原样执行）。** `bc_experiments/train_bc.py`，配置 `bc_default`，加上 trifinger_vc README 的设置 `algo.freeze_pretrained_rep=true rep_to_policy=none`。
- 策略是不做输入归一化的 MLP（2000, 2000，ReLU）。输入是指尖位置（9 维）加当前帧的编码；move 再加目标帧（示范最后一帧）的编码。reach 只动第一根手指。
- AdamW，lr 1e-4，batch 32，100 条训练示范全部用上。reach 训练 100 个 epoch，每 5 个评测一次（共 21 次）；move 训练 1000 个 epoch，每 25 个评测一次（共 41 次）。每次评测从 25 条测试示范（和 25 条训练示范）的起点和目标做 rollout。
- 分数是 scaled success：1 − 结束时到目标的距离 / 开始时的距离，下限 0（move 看方块到目标位置，reach 看指尖到方块）。单次运行取所有评测中测试示范平均分的最高值（`diff-110_max_avg_scaled_reward`），3 个种子（100/200/300）取平均。

**与官方的差异。**
1. **运行环境。** conda 环境 `lewm_tf`：`lewm_pcd` 环境的克隆（Python 3.10 + torch 2.8 cu128），装 eai-vc 子模块钉住的 trifinger_simulation（ClaireLC/trifinger_simulation @ b3629c6，1.3.0）、pin 2.6.20、pybullet 3.2.7、gym 0.26.2。trifinger_vc 和 vc_models 不安装，从 `third_party/eai-vc` 放进 `PYTHONPATH`。
2. **`run_tf_bc.py` 的运行时补丁。**
   - `algo.pretrained_rep=lewm_tf_pix / lewm_tf_pcd` 用 `lewm_cortex.py` 加载 LeWM 的 encoder + projector（192 维，与 MuJoCo 部分相同）。
   - 编码器保持 eval 模式。官方循环每个 epoch 都调用 `encoder.train()`，LeWM projector 里的 BatchNorm 会因此在示范上更新统计量（VC-1 没有 BatchNorm）；MuJoCo 部分本来就在 eval 模式下编码。
   - 只改记录，不改分数：不写逐回合的 npz / png / gif（每次运行几 GB），camera180 / camera300 不渲染（只有 camera60 进策略）；每次评测的平均分和历史最高写进 `metrics.jsonl`。
3. **LeWM 2D 的预处理。** 帧双三次缩放（抗锯齿）到 224 再做 ImageNet 归一化，与 LeWM 训练时一致。
4. **LeWM 3D：camera60 的点云代替图像。** 用 `envs.trifinger.env.Camera60PCD`，与训练数据是同一套代码：224×224 深度渲染、1200 点、5 mm 体素、方块占 25%。
   - 示范：每一行的关节角和方块位姿放进 `MoveCubeEnv` 渲染，在所有运行之前由 `run_tf_bc.py prep` 缓存到示范目录的 `assets/data/pcd/`。
   - 示范里的方块位姿来自 10 fps 相机的画面，可能比同一行的关节角早最多 0.1 s，所以 move 的 3D 示范点云里方块可能落后手指最多 0.1 s；reach 的方块不动，不受影响。这一点没有修。
   - 评测：每次观测都按当前状态渲染点云。2D 用的是平台相机最近的一帧，最多早 0.1 s。
   - 前景用模拟器的分割，只留手指的活动连杆和方块，这一点与图像输入不完全可比。

## 安装与数据

| 内容 | 位置 |
|---|---|
| conda 环境 `cortexbench` | 数据盘的 conda 目录；lewm_pcd 环境的克隆 + 上面的依赖。eai-vc 的 `vc_models`、`mujoco_vc`、`metaworld`、`mjrl`、`mj_envs`、`dmc2gym`、`mujoco-py` 以 editable 方式装在 `third_party/eai-vc` |
| eai-vc（76fe35e + 子模块） | `lewm_pcd/third_party/eai-vc`（不进 git）；VC-1 权重在其中的 `vc_models/src/model_ckpts/` |
| MuJoCo 2.0 + mjkey | `lewm_pcd/third_party/mujoco/`（不进 git） |
| 官方示范 | `$STABLEWM_HOME/datasets/cortexbench/{adroit,metaworld,dmc}-expert-v1.0/`，来自 `https://dl.fbaipublicfiles.com/eai-vc/adroit-expert-v1.0.zip`、`.../eai-vc/mujoco_vil_datasets/{metaworld,dmc}-expert-v1.0.zip`（`unpack_suite.py` 解压） |
| 3D 示范 | `$STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0-pcd/` |
| RM75 示范（2D、3D） | `$STABLEWM_HOME/datasets/cortexbench/metaworld-rm75-expert{,-pcd}/`（`make_rm75_demos.py`） |
| R3M / VIP / CLIP 权重 | `$STABLEWM_HOME/checkpoints/cortexbench_baselines/` |
| 运行目录、结果 | `$STABLEWM_HOME/logs/cortexbench/runs/<模型>/<任务>/s<种子>/`，`$STABLEWM_HOME/logs/cortexbench/results/scores.{md,json}` |
| conda 环境 `lewm_tf`（TriFinger） | 数据盘的 conda 目录；`lewm_pcd` 环境的克隆 + pin 2.6.20、pybullet 3.2.7、gym 0.26.2、trifinger_simulation（eai-vc 子模块 `third_party/trifinger_simulation`，ClaireLC fork @ b3629c6，`pip install` 非 editable） |
| TriFinger 示范 | `$STABLEWM_HOME/datasets/cortexbench/trifinger/assets/`：`bc_demos/`（官方训练 / 测试划分，从 `trifinger_vc/assets/bc_demos` 复制）、`data/trifinger-demos/`（`https://dl.fbaipublicfiles.com/eai-vc/trifinger_demos.zip`，缺失时 `run_tf_bc.py prep` 用官方函数下载）、`data/pcd/`（3D 示范点云缓存） |
| TriFinger 运行目录 | `$STABLEWM_HOME/logs/cortexbench/trifinger/<模型>/<任务>/s<种子>/`（`metrics.jsonl`）和同名 `.log` |

`STABLEWM_HOME` 默认是 checkout 根目录下的 `lewm_pcd_data/`（`env.sh` 按本文件的位置算出）。

## 运行

```bash
cd lewm_pcd/bench/cortexbench && conda activate cortexbench && source env.sh
mkdir -p $STABLEWM_HOME/logs/cortexbench
setsid nohup $PY grid.py --models lewm_mw_binpick_pix,lewm_mw_binpick_pcd --tasks bin-picking --slots 6 \
    >> $STABLEWM_HOME/logs/cortexbench/grid.log 2>&1 < /dev/null &
$PY summarize.py
```

`grid.py` 可以重复启动：已完成的作业跳过，正在跑的由新的调度进程接管。每个 MetaWorld 作业约 45–60 分钟（8 个并发时）。
单个作业：`$PY run_bc.py --config-name Metaworld_BC_config env=bin-picking-v2-goal-observable embedding=<名字> seed=100 data_dir=$STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0/ wandb.project=local wandb.entity=local hydra.run.dir=<目录>`（3D 用 `run_bc3d.py` 和 `metaworld-expert-v1.0-pcd/`）。

RM75 上的任务：先录示范（每个任务一次），再跑作业；两步都要 `MW_PY` = `lewm_mw` 环境的 python。

```bash
export MW_PY=<lewm_mw python>
$MW_PY make_rm75_demos.py bin-picking
setsid nohup $PY grid.py --models lewm_mw_binpick_rm75_pix,lewm_mw_binpick_rm75_pcd --tasks rm75-bin-picking --slots 6 \
    >> $STABLEWM_HOME/logs/cortexbench/grid.log 2>&1 < /dev/null &
```

其他任务同理，例如 button-press-topdown：`make_rm75_demos.py button-press-topdown`，模型 `lewm_mw_button_rm75_{pix,pcd}`，任务 `rm75-button-press-topdown`（drawer-open：`drawer-open`、`lewm_mw_drawer_rm75_*`、`rm75-drawer-open`；assembly：`assembly`、`lewm_mw_assembly_rm75_*`、`rm75-assembly`）。

TriFinger（`lewm_tf` 环境的 python）：

```bash
cd lewm_pcd
STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_tf python> TF_EPOCH=10 bash bench/cortexbench/run_tf_grid.sh
STABLEWM_HOME=<checkout>/lewm_pcd_data python bench/cortexbench/summarize_tf.py
```

`run_tf_grid.sh` 先准备示范目录（复制划分文件、缺示范时下载、缓存 3D 示范点云），再同时启动 12 个运行（2 个任务 × 2 个编码器 × 3 个种子）。已完成的运行跳过，未完成的运行目录拒绝覆盖。编码器 checkpoint 默认是 `$STABLEWM_HOME/checkpoints/tf_cube_{pix,pcd_dexart}`（可用 `TF_CKPT_PIX` / `TF_CKPT_PCD` 改），epoch 由 `TF_EPOCH` 指定。`results_trifinger.md` 那次 12 个运行同时跑，共用 2 小时 17 分钟。

## 文件

| 文件 | 内容 |
|---|---|
| `env.sh` | 运行环境：MuJoCo 2.0 路径、EGL、`PYTHONPATH`（lewm_pcd + 本目录）、`STABLEWM_HOME` |
| `lewm_cortex.py` | LeWM 编码器的加载函数（2D：`load_lewm`，3D：`load_lewm_pcd`），按 checkpoint 的 `config.json` + `weights_epoch_N.pt` 重建 encoder + projector |
| `baselines.py` | 论文里另外几个基线的加载函数（官方权重）：R3M、VIP（ResNet-50 全局池化 2048 维）、CLIP ViT-B/16（CLS） |
| `conf/model/*.yaml` | 编码器配置：LeWM 各任务的 `lewm_mw_<任务>_{pix,pcd}`（RM75：`lewm_mw_{binpick,button,drawer,assembly}_rm75_{pix,pcd}`）和上述基线 |
| `run_bc.py`、`run_bc3d.py` | 一次运行（官方 `hydra_launcher.py` + 上面的补丁；3D 版再换成点云） |
| `run_bc_rm75.py`、`rm75_server.py`、`rm75_env.py` | RM75 上的一次运行：官方流程 + 子进程里的 RM75 环境 |
| `make_rm75_demos.py` | RM75 的示范 |
| `pcd_fork.py`、`convert_demos_3d.py` | CortexBench MetaWorld 环境里的点云；官方示范转成点云示范 |
| `grid.py`、`summarize.py` | 批量作业；汇总成表 |
| `unpack_suite.py` | 解压官方示范（MetaWorld 的 zip 里每个文件存了两份，只取一份） |
| `run_tf_bc.py` | TriFinger 的一次运行（官方 `train_bc.py` + 上面的补丁）；`prep` 准备示范和 3D 示范点云 |
| `run_tf_grid.sh`、`summarize_tf.py` | TriFinger 的批量运行；汇总成表 |
