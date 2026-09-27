# CortexBench 跑分（MuJoCo 模仿学习部分）

用 CortexBench 的原协议（eai-vc 官方代码，commit 76fe35e）给 LeWM 编码器跑分，分数与 VC-1 论文表 8 同一含义：冻结的编码器 + 行为克隆策略，在专家示范上训练，看任务成功率。
LeWM 只在它的世界模型训练过的任务上评测（现在是 MetaWorld bin-picking）；2D 编码器输入图像，3D 编码器输入同一台相机的点云。

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

## 安装与数据

| 内容 | 位置 |
|---|---|
| conda 环境 `cortexbench` | 数据盘的 conda 目录；lewm_pcd 环境的克隆 + 上面的依赖。eai-vc 的 `vc_models`、`mujoco_vc`、`metaworld`、`mjrl`、`mj_envs`、`dmc2gym`、`mujoco-py` 以 editable 方式装在 `third_party/eai-vc` |
| eai-vc（76fe35e + 子模块） | `lewm_pcd/third_party/eai-vc`（不进 git）；VC-1 权重在其中的 `vc_models/src/model_ckpts/` |
| MuJoCo 2.0 + mjkey | `lewm_pcd/third_party/mujoco/`（不进 git） |
| 官方示范 | `$STABLEWM_HOME/datasets/cortexbench/{adroit,metaworld,dmc}-expert-v1.0/`，来自 `https://dl.fbaipublicfiles.com/eai-vc/adroit-expert-v1.0.zip`、`.../eai-vc/mujoco_vil_datasets/{metaworld,dmc}-expert-v1.0.zip`（`unpack_suite.py` 解压） |
| 3D 示范 | `$STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0-pcd/` |
| R3M / VIP / CLIP 权重 | `$STABLEWM_HOME/checkpoints/cortexbench_baselines/` |
| 运行目录、结果 | `$STABLEWM_HOME/logs/cortexbench/runs/<模型>/<任务>/s<种子>/`，`$STABLEWM_HOME/logs/cortexbench/results/scores.{md,json}` |

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

## 文件

| 文件 | 内容 |
|---|---|
| `env.sh` | 运行环境：MuJoCo 2.0 路径、EGL、`PYTHONPATH`（lewm_pcd + 本目录）、`STABLEWM_HOME` |
| `lewm_cortex.py` | LeWM 编码器的加载函数（2D：`load_lewm`，3D：`load_lewm_pcd`），按 checkpoint 的 `config.json` + `weights_epoch_N.pt` 重建 encoder + projector |
| `baselines.py` | 论文里另外几个基线的加载函数（官方权重）：R3M、VIP（ResNet-50 全局池化 2048 维）、CLIP ViT-B/16（CLS） |
| `conf/model/*.yaml` | 编码器配置：`lewm_mw_binpick_{pix,pcd}` 和上述基线 |
| `run_bc.py`、`run_bc3d.py` | 一次运行（官方 `hydra_launcher.py` + 上面的补丁；3D 版再换成点云） |
| `pcd_fork.py`、`convert_demos_3d.py` | CortexBench MetaWorld 环境里的点云；官方示范转成点云示范 |
| `grid.py`、`summarize.py` | 批量作业；汇总成表 |
| `unpack_suite.py` | 解压官方示范（MetaWorld 的 zip 里每个文件存了两份，只取一份） |
