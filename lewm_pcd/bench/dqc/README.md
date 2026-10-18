# DQC on OGBench puzzle with frozen LeWM inputs

小规模那次评测 puzzle-3x3 上 LeWM 编码器的做法（3x3 全量改用 GCIQL，见 [`../gciql/README.md`](../gciql/README.md)；这里的潜变量提取 `extract_feats.py` 和评测观测 `lewm_obs.py` 两边共用）；全量的 puzzle-4x4、4x5 也用它，见文末。编码器冻结，用在 puzzle-3x3 上做成功过的离线 RL（DQC）学策略，输入换成 LeWM 的潜变量加 9 个灯的真值。结果见 [`results_ogbench.md`](../../results_ogbench.md)。

## 协议

- **算法：**DQC（Decoupled Q-chunking，Li、Park、Levine，arXiv 2512.10926），官方代码 github.com/ColinQiyangLi/dqc，提交 df89825，原样放在 `third_party/dqc`（不进 git）。离线、目标条件：从数据里学一个无条件的动作块生成器（flow），学目标条件的价值函数，执行时每次采 32 个候选动作块、挑价值最高的。
- **超参：**官方代码默认值，加上 HF `Ievan/transformer_puzzle_3x3_26_04` 里在 puzzle-3x3 上跑到 96–100 % 的那组（h=40、h_a=5、kappa_b 0.93、kappa_d 0.8、q_agg min；DQC 论文本身没跑 3x3）。100 万步，batch 4096，种子 100001。
- **数据：**OGBench 官方 `puzzle-3x3-play-v0`（1000 回合 × 1001 步），放在 `$STABLEWM_HOME/datasets/ogbench/official/`。目标是 9 个灯的目标状态（`oraclerep`）。
- **输入（`run_dqc.py --obs`）：**`state` 原版 55 维状态向量（复现）；`lewm2d` / `lewm3d` 冻结的 LeWM 2D / 3D 投影头输出 192 维 + 9 个灯的真值；`lewm2d_nolight` / `lewm3d_nolight` 只有 192 维潜变量。
- **潜变量（`extract_feats.py`）：**官方数据每一步都存了 qpos / qvel / 按钮状态。把仿真器设回去，用训练 LeWM 时同一套渲染（`envs/ogb_puzzle_env.py`：front_pixels、224、不透明手臂；3D 用同一台相机的深度图取 1200 个点，每帧的抽点种子固定）重画，过冻结编码器（`lewm_obs.py`，加载和预处理同 CortexBench 的 `lewm_cortex.py`），只存潜变量。
- **评测：**官方 5 个任务，每 10 万步各评 50 回合。LeWM 输入在策略要动作时用同一套相机和编码器现算（`lewm_obs.LeWMObs`）。

## 环境

conda 环境 `lewm_dqc`：`lewm_mw` 的克隆加上 `jax[cuda12]==0.6.2` 和 `third_party/dqc/requirements.txt`。`PYTHONPATH` 里放 `lewm_pcd` 和 `third_party/stable-worldmodel`；DQC 自己也有叫 `envs`、`utils` 的包，`run_dqc.py` 把 `third_party/dqc` 放在 `sys.path` 最前面，`lewm_obs.py` 按文件路径加载 `envs/ogb_puzzle_env.py`。

## 运行

`scripts/run_puzzle_pipeline.sh` 的第 3 步用 `extract_feats.py` 算潜变量，第 4 步现在跑 GCIQL。单独跑 DQC：

```bash
cd bench/dqc
$DQC_PY extract_feats.py --ckpt-2d $STABLEWM_HOME/checkpoints/puzzle_3x3_pix --ckpt-3d $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \
    --out $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --split train --workers 6
$DQC_PY run_dqc.py --obs lewm3d --ckpt $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \
    --feats_dir $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --run_dir $STABLEWM_HOME/logs/dqc/puzzle_3x3/lewm3d \
    --agent.num_qs=2 --agent.policy_chunk_size=5 --agent.backup_horizon=40 --agent.use_chunk_critic=True \
    --agent.distill_method=expectile --agent.implicit_backup_type=quantile --agent.q_agg=min --agent.kappa_b=0.93 \
    --agent.kappa_d=0.8 --offline_steps=1000000 --eval_interval=100000 --eval_episodes=50 --save_every=25000
```

`run_dqc.py` 每 2.5 万步存一次完整状态（`resume.pkl`），收到 SIGTERM 存档后退出，再跑同一条命令从存档接着训。`check_eval_obs.py` 核对评测时现算的观测和训练集的观测对同一仿真状态是否一致。

## puzzle-4x4、4x5（全量）

同上，换三处：

- **棋盘：**`extract_feats.py`、`check_eval_obs.py` 加 `--env-type 4x4` / `4x5`；`run_dqc.py` 给 `--env_name puzzle-4x5-play-oraclerep-v0 --dataset_path .../official/puzzle-4x5-play-v0.npz`。输入 = 192 维潜变量 + 16 / 20 个灯的真值。
- **数据：**OGBench 官方 `puzzle-4x4-play-v0`（1000 回合）、`puzzle-4x5-play-v0`（3000 回合）及各自的 `-val`，放在 `$STABLEWM_HOME/datasets/ogbench/official/`。
- **超参：**DQC 论文自己给 puzzle-4x5 的那组（`third_party/dqc/experiments/reproduce.py`，标准数据量）：h=25、h_a=5、kappa_b 0.9、kappa_d 0.5、q_agg mean，其余官方默认值；DQC 没跑过 4x4，用同一组。

`scripts/run_puzzle_pipeline.sh` 用 `ENV_TYPE=4x4 SHARDS=4` / `ENV_TYPE=4x5 N=3000` 从采集一路跑到 DQC（`logs/dqc/puzzle_4x{4,5}/`）。

## JEPA 类基线（puzzle 3×3、4×4、4×5）

MetaWorld、scene 上跑过的三个基线，在三个棋盘上照 LeWM 的 DQC 流程跑（`scripts/run_puzzle_baselines.sh`）：

- **编码器：**LeJEPA、PLDM 用该棋盘 LeWM 的同一份数据、按各自官方配置训 10 个 epoch（同 scene）；V-JEPA 2.1 冻结原版。都按 CortexBench 的加载方式（`bench/cortexbench/jepa_baselines.py`，各自的预处理：V-JEPA 2.1 用 PIL 双三次放大到 384，另外两个 224）读 LeWM 2D 看到的同一帧（`lewm_obs.BaselineEncoder`）。
- **潜变量：**`extract_feats_baselines.py`，一次渲染供多个编码器用，每个编码器各自的完成标记，可续跑。
- **DQC：**`run_dqc.py --obs vjepa21 / lejepa / pldm --ckpt <权重>`，输入 = 该编码器的潜变量 + 灯的真值；超参、数据、评测与同棋盘 LeWM 的运行相同。评测时现算的观测（`lewm_obs.BaselineObs`）和训练集的潜变量对同一仿真状态一致（余弦 1.00000）。
- **汇总：**`summarize_dqc.py`：每个输入按 OGBench 口径（第 80、90、100 万步平均）和训练中评测的最高值。
