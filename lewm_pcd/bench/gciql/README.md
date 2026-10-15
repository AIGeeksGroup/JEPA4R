# GCIQL on OGBench puzzle-3x3 with frozen LeWM inputs

全量 puzzle-3x3 的评测：OGBench 官方的 GCIQL，按 OGBench 自己跑 `puzzle-3x3-play-v0` 的方式原样跑，只换一处：观测从原版 55 维状态换成冻结的 LeWM 潜变量加 9 个灯的真值。结果见 [`results_ogbench.md`](../../results_ogbench.md)。

## 协议

- **算法和代码：**GCIQL（goal-conditioned implicit Q-learning），OGBench 官方实现 github.com/seohongpark/ogbench 提交 a977253 的 `impls/`（`agents/gciql.py`、`utils/datasets.py` 的 `GCDataset`），原样放在 `third_party/ogbench`（不进 git），一行没改。
- **超参：**`gciql.py` 的默认值，加上官方 `impls/hyperparameters.sh` 里 `puzzle-3x3-play-v0` 那一行的 `--agent.alpha=1.0`：batch 1024，lr 3e-4，MLP 512×3（layer norm），γ 0.99，expectile 0.9，策略 DDPG+BC，价值目标 20 % 当前状态、50 % 同轨迹后面的状态（几何采样）、30 % 随机状态，策略目标取同轨迹后面的状态；100 万步，种子 0（官方 `main.py` 默认）。
- **数据：**OGBench 官方 `puzzle-3x3-play-v0`（1000 回合 × 1001 步）。
- **观测（`run_gciql.py --obs`，唯一的改动）：**`state` 官方原版 55 维状态（复现用）；`lewm2d` / `lewm3d` 冻结的 LeWM 2D / 3D 投影头输出 192 维 + 9 个灯的真值；`lewm2d_nolight` / `lewm3d_nolight` 只有 192 维潜变量。训练集的潜变量用 `bench/dqc/extract_feats.py` 算好的（和 DQC 共用）。
- **目标：**和官方一样是"目标状态的观测"。训练时 `GCDataset` 取目标状态的观测，换了观测后自然就是目标状态的潜变量 + 灯；评测时 OGBench 在 reset 里摆出目标场景（目标灯态、手臂随机、走几步稳定）再算观测，`GoalCapture` 在同一时刻用 LeWM 观测函数取这个场景的潜变量 + 灯。用原版状态观测时，这样取到的目标和 OGBench 自己的 `info['goal']` 在 5 个任务上逐位相同（`check_goal`，每次运行开头都检查）。
- **评测：**官方 `main.py` 的做法：第 1 步和之后每 10 万步，官方 5 个任务各 50 回合（`--eval_episodes=50`，同 `hyperparameters.sh`），温度 0。LeWM 观测每一步用同一套相机和编码器现算（`bench/dqc/lewm_obs.LeWMObs`）。报告按 OGBench 的口径：第 80、90、100 万步三次评测的平均。
- **和方法无关的：**不用 wandb（指标写 jsonl），不算验证集损失，评测在 GPU 上跑（官方默认把同一个模型拷到 CPU 上评测），每 2.5 万步存一次完整状态（`resume.pkl`），收到 SIGTERM 存档后退出，再跑同一条命令接着训。

之前一版（提交 02b8a6f 的 `run_gciql.py`）把目标改成只有 9 个灯（OGBench 的 oraclerep，为此改了网络构建里目标维度那一行），不是官方的用法；它的两个运行在 `logs/gciql/puzzle_3x3/`，第 10、20 万步都是 0 %。

## 环境

conda 环境 `lewm_dqc`（已有 jax、flax、distrax、ml_collections、ogbench 1.2.0）。`PYTHONPATH` 里放 `lewm_pcd` 和 `third_party/stable-worldmodel`；`run_gciql.py` 把 `third_party/ogbench/impls` 放在 `sys.path` 最前面（它的 `agents`、`utils` 包），`lewm_obs.py` 按文件路径加载 `envs/ogb_puzzle_env.py`。

## 运行

全量由 `scripts/run_puzzle_pipeline.sh` 的第 4 步跑（`lewm2d`、`lewm3d` → `logs/gciql/puzzle_3x3_official/<输入>`）。单独跑：

```bash
cd bench/gciql
$DQC_PY run_gciql.py --obs lewm3d --ckpt $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \
    --feats_dir $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 \
    --run_dir $STABLEWM_HOME/logs/gciql/puzzle_3x3_official/lewm3d --agent.alpha=1.0
```
