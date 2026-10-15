# GCIQL on OGBench puzzle-3x3 with frozen LeWM inputs

全量 puzzle-3x3 的评测：LeWM 编码器冻结，用 OGBench 自己的目标条件离线 RL 基线 GCIQL 学策略，输入是 LeWM 的潜变量加 9 个灯的真值。设置和小规模那次的 DQC（[`../dqc/README.md`](../dqc/README.md)）一样，只把算法从 DQC 换成 GCIQL。结果见 [`results_ogbench.md`](../../results_ogbench.md)。

## 协议

- **算法：**GCIQL（goal-conditioned implicit Q-learning），OGBench 官方实现 github.com/seohongpark/ogbench 提交 a977253 的 `impls/agents/gciql.py`，原样放在 `third_party/ogbench`（不进 git）。从数据里学目标条件的 Q 和 V（V 用 expectile 0.9 逼近数据内动作的 Q 上沿），策略用 DDPG+BC（最大化 Q，加行为克隆项）。
- **超参：**`gciql.py` 的默认值，加上官方 `impls/hyperparameters.sh` 里 `puzzle-3x3-play-v0` 那一行的 `--agent.alpha=1.0`：batch 1024，lr 3e-4，MLP 512×3（layer norm），γ 0.99，目标网络软更新 0.005，价值目标 20 % 当前状态、50 % 同轨迹后面的状态（几何采样）、30 % 随机状态，策略目标取同轨迹后面的状态；100 万步，种子 0（官方 `main.py` 默认）。
- **数据：**OGBench 官方 `puzzle-3x3-play-v0`（1000 回合 × 1001 步），和 DQC 那次同一份。
- **目标：**9 个灯的目标状态（OGBench 的 oraclerep，`puzzle-3x3-play-oraclerep-v0`），和 DQC 那次一样。
- **输入（`run_gciql.py --obs`）：**`state` 原版 55 维状态；`lewm2d` / `lewm3d` 冻结的 LeWM 2D / 3D 投影头输出 192 维 + 9 个灯的真值；`lewm2d_nolight` / `lewm3d_nolight` 只有 192 维潜变量。训练集的潜变量用 `bench/dqc/extract_feats.py` 算好的（和 DQC 共用）。
- **评测：**官方 `main.py` 的做法：第 1 步和之后每 10 万步，官方 5 个任务各 50 回合（`--eval_episodes=50`，同 `hyperparameters.sh`），温度 0。LeWM 输入每一步用同一套相机和编码器现算（`bench/dqc/lewm_obs.LeWMObs`）。

## 和官方代码的差别

1. **目标的维度：**官方 `GCIQLAgent.create` 按"目标和观测一样长"建网络（`ex_goals = ex_observations`）；oraclerep 的目标只有 9 维。`GCIQLOracleRep.create` 就是官方这个函数去掉这一行、目标维度改由数据给（逐行对过，只差这一行；OGBench 作者后来的代码 `third_party/dqc/agents/dqc.py` 也是这样取目标维度）。`OracleRepGCDataset` 在官方 `GCDataset` 取目标观测的地方（`get_observations`，不叠帧时只用来取目标）返回 9 个灯，采样、奖励、mask 都是官方的。
2. **LeWM 输入：**训练集的观测换成潜变量（+ 灯），评测时每步现算。
3. **其余：**不用 wandb（指标写 jsonl），不算验证集损失，评测在 GPU 上跑（官方默认把同一个模型拷到 CPU 上评测），每 2.5 万步存一次完整状态（`resume.pkl`），收到 SIGTERM 存档后退出，再跑同一条命令接着训。

## 环境

conda 环境 `lewm_dqc`（DQC 那个环境，已有 jax、flax、distrax、ml_collections、ogbench 1.2.0）。`PYTHONPATH` 里放 `lewm_pcd` 和 `third_party/stable-worldmodel`；`run_gciql.py` 把 `third_party/ogbench/impls` 放在 `sys.path` 最前面（它的 `agents`、`utils` 包），`lewm_obs.py` 按文件路径加载 `envs/ogb_puzzle_env.py`。

## 运行

全量由 `scripts/run_puzzle_pipeline.sh` 的第 4 步跑（`lewm2d`、`lewm3d` → `logs/gciql/puzzle_3x3/<输入>`）。单独跑：

```bash
cd bench/gciql
$DQC_PY run_gciql.py --obs lewm3d --ckpt $STABLEWM_HOME/checkpoints/puzzle_3x3_pcd_dexart \
    --feats_dir $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_3x3 --run_dir $STABLEWM_HOME/logs/gciql/puzzle_3x3/lewm3d \
    --agent.alpha=1.0
```
