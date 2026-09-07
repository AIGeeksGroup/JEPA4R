# 当前采样策略

当前采用**原始 OGBench `CubeMarkovOracle` + stable_worldmodel 原始 `ExpertPolicy.get_action` 噪声 + ManiSkill2 状态/动作接口适配**。没有新增停稳阶段，没有改 oracle、噪声或原生 success 条件；中途目标重采样的高度基准错误已修复，见下文。

修复后同噪声30回合预检结果仍为 **0/30 success**，其中 **28/30 仅曾到达位置条件**；原生 success 还要求手臂同时静止。旧版本的位置结果是27/30。正式点云采集门槛仍关闭。

## 来源与原始启发式策略

- LeWM 配置：`vendor/le-wm`，commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。
- 兼容采集实现：`vendor/stable-worldmodel-lewm-compatible`，commit `abdced49809d5eae38e24b27dc7b635c502c4812`；`scripts/data/collect_cube.py` 和 `scripts/data/config/ogb.yaml`。
- 实际启发式策略：已安装并锁定的 `ogbench==1.2.0` 中 `CubeMarkovOracle`。实际调用它的原始 `reset/select_action`，不是另外写的一套抓取流程。
- 噪声：上述 stable_worldmodel 的 `envs/ogbench/expert_policy.py`。直接加载该未修改文件，避免触发与当前任务无关的 OGB 图形环境导入；没有重写其函数。

Oracle 每个控制步读取真实特权状态，根据条件选择移动到方块上方、接近方块、闭合夹爪、抬高、移到目标上方、移到目标、释放、抬高退避、移到最终退避位置等阶段。它会在自己判断子任务结束或内部步数达到200时请求新目标。

主要原始参数为：XY 对齐阈值 `.04`，位置对齐阈值 `.02`，上方位置偏移 `.18`，高度判断 `.16`，位置增益 `5`，yaw 增益 `3`；动作限幅 `[-1,1]`。`min_norm=.4` 用于**位置误差向量整形**：误差向量范数小于 `.4` 时，按 `d / (||d|| + 1e-6) * .4` 整形，然后计算控制动作。它与噪声幅度是不同参数，实际移动量仍由动作单位和控制器限幅决定。

## 噪声究竟怎么加

当前使用原始默认 `policy_type="markov_oracle"`：

| 参数 | 值 | 当前作用 |
|---|---:|---|
| `action_noise` | `.1` | 每条轨迹开始均匀抽取 xi 时的固定上界 |
| `p_random_action` | `0` | 不进入纯随机动作分支 |
| `min_norm` | `.4` | 上述位置误差整形 |
| `noise_smoothing` | `.5` | 仅供 `plan_oracle` 使用；当前 markov 分支不使用 |
| policy `seed` | `None` | 保留官方默认；记录随机数状态用于重放 |

轨迹开始时抽一次 `xi ~ Uniform(0,.1)`；**同一条轨迹内 xi 保持不变**。每个控制步先取得 oracle 的5维动作 `[dx,dy,dz,dyaw,dgripper]`，再独立抽取高斯噪声：

```text
epsilon ~ Normal(0, diag([xi, xi, xi, 3*xi, 10*xi])²)
noisy_action = clip(oracle_action + epsilon, -1, 1)
```

因此每一步的噪声向量重新抽取，轨迹内共享的是标准差尺度。新增包装直接调用原始 `ExpertPolicy.get_action`，包括原有 RNG 抽样顺序、限幅和请求新目标的逻辑。

## ManiSkill2 必要接口适配

目标环境为 `PickCube-v0`，原始控制模式 `pd_ee_delta_pose`。供 oracle 使用的特权状态在策略调用边界映射，不混入 encoder 的点云输入：TCP、方块、目标的位置与 yaw、夹爪状态等。接触信号使用 ManiSkill 原生 `check_grasp`；Panda 与原始 Robotiq 的力传感器数值尺度不同。原始退避采样盒在 X 方向平移到 ManiSkill 桌面中心，盒子的尺寸保持原值。

原始 oracle 为世界坐标系5维控制，Panda 为末端坐标系7维控制，转换如下：

- 平移：原始最大 `.05m` 转换为 Panda 最大 `.1m` 的归一化动作，再用 TCP 旋转矩阵转到末端坐标系。
- 旋转：原始 world-yaw 最大 `.3rad` 转换为 Panda `.1rad` 旋转向量单位，再转到末端坐标系。最终遵守原始 Panda 控制器界限；两个机器人可实现的单步转角不同。
- 夹爪：读取 Panda 原始控制器的物理上下界 `l,u`（当前 `[-.01,.04]m`），以 `c=(u-q)/(u-l)` 表示闭合量；把官方夹爪增量加到 c，限幅到 `[0,1]`，再用同一上下界反变换为目标关节位置及归一化动作。这样零增量保持当前位置，完全闭合保留原 RL 的受力端点。

中途 `set_new_target` 不再调用仅供reset使用的 `_initialize_task`：该函数将当前方块高度作为目标高度基准，方块被举高后反复重采会让目标超出初始范围。当前适配保留原生XY范围、RNG和最小间距拒绝采样，把Z基准固定为桌面方块中心 `.02m`，目标Z范围 `.02–.52m`。原生成功定义完整保留：

```text
is_obj_placed = ||cube_position - goal_position|| <= .025m
is_robot_static = max(abs(arm_joint_velocity)) <= .2rad/s
success = is_obj_placed AND is_robot_static
```

OGBench cube 与 ManiSkill PickCube 是不同的物理任务和目标分布；以上是移植官方采集实现时的环境接口边界，不能把预检结果称为 LeWM 原论文数据或实验结果。

## 正式配置与实际执行

| 项目 | 正式采集配置 | CPU预检实际执行（另有下文10回合渲染性能短测） |
|---|---|---|
| 环境数 | 官方 `10` 个环境，默认各自独立进程 | `1` 个，串行 |
| 轨迹数 | 官方 `10,000` | 扩大预检 `30` 条 |
| 每条步数上限 | 官方 `200` | `200`，合计 `6,000` 步 |
| 配置 seed | `3072` | `3072`；派生 world seed 为 `724895` |
| 图形/点云 | 真实 RGB 与 RL 点云 | 纯 CPU 真实物理状态，无图片、无点云 |
| 原始 oracle/噪声 | 保持原实现 | 保持原实现 |
| 状态 | **未启动** | 已完成，原生 success `0/30` |

正式入口默认 `--env-backend process`：`num_envs > 1` 时用 `spawn` 启动独立环境进程，正式配置因此是10个并行worker，每个worker拥有自己的仿真环境和renderer。原始oracle及 `ExpertPolicy` 仍在主进程按原环境索引顺序调用，状态与动作通过进程间接口传递，重采目标使用RPC；轨迹由主进程统一提交writer，避免多个worker同时写同一数据集。单环境路径仍串行，`--env-backend serial` 保留旧后端供显式调试。正式环境数、轨迹数、噪声和时间间隔保持原配置；`--num-envs` 仅用于原有 `--smoke` 小规模验证。

默认30回合成功率预检仍是1环境串行，不会因新增并行后端跳过或弱化门槛。CPU 预检复用官方 `World.evaluate`、`EnvPool` 及真实 ManiSkill 物理/控制器/成功条件，仅省去图形构建。该路径明确禁止 collect，不能产生训练数据。记录确认 `SapienRenderer` 和 `RenderClient` 调用数均为0，进程 GPU 显存样本全部为0。

旧版本30条中有27条曾达到位置条件，共438个位置满足的状态；没有任何一个同时满足静止条件。历史位置满足时，最大手臂关节速度的最小值为 `.34014rad/s`，高于原生 `.2rad/s` 阈值。修复高度基准后同噪声重测，位置到达28/30、原生成功仍0/30，IK失败从199次降到186次。高度bug不能解释全部IK失败。原始oracle到达目标后释放并退避，PickCube还要求静止；没有为获得success加入停稳阶段或改写成功条件。

最新修复证据：[validation/retarget-fix.json](validation/retarget-fix.json)。历史证据：`artifacts/expert_cpu_preflight30.json`、`artifacts/expert_cpu_preflight30.trace.json`、`artifacts/expert_preflight_diagnosis.json`。所有success字段都表示原生成功，位置到达单独统计。

## 采集门槛、存储与训练片段

`scripts/collect_wm.py` 默认先验证 **30条**。成功数为0时在数据集创建前报错；已有 CPU 诊断不会伪装成点云数据，也不会自动放行正式采集。图形入口还在资源分配前读取 GPU 的实际 `memory.free`，要求至少 `max(2048,1024+256*num_envs)` MiB 空闲；评测把模型移到 CUDA 前再预留1024MiB。不会结束、覆盖或修改已有任务。

门槛满足后，数据路径仍调用官方 `World.collect`，由本地多进程环境池并行执行各环境的仿真/渲染；本项目 `CompactLanceWriter` 在官方writer上只改变点云存储类型和删除精确重复列，官方JPEG质量95保留。直接使用 `.lance` 训练及评测，不生成全量HDF5副本。兼容 `.h5/.hdf5` 入口直接流式写gzip压缩HDF5，图像沿用原Lance→HDF5路径的JPEG95编解码，既不复制全量数据也不增加JPEG压缩轮次。两种格式都由原版 `stable_worldmodel.data.load_dataset` 读取。正式采集的 `terminate_at_goal=False` 与官方一致，成功后可继续至轨迹上限。

每帧点云来自原始RL wrapper：`1200`点、`world`坐标、`xyz+rgb`共6通道。删除完全重复的 `rl_xyz/rl_rgb`，状态保留 `observation` 并删除完全重复的 `rl_state/proprio`；仿真状态、动作和RGB保留。点云仅在落盘时量化为float16，Lance仍展平为7200维，模型入口恢复 `[1200,6]` 并转float32。量化不改变形状或时间索引，但不再与原float32逐位相同：测试中绝对值≤1的分量最大舍入误差≤`2^-12 ≈ 0.00024414`，对应XYZ约0.244mm。在线观测与PPO输入保持float32。

首个episode提交后根据实际文件增长估算剩余全量空间，增加20%余量并默认保留10GiB（`--reserve-gib`可显式改）；以后每10个episode重新计算。检查使用当前空闲磁盘和剩余步数，已落盘数据不重复计数。采集前的检查只预留少量探测数据，不把未压缩fp16字节数错误当成压缩后容量下界。空间不足即停止，不缩短轨迹、不减少点数、不改噪声或JPEG质量。

2026-09-07真实10环境×10回合×200步的性能短测使用旧版**单进程串行EnvPool**，并非10个并行worker。该次0/10成功，明确标为禁止用于正式训练的benchmark；约57.94秒完成2000控制步（34.5步/秒），按旧实现吞吐外推全量约16.1小时。新10进程版本以此估算：完全线性扩展的理想值为1.61小时，考虑共享GPU渲染、IPC、进程启动和writer开销，安排 **2–4小时**。这是新配置的粗估，未新增仿真测速，不是10倍加速承诺，也不包含成功率或容量不足时的等待时间。

最终分批writer在CPU上复放相同Arrow/JPEG字节后为49,682,995字节，外推10000回合约46.27GiB；按20%余量、每回合额外1帧保守预算和10GiB保留空间需约65.81GiB可用。该估算来自少量场景，后续仍持续复核；当前磁盘不足，不因已用fp16就放行。证据：`artifacts/benchmark_sampling_20260907/report.json`、`final_storage_report.json`。

按用户最新纠正，训练片段使用 `T=3`、`num_preds=1`、本地 `frameskip=1`，每片段取4个连续控制步的帧。相对起点 s，观测对应 `s,s+1,s+2,s+3`；动作每步7维，不再拼成35维。预测使用第三帧出发的 `a_(s+2)`，目标为 `s+3` 的 latent1。SIGReg 使用全部4帧，并逐帧沿 batch 维计算。原官方数据配置 frameskip=5 仍保存在 vendor，作为原始参考；它不是新增时间编码器的参数，当前实现不再沿用该抽帧间隔。第三阶段冻结 encoder 的 PPO 同样使用连续三帧。

性能短测在邻接 `<dataset>.json` 中明确记录 `BENCHMARK_ONLY_NOT_TRAINING_DATA=true`。生产训练入口在GPU分配前拒绝此标记与 `SYNTHETIC_UNIT_TEST_ONLY`，同时兼容旧HDF5属性标记；不会要求正常外部数据必须附带manifest。

新增记录同时保存官方噪声 PCG64 状态和 NumPy 全局 RNG 状态；后者影响 oracle 的退避位置抽样。已有30条旧报告仅记录了前者，不补造缺失状态。源码锁与哈希见 `artifacts/official_sources.lock.json`；8个采集契约测试已通过，覆盖原始 canonical 噪声/RNG一致性、零成功门槛、GPU资源门槛和夹爪单位转换。存储修复和实际空间证据见 [validation/compact-storage.json](validation/compact-storage.json)。
