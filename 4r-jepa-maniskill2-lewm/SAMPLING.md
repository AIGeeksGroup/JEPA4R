# 当前采样策略

默认策略为 **OGBench `CubeMarkovOracle` 加本地 `PickCubeHoldOracle` 保持阶段，外层仍用 stable_worldmodel 原始 `ExpertPolicy.get_action` 及官方噪声**。保持阶段用于让夹持中的方块在目标附近有机会停稳；全程噪声仍然存在，失败、脱落和中间操作均保留。

正式采集默认直接开始，没有前置成功率预检或零成功硬门槛。可选诊断只记录覆盖情况，不过滤轨迹、不人为配平成功和失败比例。本机全量采集已于2026-09-07启动：10进程，10000回合，每条200步；记录目录为 `artifacts/wm_collection_full_20260907T044524Z`，启动PID158972，与16环境PPO运行目录 `artifacts/rl_identity_train_p16_20260907`（启动PID152283）并行。PID只描述本次启动，不代表未来仍是同一进程；这些本机产物不包含在 Git 仓库中。

## 源码位置与来源

| 功能 | 本仓库实现 |
| --- | --- |
| 采集入口、配置、manifest | [scripts/collect_wm.py](scripts/collect_wm.py) |
| 官方噪声复用、状态和动作映射 | [src/ms2_lewm/expert.py](src/ms2_lewm/expert.py) |
| PickCube 保持阶段 | [src/ms2_lewm/pickcube_oracle.py](src/ms2_lewm/pickcube_oracle.py) |
| 环境、目标重采样、点云及诊断观测 | [src/ms2_lewm/environment.py](src/ms2_lewm/environment.py) |
| 并行仿真与渲染 | [src/ms2_lewm/process_env_pool.py](src/ms2_lewm/process_env_pool.py) |
| 无筛选覆盖统计 | [src/ms2_lewm/coverage.py](src/ms2_lewm/coverage.py) |
| 去重、fp16、zstd12 与容量检查 | [src/ms2_lewm/storage.py](src/ms2_lewm/storage.py) |

LeWM 固定为 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，兼容 stable_worldmodel 固定为 `abdced49809d5eae38e24b27dc7b635c502c4812`；复用其 `scripts/data/collect_cube.py` 流程、`scripts/data/config/ogb.yaml` 和 `World.collect`。Oracle 来自锁定的 `ogbench==1.2.0`。所有上游来源见 [repro/sources.lock.json](repro/sources.lock.json)，由 `python3 scripts/bootstrap_sources.py` 下载到 `vendor/`，上游源码未修改。

官方 cube 是 OGBench，未提供 ManiSkill2 点云数据。本仓库对机器人接口、目标高度及保持阶段的适配均在本地源码中，不能把当前数据称为原论文数据，也不能称为采集行为与原 OGBench 完全相同。

## 原始启发式与新增保持阶段

原 `CubeMarkovOracle` 根据特权状态逐步执行：移到方块上方、接近、闭合夹爪、抬高、搬到目标上方、靠近目标、释放、抬高退避、移到最终退避位置。它按自己的结束条件或内部200步上限请求新目标。

主要原始参数为 XY 对齐阈值 `.04`、位置对齐阈值 `.02`、上方偏移 `.18`、高度判断 `.16`、位置增益 `5`、yaw 增益 `3`，动作限幅 `[-1,1]`。`min_norm=.4` 是位置误差向量整形，不是噪声大小。

默认 `--oracle-mode pickcube_hold --hold-steps 30` 增加以下本地逻辑：

1. 方块仍被抓住且距目标不超过 `.025m` 时，在释放前进入保持阶段，最多30个控制步，即1.5秒。
2. 保持阶段使用 `5 × goal_error` 的平移修正和闭合夹爪，不使用移动阶段的最小误差范数放大；外层仍对每步动作施加完整官方噪声。
3. 若失去抓取或偏离目标超过 `.08m`，退出保持阶段并回到原 oracle 的动作逻辑；之后再次满足条件仍可重试。
4. 保持结束后回到原 oracle 的释放、退避等逻辑。没有修改环境的原生 success 判定。

`--oracle-mode legacy` 显式关闭保持阶段，用于重现历史原 oracle 行为。当前采样保留全部尝试，并不会强行制造失败，也不会只保存抓取顺利的片段。能否获得足够的操作覆盖，需要审计实际数据；使用启发式加噪本身不是质量保证。

## 官方噪声

噪声直接执行固定 stable_worldmodel 的 `ExpertPolicy.get_action`，不重新实现其采样函数。

| 参数 | 值 | 作用 |
| --- | ---: | --- |
| `policy_type` | `markov_oracle` | 当前分支 |
| `action_noise` | `.1` | 每条轨迹的噪声尺度上界 |
| `p_random_action` | `0` | 不插入纯随机动作分支 |
| `min_norm` | `.4` | 原 oracle 的位置误差整形 |
| `noise_smoothing` | `.5` | 仅 plan_oracle 使用，当前分支不使用 |
| policy `seed` | `None` | 保留原默认，并记录初始 RNG 状态 |

每条轨迹开始抽取一次 `xi ~ Uniform(0,.1)`。每个控制步取得5维 oracle 动作 `[dx,dy,dz,dyaw,dgripper]`，再独立抽取噪声：

```text
epsilon ~ Normal(0, diag([xi, xi, xi, 3*xi, 10*xi])²)
noisy_action = clip(oracle_action + epsilon, -1, 1)
```

轨迹内共享的是标准差尺度；每一步的噪声向量重新抽取。保持阶段执行同一外层噪声代码，没有额外降低噪声。本机采集 manifest 保存官方噪声 PCG64 和 NumPy 全局 RNG 初始状态，记录实际 episode 噪声尺度。默认 `seed=None` 不等于跨新运行逐位确定；需要原始 manifest 的 RNG 状态才能重放相应随机流。

## ManiSkill2 状态与动作接口

环境为 `PickCube-v0`，控制模式 `pd_ee_delta_pose`，20Hz。Oracle 读取 TCP、方块、目标的位置与 yaw、夹爪状态等特权状态；这些状态不进入点云 encoder。抓取信号使用 ManiSkill 原生 `check_grasp`。原退避采样盒在 X 方向平移到 ManiSkill 桌面中心，盒子尺寸保留。

原 oracle 的世界坐标5维动作转换为 Panda 末端坐标7维动作：

- 平移：原最大 `.05m` 转为 Panda `.1m` 归一化动作，再用 TCP 旋转矩阵转到末端坐标系。
- 旋转：原 world-yaw `.3rad` 转为 Panda `.1rad` 旋转向量单位，再转到末端坐标系并遵守控制器限幅。
- 夹爪：读取 Panda 原始物理上下界 `l,u`，用 `c=(u-q)/(u-l)` 表示闭合量，加上官方夹爪增量后限幅至 `[0,1]`，反变换为目标关节位置与归一化动作。零增量保持当前开度，闭合端保留原控制器受力端点。

中途重采目标不调用任务 reset hook `_initialize_task`。该 hook 会把当前方块高度当作基准，方块举高后可能重复抬高目标。当前 `set_new_target` 保留原生XY范围、RNG和最小间距拒绝采样，Z基准固定为桌面方块中心 `.02m`，范围 `.02–.52m`。

原生成功定义未改：

```text
is_obj_placed = ||cube_position - goal_position|| <= .025m
is_robot_static = max(abs(arm_joint_velocity)) <= .2rad/s
success = is_obj_placed AND is_robot_static
```

几何到达不等于原生成功，视频里看似抓起也不能替代成功判定。

## 采集、并行和诊断入口

正式配置保留官方10环境、10000轨迹、每条200步、配置seed3072，`terminate_at_goal=False`。默认 `--env-backend process` 用10个 `spawn` 进程各自执行物理仿真和渲染；oracle 在主进程按环境索引调用，通过RPC读取状态与重采目标，主进程统一写盘。`--env-backend serial` 保留串行调试；数量和horizon覆盖仅允许 `--smoke`。

在按 [README](README.md#复现入口) 安装依赖后：

```bash
source scripts/wm_env.sh

# 默认直接开始全量采集；输出路径须不存在
.venv-worldmodel/bin/python scripts/collect_wm.py \
  --output artifacts/expert_full.lance

# 本机当前运行使用的空间预算；不是其他机器必须使用的默认值
# .venv-worldmodel/bin/python scripts/collect_wm.py \
#   --output artifacts/expert_full.lance --reserve-gib 2 --storage-safety-factor 1.05

# 仅诊断30条，不开始全量数据写入
.venv-worldmodel/bin/python scripts/collect_wm.py \
  --output artifacts/expert_diagnostic.lance --preflight-only
```

`--preflight-episodes N` 可在采集前追加N条诊断，报告为 `<dataset>.preflight.json`；默认0条。诊断是 advisory，原生成功0也不阻止采集。图形入口仍检查实际空闲显存，容量不足仍会报错；不会结束其他进程。

## 点云与存储

每帧来自原始RL wrapper的实际 `1200×6` XYZ+RGB 点云，world坐标系。仅在落盘时转float16，在线PPO输入仍float32，WM reader 读出后恢复 `[1200,6]` 并转float32。对于绝对值≤1的分量，float16量化的最大舍入误差不超过约`.00024414`；XYZ按米计约`.244mm`。因此量化后的点云不再与原float32逐位相同。

优先直接写 `.lance`：删除与点云完全相同的 `rl_xyz/rl_rgb`、与 `observation` 完全相同的 `rl_state/proprio`；RGB保留原JPEG质量95，动作、状态和仿真快照保留。Lance物理缓冲使用原生 **zstd level12无损压缩**，不额外解码或重压JPEG，不改变量化后的逻辑数值。训练和评测直接使用官方 `stable_worldmodel.data.load_dataset`，无需再生成全量HDF5副本。显式 `.h5/.hdf5` 输出直接流式写压缩HDF5，不生成中间Lance。

容量检查先允许少量探测数据，随后按已提交文件的实际增长外推剩余空间，已写入数据不重复计数。CLI默认保留10GiB、安全系数1.2；本机全量运行显式改为2GiB与1.05，不应把这两个本机选择当成通用要求。空间不足会停止，不缩短轨迹、不减少点数、不改噪声或JPEG质量。

历史10进程独占短测为6000步、43.13秒、139.1步/秒，发生在保持阶段和zstd12启用之前；GPU平均12.3%、观察峰17%，CPU按16核计平均38.7%、峰68.1%。见 [资源曲线](validation/resource-profiles.png)。当前与16环境PPO并行，不能照搬独占速度作为当前耗时保证。旧未启用zstd12的46.27GiB全量容量估计也不代表当前格式大小。

## 覆盖统计与历史记录边界

`CoverageWriter` 观察每条完整轨迹，记录接近、抓取、抬起、搬运、到达目标、原生成功和抓取后脱落等覆盖指标；随后将原轨迹原样交给存储writer。不拒绝、不裁剪、不重采轨迹，不设成功/失败比例。

本机正在运行的早期采集进程已导入旧版代码：布尔诊断列在环境信息转换中遗漏，旧覆盖报告会把缺失的抓取或success列误计成0。**这些旧0值不能用于声称全失败；对应缺失指标是未知。** 代码修复只对之后新启动的进程生效，不修改已运行进程或补造旧轨迹成功标记。点云、动作及状态数据不受该诊断统计问题影响；可检查几何轨迹，但不能把几何到达充当原生success。episode/帧数统计与这两个缺失布尔列不同。

历史 legacy oracle 的30条重测为原生0/30、位置到达28/30，记录见 [validation/retarget-fix.json](validation/retarget-fix.json)。它发生在保持阶段加入前，与当前轨迹质量不能直接等同。性能/合成测试数据分别标注 `BENCHMARK_ONLY_NOT_TRAINING_DATA`、`SYNTHETIC_UNIT_TEST_ONLY`，生产训练入口拒绝这些标记。

WM训练片段为 `T=3,num_preds=1,frameskip=1`，每片段读取四个连续控制步帧 `s,s+1,s+2,s+3`，当前动作 `a_(s+2)` 预测第4帧latent1；SIGReg在全部4帧上逐帧沿batch计算。第三阶段冻结encoder同样读取连续三帧。正式WM训练为10epochs；真实数据WM和冻结表征PPO的完整实验尚未完成，不以采集启动或测试通过声称论文结果复现。
