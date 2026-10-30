# LeWM 2D / RGB

[返回覆盖矩阵](../COVERAGE.md)

## Cube1

索引 `lewm_rgb_cube1`；原生组名 `M5`。当前输入 211 维，目标 195 维，原生 latent 192 维，不补零。

实际配置：

- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/configs/M5_w4_formal_config.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/_providers/37e06651cbfffa3b/M5_w4_formal_config.json)
- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/ogbench-official-single-play-20261006/rl3_m5_gciql_seed0_b1024_1m_w4_20261013_v1/resolved.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/_providers/b5ad2c6c672e6730/resolved.json)

Single RL3 保留原有 RNG 偏离：省略中间在线评测，也不合成消耗相同 RNG；不将其改写成原始 main 的随机数流。

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存验收 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存交付 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 训练前门控 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 完整训练验收 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| Native SR | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| SR 独立审计 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |

原生实现导航（完整映射见 `source_map.json`）：

- [official_state_2d_cache_v1/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [official_state_2d_cache_v1/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)
- [cube23_distance_gciql_v1/trainer_runtime.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/trainer_runtime.py)
- [cube4_visual_gciql_v1/functional_v2/recovery.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v2/recovery.py)
- [cube4_visual_gciql_v1/functional_v1/quad_visual.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v1/quad_visual.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_runtime.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_boundary.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py)
- [cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py)
- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/core_v1/train.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/train.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube2

索引 `lewm_rgb_cube2`；原生组名 `G05`。当前输入 211 维，目标 198 维，原生 latent 192 维，不补零。

实际配置：

- [cube23_distance_gciql_v1/five_group_extension_v1/configs/G05_cube2_rgb.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/configs/G05_cube2_rgb.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存验收 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存交付 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 训练前门控 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| Native SR | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| SR 独立审计 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |

原生实现导航（完整映射见 `source_map.json`）：

- [official_state_2d_cache_v1/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [official_state_2d_cache_v1/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)
- [cube23_distance_gciql_v1/trainer_runtime.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/trainer_runtime.py)
- [cube4_visual_gciql_v1/functional_v2/recovery.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v2/recovery.py)
- [cube4_visual_gciql_v1/functional_v1/quad_visual.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v1/quad_visual.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_runtime.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_boundary.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py)
- [cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py)
- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/core_v1/train.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/train.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube3

索引 `lewm_rgb_cube3`；原生组名 `G08`。当前输入 211 维，目标 201 维，原生 latent 192 维，不补零。

实际配置：

- [cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存验收 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存交付 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 训练前门控 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/launch_plan.json) |
| Native SR | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| SR 独立审计 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |

原生实现导航（完整映射见 `source_map.json`）：

- [official_state_2d_cache_v1/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [official_state_2d_cache_v1/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)
- [cube23_distance_gciql_v1/trainer_runtime.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/trainer_runtime.py)
- [cube4_visual_gciql_v1/functional_v2/recovery.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v2/recovery.py)
- [cube4_visual_gciql_v1/functional_v1/quad_visual.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v1/quad_visual.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_runtime.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_boundary.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py)
- [cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py)
- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/core_v1/train.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/train.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube4

索引 `lewm_rgb_cube4`；原生组名 `G11`。当前输入 211 维，目标 204 维，原生 latent 192 维，不补零。

实际配置：

- [cube4_visual_gciql_v1/functional_v1/configs/G11_cube4_rgb.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v1/configs/G11_cube4_rgb.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存验收 | 原始命令已留存 | [cube4_gciql_native_sr_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_gciql_native_sr_v1/launch_plan.json) |
| 缓存交付 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 训练前门控 | 原始命令已留存 | [cube4_visual_gciql_v1/control_v1/G11/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/control_v1/G11/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [cube4_visual_gciql_v1/control_v1/G11/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/control_v1/G11/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [cube4_visual_gciql_v1/control_v1/G11/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/control_v1/G11/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [cube4_visual_gciql_v1/control_v1/G11/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/control_v1/G11/launch_plan.json) |
| Native SR | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| SR 独立审计 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |

原生实现导航（完整映射见 `source_map.json`）：

- [official_state_2d_cache_v1/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [official_state_2d_cache_v1/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)
- [cube23_distance_gciql_v1/trainer_runtime.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/trainer_runtime.py)
- [cube4_visual_gciql_v1/functional_v2/recovery.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v2/recovery.py)
- [cube4_visual_gciql_v1/functional_v1/quad_visual.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube4_visual_gciql_v1/functional_v1/quad_visual.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_runtime.py)
- [cube23_distance_gciql_v1/state_priority_v1/state_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/state_priority_v1/state_boundary.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_runtime.py)
- [cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/five_group_extension_v1/extension_boundary.py)
- [cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py)
- [assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/core_v1/train.py](../../../lewm_pcd/bench/ogbench/downstream/vendor/train.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。
