# PLDM

[返回覆盖矩阵](../COVERAGE.md)

## Cube1

索引 `pldm_cube1`；原生组名 `pldm_cube1`。当前输入 531 维，目标 515 维，原生 latent 512 维，不补零。

实际配置：

- [pldm_cube14_gciql_v1/configs/formal/pldm_cube1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/configs/formal/pldm_cube1.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [pldm_cube14_gciql_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/run.py)
- [pldm_cube14_gciql_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/prepare.py)
- [pldm_cube14_embeddings_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/run.py)
- [pldm_cube14_embeddings_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/prepare.py)
- [pldm_cube14_embeddings_v1/cache_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/cache_bridge.py)
- [cube2_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/prepare.py)
- [cube3_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/prepare.py)
- [cube2_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube3_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube2

索引 `pldm_cube2`；原生组名 `P2_cube2_pldm_reaudit_rgb`。当前输入 531 维，目标 518 维，原生 latent 512 维，不补零。

实际配置：

- [pldm_cube2_reaudit_20261029_v1/functional_v1/configs/P2_cube2_pldm_reaudit_rgb.v1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube2_reaudit_20261029_v1/functional_v1/configs/P2_cube2_pldm_reaudit_rgb.v1.json)

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

- [pldm_cube14_gciql_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/run.py)
- [pldm_cube14_gciql_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/prepare.py)
- [pldm_cube14_embeddings_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/run.py)
- [pldm_cube14_embeddings_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/prepare.py)
- [pldm_cube14_embeddings_v1/cache_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/cache_bridge.py)
- [cube2_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/prepare.py)
- [cube3_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/prepare.py)
- [cube2_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube3_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube3

索引 `pldm_cube3`；原生组名 `P3_cube3_pldm_rgb`。当前输入 531 维，目标 521 维，原生 latent 512 维，不补零。

实际配置：

- [cube3_pldm_gciql_v1/functional_v1/configs/P3_cube3_pldm_rgb.v1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/configs/P3_cube3_pldm_rgb.v1.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存验收 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 缓存交付 | 原生代码/配置已收录；未留存独立阶段命令 | 见实际配置和下列原生代码索引 |
| 训练前门控 | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |
| Native SR | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [cube3_pldm_gciql_v1/control_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/control_v1/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [pldm_cube14_gciql_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/run.py)
- [pldm_cube14_gciql_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/prepare.py)
- [pldm_cube14_embeddings_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/run.py)
- [pldm_cube14_embeddings_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/prepare.py)
- [pldm_cube14_embeddings_v1/cache_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/cache_bridge.py)
- [cube2_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/prepare.py)
- [cube3_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/prepare.py)
- [cube2_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube3_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube4

索引 `pldm_cube4`；原生组名 `pldm_cube4`。当前输入 531 维，目标 524 维，原生 latent 512 维，不补零。

实际配置：

- [pldm_cube14_gciql_v1/configs/formal/pldm_cube4.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/configs/formal/pldm_cube4.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [pldm_cube14_embeddings_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [pldm_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [pldm_cube14_gciql_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/run.py)
- [pldm_cube14_gciql_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_gciql_v1/prepare.py)
- [pldm_cube14_embeddings_v1/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/run.py)
- [pldm_cube14_embeddings_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/prepare.py)
- [pldm_cube14_embeddings_v1/cache_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/pldm_cube14_embeddings_v1/cache_bridge.py)
- [cube2_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/prepare.py)
- [cube3_pldm_gciql_v1/functional_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/prepare.py)
- [cube2_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube3_pldm_gciql_v1/functional_v1/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/native_sr.py)
- [cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube2_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_v1/pldm_runtime.py)
- [cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/cube3_pldm_gciql_v1/functional_recovery_v3/prepare.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。
