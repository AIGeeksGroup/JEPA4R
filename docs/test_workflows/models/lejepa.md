# LeJEPA Tiny / head512

[返回覆盖矩阵](../COVERAGE.md)

## Cube1

索引 `lejepa_cube1`；原生组名 `lejepa_tiny512_cube1`。当前输入 531 维，目标 515 维，原生 latent 512 维，不补零。

实际配置：

- [lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube1.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py)
- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube2

索引 `lejepa_cube2`；原生组名 `lejepa_tiny512_cube2`。当前输入 531 维，目标 518 维，原生 latent 512 维，不补零。

实际配置：

- [lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube2.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube2.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py)
- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube3

索引 `lejepa_cube3`；原生组名 `lejepa_tiny512_cube3`。当前输入 531 维，目标 521 维，原生 latent 512 维，不补零。

实际配置：

- [lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube3.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/configs/formal/lejepa_tiny512_cube3.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py)
- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube4

索引 `lejepa_cube4`；原生组名 `lejepa_tiny512_cube4`。当前输入 531 维，目标 524 维，原生 latent 512 维，不补零。

实际配置：

- [lejepa_tiny512_cube4_resource_v3/configs/formal/lejepa_tiny512_cube4.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/configs/formal/lejepa_tiny512_cube4.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube4_resource_v3/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_bridge.py)
- [lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/lejepa_tiny512_cube14_gciql_v1/release_v1/tiny_encoder.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。
