# V-JEPA 2.1 ViT-L / D1024

[返回覆盖矩阵](../COVERAGE.md)

## Cube1

索引 `vjepa_cube1`；原生组名 `vjepa21_vitl1024_cube1`。当前输入 1043 维，目标 1027 维，原生 latent 1024 维，不补零。

实际配置：

- [vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube1.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube1_v1/control/train/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube1_v1/control/train/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json) |
| Native SR | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube1_post_train_v1/control/post_train/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [vjepa_vitl1024_gciql_v2/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/run.py)
- [vjepa_vitl1024_gciql_v2/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube2

索引 `vjepa_cube2`；原生组名 `vjepa21_vitl1024_cube2`。当前输入 1043 维，目标 1030 维，原生 latent 1024 维，不补零。

实际配置：

- [vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube2.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube2.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube2_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [vjepa_vitl1024_gciql_v2/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/run.py)
- [vjepa_vitl1024_gciql_v2/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube3

索引 `vjepa_cube3`；原生组名 `vjepa21_vitl1024_cube3`。当前输入 1043 维，目标 1033 维，原生 latent 1024 维，不补零。

实际配置：

- [vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube3.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube3.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube3_gate_tail_v1/control/tail/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube3_gate_tail_v1/control/tail/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube3_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [vjepa_vitl1024_gciql_v2/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/run.py)
- [vjepa_vitl1024_gciql_v2/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube4

索引 `vjepa_cube4`；原生组名 `vjepa21_vitl1024_cube4`。当前输入 1043 维，目标 1036 维，原生 latent 1024 维，不补零。

实际配置：

- [vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube4.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/configs/formal/vjepa21_vitl1024_cube4.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_model_gates_v1/control/gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_model_gates_v1/control/gates/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_model_gates_v1/control/gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_model_gates_v1/control/gates/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json) |
| Native SR | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/priority_cube4_v1/control/full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [vjepa_vitl1024_gciql_v2/run.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/run.py)
- [vjepa_vitl1024_gciql_v2/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/vjepa_vitl1024_gciql_v2/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。
