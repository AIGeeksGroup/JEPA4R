# VISReg ViT-B / CLS768

[返回覆盖矩阵](../COVERAGE.md)

## Cube1

索引 `visreg_cube1`；原生组名 `visreg_vitb768_cube1`。当前输入 787 维，目标 771 维，原生 latent 768 维，不补零。

实际配置：

- [foundation_visual_gciql_v1/configs/visreg_vitb768_cube1.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/configs/visreg_vitb768_cube1.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| Native SR | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [visreg_cube34_gciql_v1/release_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/release_v1/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube2

索引 `visreg_cube2`；原生组名 `visreg_vitb768_cube2`。当前输入 787 维，目标 774 维，原生 latent 768 维，不补零。

实际配置：

- [foundation_visual_gciql_v1/configs/visreg_vitb768_cube2.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/configs/visreg_vitb768_cube2.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| 独立重载 | 原始命令已留存 | 见实际配置和下列原生代码索引 |
| GCIQL 训练 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| Native SR | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [visreg_cube34_gciql_v1/release_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/release_v1/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube3

索引 `visreg_cube3`；原生组名 `visreg_vitb768_cube3`。当前输入 787 维，目标 777 维，原生 latent 768 维，不补零。

实际配置：

- [visreg_cube34_gciql_v1/configs/formal/visreg_vitb768_cube3.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/configs/formal/visreg_vitb768_cube3.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_model_gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_model_gates/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_model_gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_model_gates/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json) |
| Native SR | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube3_full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [visreg_cube34_gciql_v1/release_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/release_v1/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。

## Cube4

索引 `visreg_cube4`；原生组名 `visreg_vitb768_cube4`。当前输入 787 维，目标 780 维，原生 latent 768 维，不补零。

实际配置：

- [visreg_cube34_gciql_v1/configs/formal/visreg_vitb768_cube4.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/configs/formal/visreg_vitb768_cube4.json)

| 阶段 | 收录状态 | 原始启动计划 |
|---|---|---|
| 特征缓存生产 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存验收 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 缓存交付 | 原始命令已留存 | [foundation_visual_gciql_v1/control/formal_w8/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/control/formal_w8/launch_plan.json) |
| 训练前门控 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_model_gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_model_gates/launch_plan.json) |
| 独立重载 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_model_gates/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_model_gates/launch_plan.json) |
| GCIQL 训练 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json) |
| 完整训练验收 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json) |
| Native SR | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json) |
| SR 独立审计 | 原始命令已留存 | [visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/control/stages/cube4_full/launch_plan.json) |

原生实现导航（完整映射见 `source_map.json`）：

- [visreg_cube34_gciql_v1/release_v1/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/visreg_cube34_gciql_v1/release_v1/prepare.py)
- [foundation_visual_gciql_v1/cache/base/produce.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/produce.py)
- [foundation_visual_gciql_v1/functional/prepare.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/prepare.py)
- [foundation_visual_gciql_v1/functional/runtime.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/runtime.py)
- [foundation_visual_gciql_v1/functional/native_sr.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/functional/native_sr.py)
- [foundation_visual_gciql_v1/cache/base/audit_cache.py](../../../lewm_pcd/bench/ogbench/test_workflows/snapshot/foundation_visual_gciql_v1/cache/base/audit_cache.py)

外部数据、权重、缓存及其哈希绑定保留在配置中。本次没有重跑该组训练、重载或 Native SR，也不重新确认旧结果。
