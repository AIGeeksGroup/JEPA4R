# 阶段与使用方法

[返回总览](../../README_TEST_WORKFLOWS.md)

## 阶段顺序

视觉组沿用各自原生链路：准备数据与冻结 encoder → 生产缓存 → 缓存验收/交付 → 模型与输入门控 → 独立重载 → GCIQL 训练 → 完整 CPU 验收 → Native SR → 独立审计。State 不经过视觉缓存阶段。旧组的阶段划分、顺序和门控条件以原始配置/计划为准。

检查入口使用下列阶段名：

| 参数 | 含义 |
|---|---|
| `encode` | 原生 encoder、预处理与缓存生产 |
| `cache-audit` | 原始索引、true terminal、shape/dtype/finite、哈希和来源检查 |
| `deliver` | 缓存交付及配置绑定 |
| `gate` | 原生训练前诊断门控；检查入口本身不会执行 |
| `reload` | 独立进程重载门控 |
| `gciql` | 原始下游训练 |
| `accept` | 原始完整模型、优化器、计数器与配置验收 |
| `native-sr` | 原始 Native 闭环评测 |
| `native-audit` | 结果独立审计 |

`lewm2d`/`lewm3d` 分别对应 `lewm_rgb`/`lewm_pcd`；`visreg_vitb768`、`vjepa21_vitl1024`、`lejepa_tiny512` 也可用于查询。

## 读取一组

```bash
python lewm_pcd/scripts/inspect_test_workflows.py --model lewm3d --cube-count 2
python lewm_pcd/scripts/inspect_test_workflows.py --model lejepa --cube-count 4 --stage gciql
```

输出包含真实配置路径、原生输入维度、原始阶段对象和外部资产引用。`recorded_phases` 带有来源计划和数组索引；`recorded_commands` 带有来源配置及原字段名。它们保留历史 argv/env/cwd/输出目录，不是新一次运行的授权或已生成的新输出配方。

## 还原代码布局

仅在需要一个独立、空的审阅/复现目录时使用：

```bash
PYTHONPATH=lewm_pcd python -m bench.ogbench.test_workflows.materialize --destination /absolute/new/test-source
```

程序先核对代码哈希，目标必须不存在；只还原源码、配置和静态资源，不复制数据集、模型、缓存或日志，不执行任何原生入口。已有原生文件中的绝对路径不自动替换。后续运行需要满足[环境与路径条件](REPRODUCIBILITY.md)，保留新配置与原配置的差异及绑定证据。

## 原有 switch

`run_ogbench.py` 的上游训练与下游接口保持原样。Test 索引入口单独列出其实际原生配方，保留 `async100`、旧 Single RL3、资源版本和不同阶段环境的来源，不将其默认为 v3 同步参考训练器的新验收结果。
