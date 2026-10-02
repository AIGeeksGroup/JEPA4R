# Pen 25-demo 历史记录

本页保留此前 README 的 25-demo 设置与结果，便于追溯。它们不是本轮 100-demo 对照，未在此次文档更新中重新训练或重新逐文件验收。

配置：25 demos、100 epochs、每 5 epochs 评测、每轮 25 条在线轨迹、policy seeds `100/200/300`；主指标为 Best-of-20。标准差使用总体标准差。

| 历史轨道 | Seed 100 best / final | Seed 200 best / final | Seed 300 best / final | Best-of-20 均值 ± std | Final 均值 ± std |
|---|---:|---:|---:|---:|---:|
| LeWM 2D RGB，25 demos | 28% / 16% | 32% / 24% | 32% / 28% | 30.7% ± 1.9% | 22.7% ± 5.0% |
| LeWM 3D PCD，25 demos | 68% / 52% | 68% / 68% | 64% / 60% | 66.7% ± 1.9% | 60.0% ± 6.5% |

| 历史产物 | 路径 |
|---|---|
| RGB seeds 100/200/300 | `/autodl-fs/data/lewm/cortexbench-adroit-pen-official-bc-2d-20260928-v2` |
| PCD seeds 100/200 | `/autodl-fs/data/lewm/cortexbench-adroit-pen-pcd-bc-20260929-v1` |
| PCD seed 300 retry | `/autodl-fs/data/lewm/cortexbench-adroit-pen-pcd-bc-retry-seed300-run2-20260929-v1` |

此前含 25-demo 行的 [公开结果参考图](../figures/adroit_pen_success_comparison.png) 留作历史图，不在当前 README 中代表最新成绩。该图混合公开论文与本地不同设置，不能作为严格排行榜。

2026-09-30 临时尝试的“25 demos 训练 / 每轮 100 条在线轨迹”只完成 RGB smoke；PCD smoke 在导入阶段失败。用户随后撤回该设置，代码回退；这批 smoke 未进入正式成绩。

最新已验收的 100-demo 成绩见 [README](../../README.md) 和 [实验记录](adroit_pen_100demo_random_control.md)。
