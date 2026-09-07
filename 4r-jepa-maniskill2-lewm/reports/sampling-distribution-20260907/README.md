# 采样分布审计（固定快照334）

入口为 `report.html`。主分析覆盖3331个完整回合、666200个状态；点云可见性抽查32回合、6400帧。所有数据均来自已经写入的真实采样，未修改运行中的采集或训练。

关键结果：99.97%回合曾抬起方块至少5cm，首次抬起中位0.85秒；93.0%回合曾满足原生成功条件，但成功状态只占9.95%帧；30.4%回合在首次成功前回落至桌面。回落可能是主动释放，不直接等同抓取失败。旋转指令99.59%的平方幅度集中在yaw，缺乏独立roll/pitch探索。视觉抽查中方块通常可见，不能据此证明遮挡泛化。

成功依据保存的物理状态按原生距离/关节速度条件重算，与reward=1逐帧完全吻合；旧布尔列缺失时不推断native grasp。动作与状态配对排除每回合末行的动作回绕。上游的 `obs_frame=world` 实际使用base系，物理cube/goal统计使用world系；坐标差异已核对。

从项目根目录重跑主分析（需已有数据及CPU依赖）：

```bash
source scripts/wm_env.sh
.venv-worldmodel/bin/python scripts/analyze_sampling_distribution.py   artifacts/wm_collection_full_20260907T044524Z/pickcube_expert_mixed.lance   --snapshot-version 334 --output artifacts/distribution_audit_20260907
```

`distribution_audit.ipynb` 已逐个代码单元顺序执行，包含固定版本重算和一致性检查。数据集本身不随Git分发。HTML已通过内容与结构检查；增强图表浏览器检查未通过，完整静态表格与文字仍可阅读，见 `report-qa.json`。

本报告评估数据覆盖，不等于已经验证世界模型的预测或冻结表征PPO性能。没有基于这份报告自动更改采样策略，也没有人为增加失败比例。
