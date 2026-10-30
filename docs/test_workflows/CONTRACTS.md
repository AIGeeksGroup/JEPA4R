# 数据与模型契约

[返回总览](../../README_TEST_WORKFLOWS.md)

## 数据边界

Test 下游使用各组绑定的官方 Play NPZ 与按相同 raw ID 对齐的冻结特征。官方每 episode 保留 1001 个原始观测和 1000 个转移；末端 next 使用真实 terminal，不用上一帧复制。各组实际 NPZ、统计、episode offset/length、缓存与 checkpoint 的哈希以原始配置为准。

Foundation 公共缓存任务的 Cube1/2 使用 1000 train + 100 val episodes，Cube3 使用 3000 + 300，Cube4 使用 5000 + 500。早期组保留其实际配方；不依据表格为它们补写新的数据绑定。

原有世界模型训练的 collected HDF5/VDS、Single LeWM raw201/window split 与下游官方 NPZ 是分开的协议，不能互换。它们继续由 [v3 训练说明](../../README_UNIFIED.md) 管理。

## 原生输入维度

| 模型 | latent D | current | goal |
|---|---:|---|---|
| State | 0 | `19+9N` | `3N` |
| LeWM RGB / PCD | 192 | 211 | `192+3N` |
| PLDM | 512 | 531 | `512+3N` |
| VISReg | 768 | 787 | `768+3N` |
| V-JEPA 2.1 | 1024 | 1043 | `1024+3N` |
| LeJEPA Tiny | 512 | 531 | `512+3N` |

N 为 Cube 数量。视觉 current 使用原生 latent + proprio19；goal 使用同 raw ID 的目标 latent + 各 Cube XYZ。保留原始 object-GT 排除策略，不做 PCA、零填充或公共投影。Tiny policy 输入为 head512，训练 projector16 不是下游输入。

## 模型与数值

- LeWM RGB/PCD、PLDM、Tiny 使用各组精确绑定的 checkpoint 与原生预处理。PCD 的几何、颜色、裁剪、点数和 factory 保持原值。
- VISReg 使用原生 ViT-B/16 最后层 CLS768；V-JEPA 使用官方完整 forward 和最后层 patch mean1024。当前 V-JEPA FP16 编码、FP32 特征存储与较早的精度诊断版本分别留有来源，不混作同一次验收。
- VISReg/V-JEPA 在 Test 使用冻结官方预训练权重；本次收录不增加一套从头预训练它们的流程。
- GCIQL 的采样、目标 RNG、距离奖励、mask、agent/update/optimizer 与模型保存主体均保留原实现。通常的 fresh seed0、B1024、1M 更新、验证/保存间隔以各组真实配置为准，不用统一默认值覆盖旧组。
- Single RL3 的 M5/M6 保留原有中间评测省略及 RNG 偏离说明；State 的原始官方入口和自定义距离奖励入口也分别保留。

## 相机与评测

Cube1 保留其实际 `front_pixels` 绑定；Cube2/3 的已绑定配方保留 front 上抬 5°，Cube4 保留额外后移 0.1m。每组实际 position/quaternion/FOV、224 RGB、AA、PCD crop 以配置/producer 为准，不用概括描述代替缺失姿态证据。

Native 保留原任务、seed、episode 上限、策略与在线 encoder 路径。最终 1M checkpoint 的 Native250、诊断 tiny 模式和额外 checkpoint sweep 各自保留原生入口；代码目录存在不表示任何一项评测完成。
