# 覆盖矩阵
主矩阵为六个视觉模型 × 四个环境；LeJEPA 只保留 Tiny 四组。State 四组在下方单列为 baseline。表格表示代码/配置覆盖，不表示训练或评测完成。
| 模型 | Cube1 | Cube2 | Cube3 | Cube4 |
|---|---|---|---|---|
| LeWM 2D / RGB | [代码与配置](models/lewm_rgb.md#cube1) | [代码与配置](models/lewm_rgb.md#cube2) | [代码与配置](models/lewm_rgb.md#cube3) | [代码与配置](models/lewm_rgb.md#cube4) |
| LeWM 3D / PCD | [代码与配置](models/lewm_pcd.md#cube1) | [代码与配置](models/lewm_pcd.md#cube2) | [代码与配置](models/lewm_pcd.md#cube3) | [代码与配置](models/lewm_pcd.md#cube4) |
| PLDM | [代码与配置](models/pldm.md#cube1) | [代码与配置](models/pldm.md#cube2) | [代码与配置](models/pldm.md#cube3) | [代码与配置](models/pldm.md#cube4) |
| VISReg ViT-B / CLS768 | [代码与配置](models/visreg.md#cube1) | [代码与配置](models/visreg.md#cube2) | [代码与配置](models/visreg.md#cube3) | [代码与配置](models/visreg.md#cube4) |
| V-JEPA 2.1 ViT-L / D1024 | [代码与配置](models/vjepa.md#cube1) | [代码与配置](models/vjepa.md#cube2) | [代码与配置](models/vjepa.md#cube3) | [代码与配置](models/vjepa.md#cube4) |
| LeJEPA Tiny / head512 | [代码与配置](models/lejepa.md#cube1) | [代码与配置](models/lejepa.md#cube2) | [代码与配置](models/lejepa.md#cube3) | [代码与配置](models/lejepa.md#cube4) |

## State baseline

| Baseline | Cube1 | Cube2 | Cube3 | Cube4 |
|---|---|---|---|---|
| State | [代码与配置](models/state.md#cube1) | [代码与配置](models/state.md#cube2) | [代码与配置](models/state.md#cube3) | [代码与配置](models/state.md#cube4) |

## 三种记录状态

- **原始命令已留存**：来自实际配置的 `commands`，或存档启动计划中的完整阶段对象；argv、环境、cwd、检查条件均保留原值。
- **原生代码/配置已收录**：具有原始实现和配置，但存档中没有该组单独的完整阶段命令；从原生入口和原始计划继续查看，不补造一条历史命令。
- **不适用**：State baseline 没有视觉 encoder、embedding 生产或缓存交付阶段。

LeWM 与 State 的早期入口、迁移后的 Single RL3 参考和新的 Foundation/Tiny 流程有不同的原生启动方式。目录整理保留这些差异，不把它们默认为同一次实验或同一套 RNG 流。
