# Test 服务器：Cube 训练与下游流程

主矩阵为 **6 个视觉模型 × Cube1–4，共 24 组**：LeWM RGB、LeWM PCD、PLDM、LeJEPA Tiny、VISReg、V-JEPA。LeJEPA 的四组使用 Tiny 配方，已替换之前的版本。**State 四组单独作为 baseline**，不计入这 24 组。

这里保存原生实现、实际配置、阶段命令和来源绑定。各组的实验完成状态、成功率与模型验收不由本次代码整理重新认证。数据集、特征数组、模型权重和训练日志仍是外部实验产物。

## 从哪里开始

- [总覆盖矩阵](docs/test_workflows/COVERAGE.md)：按模型和 Cube 查看入口与配置。
- [阶段与使用方法](docs/test_workflows/USAGE.md)：缓存、训练、验收、重载和 Native SR。
- [数据与模型契约](docs/test_workflows/CONTRACTS.md)：官方 NPZ、真实 terminal、相机、特征宽度和原有 seed/RNG。
- [环境与复现边界](docs/test_workflows/REPRODUCIBILITY.md)：依赖、原生绝对路径及外部资产。
- [来源与清理规则](docs/test_workflows/PROVENANCE.md)：逐文件哈希、代码复用、历史版本依赖和本次验收。
- [原有世界模型训练 switch](README_UNIFIED.md)：保留的 PLDM、LeJEPA Tiny、LeWM 2D/3D 上游训练入口。

## 只读检查

从仓库根目录运行，以下命令只读取代码/配置，不加载模型，不开始训练或评测：

```bash
python lewm_pcd/scripts/inspect_test_workflows.py --list
python lewm_pcd/scripts/inspect_test_workflows.py --verify-sources
python lewm_pcd/scripts/inspect_test_workflows.py --model vjepa --cube-count 4 --stage gciql
python lewm_pcd/scripts/inspect_test_workflows.py --model visreg --cube-count 3 --stage native-sr
```

该检查入口没有 `--execute`。保留的原生代码仍可用于后续明确授权的实验；原始启动参数和历史输出路径原样留存，复现时需要重新绑定新输出目录并通过对应原生校验。
