# 环境与复现边界

[返回总览](../../README_TEST_WORKFLOWS.md)

## 实际来源

本次只读采集使用本机 `ssh test`，端点为 `root@connect.westc.seetacloud.com:16755`。原生源码根目录：

```text
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
```

代码曾由较早的 Test 端点迁移而来，目录名和原始记录中的端口保留历史含义。来源时间和逐文件 stat/SHA 在 `source_map.json` 中。

## 环境

采集时的现有下游环境为 Python 3.12.3；安装元数据为：

- flax: `0.11.2`
- jax: `0.7.2`
- jaxlib: `0.7.2`
- numpy: `2.2.6`
- pytest: `8.4.2`
- stable-pretraining: `0.1.8`
- torch: `2.8.0`

该版本表记录采集环境，不能代替 CUDA、算子或完整实验验收。当前 Test 容器为 CPU-only，资源上限 0.5 CPU / 2GiB，本次不声称 GPU 或数值实验验证。

原生 `requirements.runtime.lock` 和启动环境记录随源码保存。World-model 与下游 MPC/GCIQL 的环境继续分离；PLDM 未改动的上游 author factory 仍按原有 Python 3.10 环境使用，不因这里的下游版本表改换依赖。

JAX 的 CPU reload、CUDA train/gate，Native 的 JAX CPU + Torch CUDA，以及 EGL/library path、Threefry、线程与 `async100` 等阶段设置，均从原始计划读取，不由索引工具统一覆盖。

## 路径和外部资产

材料化工具只还原原生相对布局，不改写源码中的绝对 ROOT、原始 JSON、producer provenance、checkpoint/cache SHA 或历史输出路径。需要执行原生入口时，可在新的主机/容器按原生根目录还原代码，或另做显式、可审阅的路径适配并重新完成对应校验。

官方 NPZ、HDF5/VDS backing、模型权重、缓存向量、已保存策略、验收 gate batch 和运行日志是外部资产。原始配置给出路径和哈希；本次不复制大文件，也不把这些路径目前是否可用标成通过。

每次新实验需要独立输出命名空间和与其实际 source/config/data 绑定的新验收记录。存档命令继续指向旧输出的事实会在只读计划中显示，不会被工具静默改写。

## 本次验证范围

新增测试验证 24 组视觉对比和独立的四组 State baseline、实际配置/源文件哈希、输入维度、原始阶段对象一致性、只读查询、越界/覆盖拒绝和全量字节还原。既有实现的历史训练/数值/正式 SR 证据不被本次测试替代。最终本地验收见仓库的 `UNIFIED_ACCEPTANCE.json`。
