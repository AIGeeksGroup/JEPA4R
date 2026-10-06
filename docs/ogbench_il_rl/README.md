# Official Cube → Frozen LeWM → single-frame MLP GCBC

实验顺序、数据/smoke/正式训练/全量SR的阶段进度及下一步统一见[根README](../../README.md)。本页保留输入契约与代码使用方式。

## 数据与模型契约

上游 encoder＋projector 来自 custom fixed5 的10000条世界模型预训练数据。下游动作头使用官方 `cube-single-play-v0` 的1000条 train、100条 val；每条1001原始行、1000有效 transition。官方 actions/terminals 不修改。Cube state 与 Visual Cube 是两份独立轨迹，不能混配。

RGB224/PCD从 Cube 包自身 qpos/qvel逐行 restore＋forward派生。这个路径用于生成静态观测，不声称只凭 qpos/qvel能够精确恢复完整动力学 rollout。state observations不输入我们的 RGB/PCD policy。

当前接口为 **单帧 current → frozen192** 与 **单帧 goal → frozen192** 拼接384，接512×3 MLP输出 mean5。每次决策都做两个真实 encoder/projector前向；不缓存 encoder features，BN/dropout保持冻结，无额外 policy warmup。

MLP跟随固定官方 GCActor 的 GELU(tanh approximation)、fan-average uniform初始化、hidden scale1 / mean scale0.01、zero bias、固定std1。目标是原始归一化 Box动作的 Gaussian NLL：动作维求和，样本求平均；Adam3e-4、weight decay0。训练不做 action ZScore，推理返回未裁剪 mean，原官方 evaluate沿用自身[-1,1] clip。Torch seed0与Flax PRNG不等价，跨框架对照使用复制的相同权重。

## API

按 [依赖说明](DEPENDENCIES.md) 将 `lewm_pcd` 与锁定 stable-worldmodel源码加入 PYTHONPATH：

```python
from ogbench_il_rl.models.policy import create_policy, make_optimizer, unit_gaussian_nll
from ogbench_il_rl.data.dataset import CubeOfficialDataset, LogicalSampler

policy = create_policy(source_checkpoint=world_model_epoch10,
                       modality='pixels', history_size=1, seed=0)
optimizer = make_optimizer(policy)
mean = policy(current_rgb_tensor, goal_rgb_tensor)  # BCHW → B5
loss = unit_gaussian_nll(mean.float(), raw_action_tensor.float())
```

`CubeOfficialDataset`读取已经独立验收的 `dataset_index.json`及其不可变 RGB分片。输入预处理须从原 epoch10配套配置实例化。使用 `LogicalSampler`传入(current, goal)索引对，原始terminal行不进入 compact当前/目标集合，末个有效 transition允许同位置goal；随机流与四舍五入边界以实际官方 GCDataset 为准。

`LogicalSampler`保存已消费边界、完整order、预取goal plans和RNG状态，优化器提交一个逻辑批次后才调用 `commit_logical_batch`。当前预算是micro64×16＝有效1024；尾批576按实际样本权重累积，1M样本不丢弃，每epoch977更新。正式10epoch训练循环仍待主实验完成；本次发布不提供旧history3/MSE训练入口。

`modality='pcd'`的policy形状接口为 BN6，但当前正式dataset/native RGB接口尚未完成PCD适配验收，不能仅改字符串就声称完成3D训练。

## 派生与在线评测

`data/rgb_shards.py`保留原数值NPY无损缓存、Zstd9/chunk10 RGB分片写入、source IDs、SHA与提交前缀恢复。`data/rgb_renderer.py`接收外部reference config/status，校验Python/依赖/环境源码、scene XML和assets，再逐行渲染；reference需包含对应锁定运行的版本和SHA。数据和reference运行产物不随源码上传。

点云 `sample_foreground`正常路径保持原裁剪/体素/采样顺序精确；仅裁剪池为空时回退到同camera真实未裁剪foreground。同voxel/quota/RNG采1200点，raw foreground为空仍fail，不加背景、零点或丢帧。配对val pilot仅757–759三帧回退，少点允许真实重复采样；新MLP的3D真实数据/精度/效果尚待验收。

`evaluation/native_single_frame.py`保留真实当前RGB与独立goal MjData渲染接口，重放已归档的两次内部Box reset动作，核原state基线的完整初态/goal/ctrl/mocap/private/RNG。单凭相同seed不能声称跨模态物理初态一致。其CPU准备入口只验收参考与协议，不生成新SR；真实环境/goal-render/短rollout门及正式250episode SR仍待运行。

`evaluation/official_state_entry.py`只适配官方源码位置、数据位置及现代WandB离线日志，其余参数传给原 `impls/main.py`。最终参数与原源码 SHA见 `configs/official_state.json`。

`evaluation/official_state_sr.py`要求已完成run的 `plan.json`、`result.json`、`final_cpu_restore.json`、相应退出码及checkpoint同目录的 `flags.json`，校验最终100万步和完整恢复后调用原evaluate。指定 `--source --data --train-root --output`，CPU依赖可用 `--cpu-deps`；按 `--stage gate`、task1至5、aggregate执行，接口诊断排除于最终250条分母。

## 配置与验证范围

`configs/gcbc.json`记录最新单帧MLP/NLL/Adam、seed0、10epoch、micro64/effective1024。RGB派生已选16worker；训练reader暂保留此前4worker，新MLP全数据性能门尚未完成。worker测速过程不上传。

2026-10-06，发布副本CPU正确性测试 **50 passed，无跳过**，包括官方goal sampler数值对照、预取断点恢复、冻结/更新/严格factory恢复、1024与576累积、Flax头/loss/梯度及同梯度Adam数值对照、RGB分片损坏拒绝、物理参考重放与上游配对采集协议。CPU测试不能替代新协议的真实数据CUDA/BF16或正式训练结果。

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  OGBENCH_SOURCE=/path/to/pinned/ogbench \
  OFFICIAL_OGBENCH_SOURCE=/path/to/pinned/ogbench \
  OGBENCH_CPU_DEPS=/path/to/optional/cpu-deps \
  python -m pytest lewm_pcd/ogbench_il_rl/tests \
    lewm_pcd/scripts/data/test_collection_protocol.py \
    lewm_pcd/scripts/data/test_paired_writer.py -q
```
