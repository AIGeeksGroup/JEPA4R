# Official Cube → frozen visual encoder → GCBC

实验顺序、数据/smoke/正式训练/全量SR的阶段进度及下一步统一见[根README](../../README.md)。本页保留输入契约与代码使用方式。

## 最新state融合接口

三路线共享 `models/state_fusion.py`，原 `models/policy.py` 的visual-only行为不改。默认 `goal_state_mode='full'` 保留完整goal28；本次配置显式选择 `'object_xyz'`，在policy内部clone后仅保留19:22，保持440／824维checkpoint形状。data wrapper和在线actor仍传完整goal28，避免重复mask或换成privileged target。

```python
from ogbench_il_rl.models.state_fusion import create_policy, make_optimizer, unit_gaussian_nll
from ogbench_il_rl.data.state_fusion import CubeStateVisualDataset
from ogbench_il_rl.data.dataset import CubeOfficialDataset, LogicalSampler
from ogbench_il_rl.data.dataset_pcd import CubeOfficialPCDDataset

policy_2d = create_policy(source_checkpoint=rgb_wm_epoch10, modality='pixels',
                         goal_state_mode='object_xyz', seed=0)
policy_3d = create_policy(source_checkpoint=pcd_wm_epoch10, modality='pcd',
                         goal_state_mode='object_xyz', seed=0)
policy_dino = create_policy(model_folder=local_dinov2_small, modality='pixels',
                           latent_dim=384, goal_state_mode='object_xyz', seed=0)
# 为每组提供各自原始预处理和同源visual reader。
visual = CubeOfficialDataset(rgb_index, 'train', original_rgb_preprocessor)
paired = CubeStateVisualDataset(visual, train_observation_npy,
                               expected_sha256=train_observation_sha256)
# 将paired和原LogicalSampler用于已有DataLoader/训练循环。
mean = policy_dino(batch['current'], batch['goal'],
                   batch['current_state'], batch['goal_state'])
loss = unit_gaussian_nll(mean.float(), batch['action'].float())
optimizer = make_optimizer(policy_dino)  # trainable参数严格只有action_head
```

PCD换用 `CubeOfficialPCDDataset`，预处理来自原PCD世界模型config中的PointCloudPreproc。它保留原1200点＋颜色、通用`pixels`别名和相同trajectory采样，不改点云提取主干。

state NPY必须来自visual index相同的官方Cube NPZ／split，按完整raw1001行排列，float32、shape `(episodes*1001,28)`。wrapper按 `source_row`／`goal_source_row`读取，不能用compact索引直接查raw数组；terminal行1000排除，最后有效current999允许goal999。基础reader同时核请求的episode和row，显式 `expected_sha256` 可绑定已验收的state文件。目录名或相同shape不能独立证明同源，原始NPZ／index／seal验收仍由调用者负责，Visual Cube另一份轨迹不能混配。

配置 [state_fusion.json](../../lewm_pcd/ogbench_il_rl/configs/state_fusion.json) 包含三组factory、reader和共同batch256／100K预算。`source_checkpoint`／`model_folder`的null需填入实际资产路径。DINO只从本地完整模型严格加载，RGB沿原完整视野224／ImageNet预处理，不做HF默认中心裁剪或重复归一化；内部LayerNorm保持，不加CLS/state归一化、Fourier或特征缓存。

完整历史训练与续跑调度不复制到本包；四参数forward可接入已有训练循环，JSON的`_target_`用于factory重建。保存时须同时保留完整state_dict与factory配置，包含 `goal_state_mode`、`latent_dim`和资产路径。mask模式不增加weight tensor，只换权重文件无法证明所用模式正确；新的对照头fresh seed0。

RGB native actor显式使用 `SingleFrameAgent(..., state_fusion=True)`，转发官方evaluate的真实current／goal observation28为float32 B28；默认False保留原入口。mean5／clip／ctrl7、reset、独立goal MjData及物理配对流程不改。3D policy的`get_action`接受真实PCD和完整state；在线PCD provider与DINO融合完整SR的预算metadata驱动需现有评测程序提供，新增接口单测不等于完成250条SR。

2026-10-11发布副本CPU验收：**34 passed，无跳过**，覆盖原包回归和三路线新增接口。额外使用真实本地DINO-small与冻结运行版本逐项比较，完整state、mean、loss及动作头梯度精确一致，未使用GPU或执行optimizer step。这些验收不代表新100K训练完成或最终SR。

以下10epoch／batch1024说明保留为旧visual-only接口历史，不作为本次state融合预算。

## 数据与模型契约

上游 encoder＋projector 来自 custom fixed5 的10000条世界模型预训练数据。下游动作头使用官方 `cube-single-play-v0` 的1000条 train、100条 val；每条1001原始行、1000有效 transition。官方 actions/terminals 不修改。Cube state 与 Visual Cube 是两份独立轨迹，不能混配。

RGB224/PCD从 Cube 包自身 qpos/qvel逐行 restore＋forward派生。这个路径用于生成静态观测，不声称只凭 qpos/qvel能够精确恢复完整动力学 rollout。原visual-only接口不输入state；上述显式融合接口加入同源state。

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
