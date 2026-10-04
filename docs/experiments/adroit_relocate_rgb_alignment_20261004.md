# Relocate / Pen 2D RGB 下游 BC 对齐审计

审计日期：2026-10-04。主 agent 与三个子 agent 在授权范围内只读核对 927-5090 的配置、源码、状态、日志、指标和现有报告；没有修改远程代码、启动模型计算、仿真、训练或评测。

发布证据：[当时读取的 80 项远程配置、状态、指标与源码快照及 SHA](../results/adroit_relocate_rgb_alignment_sources_20261004.json)；[后续完整六 seed 指标和 repair 结果](../results/adroit_relocate_100demo_20261004.json)。下面所述 `runtime_metadata.json` 和 `pen_version_sources.json` 的本地完整位置为本 Git checkout 的父目录下 `artifacts/relocate_pen_rgb_alignment_20261004_readonly_v1/`；这些环境元数据文件未单独发布，也不能由当前静态版本推断历史运行环境未变化。

结论：主要 BC 超参数、图像方向、history/proprio 和动作标签接口已获得对齐证据。尚未找到能解释整条 Relocate 曲线低 SR 的确定性根因。已确认的差异包括运行环境、预训练轨迹来源和优化更新预算；另有 checkpoint 保存规则问题，以及两条线共有的 WM→BC resize 差异。

## 1. 实际结果与有效运行

仅对照本次 Relocate 与历史 Pen 的 100 demos / 25 eval；不混入后续 Pen 25/100 smoke、随机 encoder 对照或 PCD。

| 运行 | seed | Best-of-20 SR | final SR | 完整性证据 |
|---|---:|---:|---:|---|
| Relocate formal RGB CUDA v2 | 100 | 8% | 0% | 100 loss、20 eval、最后 epoch99、exit0、validation/policy 存在 |
| Relocate formal RGB CUDA v2 | 200 | 8% | 0% | 同上 |
| Relocate repair seed300 v1 | 300 | 8% | 4% | 同上，repair launcher exit0 |
| Pen 历史 pretrained RGB | 100 | 48% | 44% | 同上 |
| Pen 历史 pretrained RGB | 200 | 44% | 36% | 同上 |
| Pen 历史 pretrained RGB | 300 | 56% | 32% | 同上 |

三 seed Best-of-20 平均：Relocate 8%，Pen 49.33%。两个任务的难度、状态分布和成功定义不同，这个差距用于定位流程差异，不能单独证明实现错误。

原 Relocate v2 的 seed300 缺 unique best_policy.pickle，seed/launcher 验证失败并 exit1；有效汇总必须使用 repair root 的 seed300。修复解决保存完整性，未提高整条曲线的最高 SR。

## 2. 已对齐的配置和输入输出

两条线有效 job_config 均为：100 demos、25 eval trajectories、100 epochs、eval_frequency5、save_frequency10、seeds100/200/300、history_window3、add_proprio=true、vil_camera、256×256 RGB、BatchNormMLP、hidden(256,256,256)、ReLU、MSE、batch256、lr0.001、dropout0。

BC 调用契约：

`RGB → resize224/ImageNet → frozen CLS + projector(192D) → FLARE(576D) + 当帧 proprio → BatchNormMLP → 当帧 raw action → 环境裁剪与 actuator 映射`

- FLARE 是 `[z(t-1)-z(t-2), z(t)-z(t-1), z(t)]`；首帧历史复制，reset 清空 history。
- Pen proprio/action 为24/24，BC 输入600D；Relocate为30/30，输入606D。这是正确任务适配。
- WM 的 action_encoder 输入120→150，来自 frameskip5×任务原子动作维度24→30；BC仍输出单步24/30D动作，不输出150D动作块。
- 两个 WM 保存配置均为 ViT tiny、patch14、224、embed192、history3、frameskip5、10epochs、BF16、AdamW lr5e-5、batch128、SIGReg weight0.09。

## 3. 已确认存在的差异，因果影响尚未验证

### 3.1 运行环境

| 当前环境文件 | Pen 历史命令指向的环境 | Relocate 本次 CUDA 环境 |
|---|---|---|
| Python | 3.8 | 3.10 |
| Torch | 1.13.0，cuda=None | 2.8.0+cu128 |
| torchvision | 0.14.0 | 0.23.0+cu128 |
| Transformers | 4.43.4 | 5.17.0 |
| Hydra | 1.3.2 | 1.3.6 |
| 设备配置 | cpu / data_parallel=false | cuda / data_parallel=true |

版本来自本次静态读取的环境文件，不能证明历史运行时环境从未变化。data_parallel 在当前 BC 核心没有发现消费位置，不能把这个配置字段直接归因于低 SR。

本次 CUDA 日志证明 encoder/head 位于 CUDA、记录的梯度 finite、MuJoCo rollout 位于CPU。没有发现新增的明显设备或 train/eval 切换错误。它们不证明 CPU/CUDA 或跨库版本输出相等。

现有 RGB smoke 只用1demo/1eval/1epoch。已有 CPU/CUDA paired smoke 报告属于 PCD，不能替代这次 RGB 的数值对照。

### 3.2 相同 epochs 对应不同优化预算

Pen 官方 BC 为100×100=10,000帧，每epoch40 minibatches，共4,000次Adam更新；Relocate为100×200=20,000帧，每epoch79 minibatches，共7,900次更新。DataLoader 未 drop_last。

Relocate最终训练MSE约1.7e-4～1.8e-4，而在线SR低；这只证明训练标签拟合损失下降。任务动作尺度不同，不能将Pen与Relocate的MSE绝对值直接比较，也不能据此确认过拟合或表征坍塌。

### 3.3 WM 预训练轨迹来源不同

- Pen 使用按目标轴匹配官方demo后逐时刻取动作的策略；扰动后动作不依赖当前状态反馈。10,000episodes、901,470帧，长度27～100；掉笔会提前终止。
- Relocate使用官方 state policy，根据当前39D状态闭环生成动作。10,000episodes、2,000,000帧，固定200步。
- 50/30/20的采样类型比例接近，但轨迹长度、专家反馈机制、扰动后的恢复和状态覆盖不同；因此不是相同WM输入分布。
- 静态采集链是pre-step RGB/state→生成动作→真实step→保存当前观测及同一执行动作。噪声确实进入执行，失败轨迹保留。没有发现事后修改action或新增时序错位。

### 3.4 预训练到 BC 的 resize 差异

已确认两条线共有：WM 使用 tensor ToImage/ImageNet normalize 后 BILINEAR resize；BC使用PIL BICUBIC resize 后ToTensor/ImageNet normalize。ImageNet统计相同，resize核和处理路径不同。

这不是Relocate专属新增差异，不能直接解释其相对Pen的低SR；embedding影响尚未做数值验证。

## 4. 已确认的保存和报告边界

### 4.1 best_policy 以 reward 选取，汇总以 SR 选取

当前共享 train_loop.py:250-263、318-323 按最高mean reward更新 best_policy，highest_success独立统计。旧条件还要求碰上save_frequency10，可能漏掉在epoch5/15/25等产生的最高reward；新增is_new_best_eval修复了这种漏存。

例如Relocate repair seed300的最高reward在epoch55，该点SR4%；它的Best-of-20 SR为8%。当前保存规则不能保证保存最高SR点。

子agent只读解析现有policy的BatchNorm计数，推断Relocate seed100保存于epoch40（该点日志SR0%）、seed200于40（8%）、repair300于55（4%）。这是由num_batches_tracked与每epoch批数推断，尚未重新加载评测验证，不能直接宣称其实际重评SR。

保存问题可解释使用现存policy时与曲线Best-of-20不一致；不能解释Relocate所有评测点只达到0～8%。Pen历史也有同类保存规则问题。

### 4.2 train/success 复用在线 eval paths

run_bc.py:76-89将rollout_from_init_states替换为最近一次eval paths。train/success与eval/success相同不能证明策略在训练demo初始状态上成功；这项独立闭环验证没有发生。

### 4.3 图像 audit 覆盖有限且 PASS 没有数值门槛

现有RGB图像audit只检查3demo×前3frame，共9帧。vertical flip后mean_abs约0.058、99.6%左右像素精确相等；direct mean_abs约59，支持当前图像方向。脚本status=PASS直接写入，没有阈值assert，不能称全量图像或embedding一致性通过。

### 4.4 reload gate 的有效范围

v6 full WM报告证明strict完整state_dict、真实数据encode/predict/loss finite、同环境两次reload输出差0。legacy embedding报告只证明一帧HDF5→FP32[1,192] finite；没有比较原WM编码器与BC重建编码器输出、离线batch与在线single输出。

### 4.5 当前源码相同不能证明历史源码相同

当前Pen旧root与Relocate快照的run_bc.py相同；2D loader分支相同。两个third_party/eai-vc实际共享同一个eai-vc-reference symlink。历史Pen文件存在后续修改时间，缺少独立封存版本，不能从现在的hash相同推回历史执行源码相同。

## 5. 动作层排查证据

- 正式Relocate与repair命令使用官方raw demo目录，SHA记录匹配用户给出的c71c7af5...；没有误读clipped probe/target-demo。
- action audit对20,000行raw-target逐元素一致；raw范围约[-1.513,1.528]。
- 将raw裁剪到[-1,1]改变90.05%的行、5.13%的元素；前3demo共600步raw/clipped重放后继状态差不超过约7.8e-15，因为环境内部裁剪。
- 官方raw标签直接做MSE、无额外tanh/动作缩放；Pen使用相同执行语义。不能以raw超出1为理由直接rescale。
- 抽查支持当帧proprio/state/action对齐；它不覆盖全量encoder/FLARE输出一致性或BC闭环任务成功。

## 6. 下一阶段建议；本次未执行

1. 固定同一RGB checkpoint、官方demo帧和输入tensor，对比原WM与BC loader、旧CPU与新环境CPU/CUDA、batch与single的embedding，以及同一BC policy动作。先定位差异发生在preprocess、encoder/projector还是head；不能先混改插值、动作与训练协议。
2. 数值门禁通过后，在新CUDA栈用已有Pen权重/官方demo做一个相同100/25协议的单seed对照，用已成功Pen线校验新栈。若Pen也下降，优先定位环境/实现；若Pen稳定，再查Relocate表征和数据分布。该步骤需要后续明确执行授权。
3. 对Relocate做独立的训练初态闭环诊断、留出demo逐关节动作误差、目标位置敏感性分析。现有train/success不能替代该诊断。
4. 明确区分最高reward policy、最高SR policy和Best-of-20统计；需要修改保存语义时另行授权并保留旧产物。

## 7. 本地证据

- evidence.json：本次读取的80个远程配置、状态、指标、audit与源码内容，含各文件SHA、原路径和symlink realpath；缺失项为原formal seed300的validation.json。
- runtime_metadata.json：当前环境包METADATA；Pen Torch/torchvision没有dist-info，精确版本补在pen_version_sources.json。
- 远程仅做读取；长期盘当前100%使用，未新增远程数据或文件。

关键远程源码：

`/root/autodl-tmp/workspace/users/lewm/eai-vc-reference/cortexbench/mujoco_vc/visual_imitation/train_loop.py`

`/root/autodl-tmp/workspace/users/lewm/adroit_full/relocate_official_10000_150_20261001_v1/recollect_code_target_v1/lewm_pcd/bench/cortexbench/run_bc.py`

`/root/autodl-tmp/workspace/users/lewm/adroit_full/relocate_official_10000_150_20261001_v1/recollect_code_target_v1/lewm_pcd/bench/cortexbench/lewm_cortex.py`

`/root/autodl-tmp/conda/envs/jepa4r-smoke/lib/python3.10/site-packages/stable_pretraining/data/transforms.py`
