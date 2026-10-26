# 原生 Adroit Pen：四条 RGB 基线与 BC 结果

本次完成了 PLDM、LeJEPA 的 Pen 数据预训练及其 BC，以及 VISReg、V-JEPA 2.1 发布权重的 frozen-encoder BC。任务均为原生 `pen-v0`，动作24维；没有使用 RH56 本体或其控制器。结果和逐 seed 的验收摘要见[可移植结果清单](../results/adroit_pen_baselines_20261026.json)。权重、示范数据、策略文件和原始运行报告均保留在外部存储，不随本代码发布。

## 结果与统计口径

每个方法使用 seeds `100/200/300`，每个 seed 训练100轮 BC、每次在线评测25条轨迹，共20次评测。表中成功率单位为百分数；`±` 是三 seed 的**样本标准差，ddof=1，单位百分点**。这与旧文档中可能使用的 ddof=0 口径不同，不能直接混写。

| 方法 | Best-of-20，seeds100/200/300 | Best 均值 ± 样本SD | Final，seeds100/200/300 | Final 均值 ± 样本SD |
|---|---|---|---|---|
| PLDM | 72 / 76 / 64 | 70.67 ± 6.11 | 52 / 76 / 64 | 64.00 ± 12.00 |
| LeJEPA | 68 / 68 / 64 | 66.67 ± 2.31 | 56 / 52 / 64 | 57.33 ± 6.11 |
| VISReg | 52 / 52 / 52 | 52.00 ± 0.00 | 8 / 24 / 44 | 25.33 ± 18.04 |
| V-JEPA 2.1 | 64 / 64 / 72 | 66.67 ± 4.62 | 52 / 44 / 48 | 48.00 ± 4.00 |

原循环采用0-based评测索引 `5,10,...,95,99`。Best-of-20 是这20次成功率的最大值；Final 是索引99时的成功率。每个 seed 在多轮评测中复用相同25个评测初态，不能将500次 rollout 当作500个独立初态，也不能将三 seed 均值称为合并轨迹成功率或置信区间。

原代码按 mean reward 和条件保存频率保存 `policy/iterations/best_policy.pickle`。这个文件的 SR 不自动等于 Best-of-20 或 Final；本次没有补造 `final_policy.pickle`。`train/*` 日志沿用现有入口的 eval 路径复用逻辑，不是独立的训练初态评测。

这里只报告本次 Pen 单任务结果。Relocate 未在本次四基线实验中执行，不计算 Adroit 两任务均值，也不将此表称为官方排行榜成绩。三 seed 结果没有建立方法之间的因果或统计显著性结论。

## 方法、数据与最小接口

实际生产代码为 `a10d76e260bfc750e47bef281d761c1d16418b03`，其配方来自 `cortexbench@6cec281879e8b866e567dc2e936072bb652b7326`；PLDM 框架版本为 `6f1e499e9cc0c898d326112f485c1062c3d20f24`。发布权重的加载与 BC 接口还核对过 `cortexbench@3de0e4caef82478a55a00acb42ea42c04fd33baa` 的相同 Git blobs。清单保留了这些真实生产版本，未把后续发布提交伪装成当时的运行版本。

| 方法 | 本次编码器来源与读出 | 特征维度 | BC 输入维度 |
|---|---|---:|---:|
| PLDM | Pen 预训练10轮；ViT-Tiny/14 CLS 经原 projector | 192 | 600 |
| LeJEPA | Pen 预训练10轮；ViT-Tiny/14 经原512维 head | 512 | 1560 |
| VISReg | 作者分享的 ImageNet-1K 发布权重；ViT-B/16 最后归一化 CLS | 768 | 2328 |
| V-JEPA 2.1 | Meta 发布的 ViT-B/16 蒸馏权重；EMA image path 的576个最后归一化 patch token取均值 | 768 | 2328 |

BC 输入沿用原 FLARE/history3 拼接，再加24维 proprio，即 `3 × embedding_dim + 24`，输出24维动作。没有加入 PCA、补零、截断或额外学习适配层。

- 预训练复用已有 Pen RGB collection：**10000 episodes / 914082个T+1观测行**。配对 RGB/PCD 是同一批轨迹，本次四条基线均走 RGB。
- PLDM 保留原四帧、`frameskip=5`、20行跨度、原 clip 随机90/10划分及动作条件动态模型、predictor、action encoder 和完整原生配置损失。划分单位是 clip，不是 episode-disjoint。动作末行 NaN 由原 normalizer/forward 处理，没有新增窗口过滤。
- LeJEPA 保留全部914082个有效视觉帧，包括终止图像；B256/drop_last 为3570步/轮。原四视图 CPU 增强、invariance/SIGReg、优化器和调度器保持；训练 projector 为16维，BC 使用512维 head。原流程没有 validation loop，不另造9000/1000划分。
- 另有150条独立测试 episode，没有用于预训练或 validation。
- BC 单独使用官方 `pen-v0.pickle` 的100条示范，每条100帧，共10000个样本；它与10000-episode WM collection不同。原 BC 的100轮、batch256、MSE、Adam lr1e-3及在线评测循环保持。
- VISReg 与 V-JEPA 2.1 使用发布权重冻结读出，没有本地10轮预训练。它们与 Tiny 方法在预训练数据、模型容量、输入分辨率、特征宽度和样本曝光上不同。

薄入口只处理数据/输出路径、父 HDF handle 生命周期和 worker/prefetch 调度；预处理、采样索引和学习核心复用原实现。默认只生成计划，显式 `--mode smoke` 或 `--mode formal` 才执行。PLDM 原入口的 logger 作用域兼容修复只作用于私有内存模块，不重写框架源码。

## 本次验收及恢复边界

PLDM 的真实第3轮 full checkpoint 保留了15273步，前3轮使用8 workers/prefetch2；后7轮由24/2接续新增35637步，合计50910个训练 batch、5660个 validation batch；两个 Adam 的累计 step 均为50910。清单的24/2字段描述最终采纳配置，具体生产历史另行列出。模型、优化器、调度器从 full checkpoint 恢复，但没有证明不中断 RNG/DataLoader 顺序等价。其他停止尝试和未保存尾部更新没有额外混计进已完成预算。

LeJEPA 的新24/2 run从原 seed0完整执行10轮，真实 AdamW 与 forward 计数均35700，十份原生 epoch 权重均有限；原 Encoder 的 backbone/head/proj 严格重载通过。native checkpoint 只含权重，不含 optimizer/scheduler，因此不声称它支持严格训练恢复。此前停止尝试的9000步没有计入这个新 run。

两条本地预训练的 native wrapper/observer 实际退出均0，epoch10经过原 BC loader 的严格加载。VISReg/V-JEPA 的权重、使用的子状态、冻结/eval、真实 Pen RGB 前向、原读出和零步原生环境接口分别验收；V-JEPA 的有限性记录只指使用的158个 EMA 张量，不指发布文件中未使用的全部训练 payload。

12个 BC run 都有100轮 loss记录、完整20次在线评测和真实子进程退出0；保存的 reward-selected policy 的 CPU重载对照 maxabs均0。每个 BC 的4000次 Adam 更新是由原循环计算的**期望预算**，本次没有运行 optimizer-step hook 或编码器训练前后哈希观察，不能把它们写成已直接测量的证据。

## 外部输入与来源

- WM `train_rgb.h5` header SHA256：`c85cc6368e4b4f9e485d3a817f68e0b9851f4c8accc4d3e75a7e589ed62cec2f`。VDS header 与它引用的 payload 必须一起保留；仅 header 的 SHA 不覆盖所有 payload 字节。
- 官方100-demo `pen-v0.pickle` SHA256：`cb6e122487f3abaed13e722eec9b4ba59441ea232ccb8a423c7e3a2c5fcfb557`。
- PLDM epoch10 SHA256：`5d983929ddc54e25cdc0c34586f8b68369843d9b2ed1d1904925a817ec118d1a`。
- LeJEPA epoch10 SHA256：`729e7a1d0c965defbd899263e2e1b63cc1d7c8460ffc311c42ec71d61d05d9e1`。
- VISReg 外部文件：`visreg-vit-b-inet1k.pth`，368495235字节，SHA256 `23e6aa5fe93cc61b3d9ab13e35cb4966580612d39bae39b1991f35c19c8087b8`。通过有权使用的作者分享渠道取得；本仓库不发布私有分享标识或权重文件。
- V-JEPA 官方[ViT-B/384权重](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)，1664223428字节，SHA256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`；官方源码 pin 为 `204698b45b3712590f06245fbfba32d3be539812`，使用该提交的[codeload归档](https://codeload.github.com/facebookresearch/vjepa2/tar.gz/204698b45b3712590f06245fbfba32d3be539812)，归档 SHA256 `e7e9c554c67e72c2c6884d751f4731bbf25d0d203b01d4b367b5157371b13ee1`。

上述权重 SHA 是本次原文件的内容指纹，不声称是发布方公布的 checksum。可移植清单还保留了240个 eval 成功率、逐 run 的 metrics/validation/policy/CPU probe SHA，以及两份外部源报告的全 SHA；不包含机器身份或绝对存储路径。

## 复用入口

WM 与 BC 使用分离的既有环境，依赖分别按原项目和原 CortexBench/eai-vc 配置。外部示范、已验收 checkpoint、MuJoCo运行资源及依赖不打包进仓库。以下是路径模板；输出必须选新的外部目录，保留失败尝试，不覆盖已有训练或结果。

预训练入口从 `lewm_pcd` 运行，`SWM_ROOT` 指向上述 pinned stable-worldmodel 源码；两个入口均通过原方法执行：

```bash
set -euo pipefail
WM_PY=/path/to/wm-env/bin/python
SWM_ROOT=/path/to/pinned-stable-worldmodel
PEN_RGB=/path/to/pen/published/train_rgb.h5
OUT_ROOT=/path/to/fresh-experiment-root
export PYTHONPATH="$PWD:$SWM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 只查看计划，不启动训练；LeJEPA同样支持 --dry-run。
"$WM_PY" scripts/baselines/pen/train_pldm.py --dry-run \
  --framework-root "$SWM_ROOT" --dataset "$PEN_RGB" \
  --output "$OUT_ROOT/pretraining/pldm" --workers 24 --prefetch-factor 2

# 两条线按顺序运行；每条先完成真实 smoke/退出检查再使用 formal。
"$WM_PY" scripts/baselines/pen/train_pldm.py --mode formal \
  --framework-root "$SWM_ROOT" --dataset "$PEN_RGB" \
  --output "$OUT_ROOT/pretraining/pldm" --workers 24 --prefetch-factor 2
"$WM_PY" scripts/baselines/pen/train_lejepa.py --mode formal \
  --dataset "$PEN_RGB" --output "$OUT_ROOT/pretraining/lejepa" \
  --workers 24 --prefetch-factor 2
```

短 smoke 使用相同参数、另一个 fresh output，改为 `--mode smoke --max-steps 2`。退出0或生成权重文件本身不等于完整训练验收；LeJEPA 原 smoke不保存native checkpoint。上述普通入口从头执行；本次PLDM的full-state恢复事实见前文，不承诺用bare weights恢复优化状态。

BC 使用现有 `run_bc.py` 和 embedding selector。`CORTEXBENCH_EAI_VC_ROOT` 是已有的外部 eai-vc 路径接口；沿用已配置 BC 环境中的 MuJoCo/渲染资源，避免把 WM 训练环境混入仿真依赖。V-JEPA pinned源码须位于现有 loader 预期的 `lewm_pcd/third_party/vjepa2`，可保留完整官方归档及版本凭据，不要求伪造 Git 元数据。

目标 `cortexbench-adroit` 的原 `run_bc.py` 保留已有 `CORTEXBENCH_EAI_VC_ROOT` 和 `base_seed=54321` 下的 `ADROIT_FRESH_EVAL` 兼容 hooks；与生产提交 `a10d76e` 的入口差别限于这些已有兼容处理，原学习核心保持。此次 seeds100/200/300 的参数透传路径已有契约测试；未对本次出版提交重新执行完整 GPU 训练，也不声称与生产版本 bitwise 等价。

```bash
set -euo pipefail
BC_PY=/path/to/bc-env/bin/python
LEWM_ROOT=/absolute/path/to/checkout/lewm_pcd
export CORTEXBENCH_EAI_VC_ROOT=/path/to/pinned-eai-vc
export PYTHONPATH="$LEWM_ROOT:$LEWM_ROOT/bench/cortexbench:$CORTEXBENCH_EAI_VC_ROOT/cortexbench:$CORTEXBENCH_EAI_VC_ROOT/vc_models/src${PYTHONPATH:+:$PYTHONPATH}"
export PEN_PLDM_CHECKPOINT=/path/to/accepted-pldm-epoch10-directory
export PEN_LEJEPA_CHECKPOINT=/path/to/accepted-lejepa-epoch10-directory
export STABLEWM_HOME=/path/to/external-released-weight-root
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=disabled
DEMO_DIR=/path/to/adroit-expert-v1.0
OUT_ROOT=/path/to/fresh-bc-results

# STABLEWM_HOME/checkpoints/visreg/visreg-vit-b-inet1k.pth
# STABLEWM_HOME/checkpoints/cortexbench_baselines/vjepa2_1_vitb_dist_vitG_384.pt
cd "$LEWM_ROOT/bench/cortexbench"
for embedding in pldm_pen lejepa_pen visreg_vitb vjepa21_vitb; do
  for seed in 100 200 300; do
    out="$OUT_ROOT/$embedding/s$seed"
    test ! -e "$out" || exit 1
    "$BC_PY" run_bc.py --config-name Adroit_BC_config \
      env=pen-v0 embedding="$embedding" seed="$seed" data_dir="$DEMO_DIR/" \
      hydra.run.dir="$out" hydra.job.chdir=true job_name=policy \
      wandb.project=local wandb.entity=local
  done
done
```

各方法使用自己的已验收 checkpoint；先确认前一条真实退出和资源释放，再开始下一条。`workers24/prefetch2`属于预训练读取设置，不把它硬套到小型 BC head 或在线环境中。
