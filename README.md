# 点云世界模型（PTv3 空间聚合 + 时间 Transformer + HMA 动作条件预测器 + LeWM 损失）

一个 `nn.Module`（`models/world_model.py: WorldModel`）就是整条流水线：

```
点云序列 [B, T, K, C]
  → SpatialAggregator（逐帧，仓库 A: Pointcept PTv3 v3m2）          → latent1 [B, T, K, D]
  → TemporalEncoder（+ 位置编码(k) + 时间编码(t)）                    → latent2 [B, K, D]
  → Predictor（架构与动作输入来自仓库 B: HMA；损失来自仓库 C: LeWM）  → 预测的下一帧 latent1 [B, K, D]
```

每帧 K 个点、聚合后 K 个向量，一个点对应一个向量；全程向量维度统一为 D。

## 安装

要求：Python 3.11、NVIDIA GPU（驱动需兼容 CUDA 12.6，RTX 50 系用 12.8）。**训练和测试以 CUDA 为准**（spconv 的 GPU 算子只在 CUDA 上有完整实现；
没有 CUDA 时代码会回退到 CPU 上 spconv 的 Native 算法，只适合跑形状测试和冒烟，很慢）。
所有 CUDA 算子都用预编译轮子，**不需要本地编译**：

```bash
# pip（一条命令）
make install                                   # = pip install -r requirements.txt && pip install -e . --no-deps
make install REQ=requirements-cu128.txt        # RTX 50 系（Blackwell，sm_120）用这条

# 或 conda
make env                # = conda env create -f environment.yml   （环境名 pcwm，内容与 requirements.txt 一致）
conda activate pcwm && pip install -e . --no-deps
```

`requirements.txt` 锁定了版本与索引：`torch==2.9.0+cu126` / `torchvision==0.24.0+cu126`（官方 cu126 索引；torchvision 是 timm 的依赖，
必须与 torch 同版本同 CUDA）、`spconv-cu126==2.3.8`（PyPI 预编译）、`timm`、`addict`、`einops`、`omegaconf`、`pytest`。
不装 flash-attn、xformers、muP，也不再依赖 torch-scatter（见下面 A 的改动栏）。

- 没有 `make` 的机器（例如 Windows）直接跑 Makefile 里的两条命令即可。
- **RTX 50 系（Blackwell，sm_120）**：sm_120 需要 CUDA ≥ 12.8，cu126 的 torch 在这类卡上会报 `no kernel image is available`。
  `requirements-cu128.txt` 换成 cu128 的 torch / torchvision；spconv 没有 cu128 预编译包，但 `spconv-cu126` 自带 CUDA 运行时，
  与 torch cu128 同装可用（已在 RTX 5080 Laptop 上跑通全部测试与训练）。
- 换其他 CUDA 版本：把 requirements 里的 torch 索引、torch、torchvision 三处一起改；spconv 的 PyPI 预编译包目前只到 `spconv-cu126`，
  只有当它在你的显卡上跑不了时才需要 `make spconv-src` 从源码编译（需要 CUDA toolkit + C++ 编译器）。

## 运行

```bash
make test               # pytest tests：单模块形状 + 整体前向/反向 + 展开
make train              # python train.py：合成数据上的最小训练循环
python train.py model.K=256 train.steps=50 train.batch_size=2     # OmegaConf dotlist 覆盖任意配置
```

配置文件 `configs/default.yaml`，模式由 dataclass 定义并校验（`models/*.py`、`losses/lewm.py`、`train.py`）。
所有维度（K、D、T、C）、层数、排序方式、读出方式、融合方式、λ 都在配置里。

## 输入约定与对外接口

- 一个样本 = T+1 帧点云 + T 个动作。第 t 帧 `P_t ∈ R^{K×C}`，C ≥ 3，前三维 xyz，其余通道可选；每帧点数固定为 K（数据预处理阶段采样到 K）。
- 帧与帧之间的点没有一一对应，点的存储顺序没有意义（测试里检验了排列不变性）。
- 动作 `a_t` 是 t → t+1 的转移动作，最后一维就是该环境的动作维度；每个环境的动作维度和均值/方差在 `model.action.envs` 登记，
  一个 batch 只含一个环境，环境名作为额外参数 `env` 传入前向和展开（缺省取配置里的第一个环境）。新环境只加配置不改代码。

```python
forward(points [B,T+1,K,C], actions [B,T,A_env], env=None) -> dict
  # 前 T 帧 → latent1 → latent2；latent2 + 第 T 个动作 → pred；第 T+1 帧过空间聚合得 target（就是下一帧的 latent1）
  # {"pred": [B,K,D], "target": [B,K,D], "loss": 标量, "latent1": [B,T,K,D], "latent2": [B,K,D],
  #  "pred_loss": 标量, "reg_loss": 标量}
encode(points [B,T,K,C]) -> (latent1 [B,T,K,D], latent2 [B,K,D])           # 只编码不预测
rollout(points [B,T,K,C], actions [B,H,A_env], env=None) -> [B,H,K,D]      # 每步把预测接到上下文末尾，滑动窗口保持长度 T
```

## 数据流与各张量形状

| 模块 | 文件 | 输入 | 输出 |
|---|---|---|---|
| `Ordering` | `models/ordering.py` | 点坐标 `[B,K,3]` | 排序索引 `[B,K]`（`voxelize`：格点 `[B,K,3]` int32） |
| `SpatialAggregator` | `models/aggregator.py` | 单帧 `[B,K,C]` | tokens `[B,K,D]`、重排后的坐标 positions `[B,K,3]` |
| `TemporalEncoder` | `models/temporal.py` | latent1 `[B,T,K,D]` | latent2 `[B,K,D]` |
| `ActionEncoder` | `models/action.py` | 原始动作 `[B,M,A_env]` + `env` | `[B,M,D]`（预测器用 M=1，即第 T 个动作） |
| `Predictor` | `models/predictor.py` | latent2 `[B,K,D]`、动作向量 `[B,1,D]`、`env` | 下一帧 latent1 的点估计 `[B,K,D]` |
| `LeWMLoss` | `losses/lewm.py` | pred/target `[B,K,D]`、全部 T+1 帧 latent1 `[B,T+1,K,D]` | `loss = pred_loss + λ·reg_loss` |

`SpatialAggregator` 内部的数据流（单帧）：

1. `Ordering.voxelize`：`grid = clamp(floor((xyz - origin) / grid_size), 0, 2**depth - 1)`，原点与深度固定在配置里；
   `Ordering.forward`：沿曲线（默认 Hilbert）编码、稳定排序 → `order [B,K]`。
2. 体素去重：键 `(样本索引, 格点)`，`torch.unique` 得到唯一体素（样本索引在最前 ⇒ 按样本有序）与逆映射；同体素的点取坐标和特征的均值。
3. 唯一体素展平成 PTv3 的点云字典 `{coord, feat, batch, grid_coord, grid_size}` 送入完整的编码-解码 U-Net；
   格点由排序模块算好直接塞进字典，骨干内部不再重算；解码器末级通道数 = D。
4. 输出逐体素特征按逆映射散回 K 个点，再和坐标一起按 `order` 重排 → `tokens [B,K,D]`、`positions [B,K,3]`。

`WorldModel` 对 T（或 T+1）帧逐帧独立调用聚合器（各帧展平到骨干的 batch 维，帧间不共享信息），堆叠成 latent1。

## 目录

```
pyproject.toml / requirements.txt / requirements-cu128.txt / environment.yml / .python-version / Makefile
configs/default.yaml        # 冒烟规模默认配置
models/
  ordering.py               # 4.1 排序子模块（仓库 A 的序列化编码）
  aggregator.py             # 4.1 空间聚合（仓库 A 的 PTv3 v3m2 骨干 + 体素去重/散回）
  temporal.py               # 4.2 时间 Transformer
  action.py                 # 4.3 动作编码（仓库 B 的动作词干）
  predictor.py              # 4.3 预测器（仓库 B 的 ST Transformer 解码器 + 线性回归头）
  world_model.py            # 第 3 节对外接口 + 配置加载
losses/lewm.py              # 4.4 损失（仓库 C）
data/synthetic.py           # 合成数据集
train.py                    # 最小训练循环
tests/test_shapes.py
```

## 每个模块移植自哪里、改了什么

### A：Pointcept — Point Transformer V3 v3m2（Sonata 所用版本）→ `models/ordering.py`、`models/aggregator.py`

来源：`pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py`（SerializedAttention / Block / GridPooling /
GridUnpooling / Embedding / PointTransformerV3）、`pointcept/models/utils/structure.py`（Point）、`pointcept/models/modules.py`
（PointSequential）、`pointcept/models/utils/serialization/`（Z 序 / Hilbert 编码）。

改动栏：

- **关闭 flash 分支**：只保留普通注意力分支（走 `softmax(QKᵀ)V`，不依赖 flash-attn）。
- **随机打乱开关补传到下采样层**：`shuffle_orders` 从顶层传给每一级 `GridPooling`（原版没有传，下采样层固定为 True，评估时输出仍随机）；配置默认 `false`。
- **进网前体素去重、输出散回**：原版假定进网前每个体素只有一个点；本仓库采样到 K 后直接进网，
  进网前按（样本索引, 格点坐标）去重，同体素的点取坐标和特征的均值（本仓库自己的选择），骨干只跑唯一体素，输出按逆映射散回 K 个点。
- **排序的曲线深度和原点固定**：曲线种类（z / z-trans / hilbert / hilbert-trans，默认 hilbert）、深度（默认 16）、体素原点（世界系包围盒最小角）
  都在配置里，不随样本自适应；格点截断到 `[0, 2**depth - 1]`；排序用稳定排序，同体素的点编码相同。
  骨干内部各级序列化同样用这个固定深度（原版按样本自适应，格点接近 2**16 时会触发 depth ≤ 16 的断言）。
- **分块大小随批变化无法避免**：普通注意力分支的分块大小 = `min(批内最少体素数, patch_size)`，
  所以同一样本的输出会轻微依赖同批其他样本（包括同一 batch 里展平进去的其他帧）。
- 去掉与本任务无关的分支：RPE、LayerScale、mask_token、enc_mode（必须走完整编码-解码，编码器最深层是体素级读不出逐点输出）、
  freeze_encoder、traceable 上采样、ocnn；序列化编码只保留 encode。
- xCPE 里的稀疏卷积不带 bias（原版 `bias=True`）：紧随其后的 Linear 的 bias 能完全吸收它，参数化等价；
  这样 spconv 的 CPU 前向（不支持 fused bias、只有 Native 算法）也能跑形状测试；在 CUDA 上仍用 spconv 默认的隐式 GEMM 算法。
- 不依赖 torch-scatter：原版 GridPooling 用 `torch_scatter.segment_csr`，本仓库的体素去重用 `scatter_mean`，
  这里都用 torch 原生算子等价实现（`torch.segment_reduce`、`index_add_`，见 `models/aggregator.py`）。
  原因：2026-09 起 PyG 的预编译轮子托管 `data.pyg.org`（S3 存储桶）被禁用，torch-scatter 无法一条命令安装。

### B：HMA（Heterogeneous Masked Autoregression）→ `models/action.py`、`models/predictor.py`

来源：`hma/model/st_mask_git.py`（ActionStat 归一化、BasicMLP 动作词干、ModulateLayer、按环境名索引的模块字典）、
`hma/model/st_transformer.py`（Mlp / STBlock / STTransformerDecoder，含零 dropout 默认与初始化）、`hma/model/attention.py`（BasicSelfAttention）。

改动栏：

- **输出头换回归**：HMA 原来的头是离散 token 分类或扩散，没有均值路径可取，换成一个 `nn.Linear(D, D)`。这是对它唯一的结构改动。
- **去掉 xformers 改用原生注意力**：`attention.py` 顶层硬依赖 xformers，删掉；用它自带的 BasicSelfAttention，内部换成 `torch.nn.functional.scaled_dot_product_attention`。
- **关闭 muP**：`use_mup=False`（原解码器默认打开）。
- **不提供交叉注意力融合**：原版在 batch > 1 时把动作平铺错了样本。融合方式配置项 `predictor.fusion`：`add`（HMA 默认，原代码里叫 `mlp`：动作向量加到该帧所有空间 token 上）或 `modulate`（ModulateLayer 调制）。
- latent2 视作时间长度 1、空间长度 K 的输入；时间长度为 1 时时间注意力退化为一个残差线性层，保留原样。
- 动作编码器：每个环境一个「归一化 → 两层 MLP」词干，输出维度 D；去掉了 ActionStat 里把数据集 stride 折进最后一维的 `(s d)` 重排。

### C：LeWM（LeWorldModel）→ `losses/lewm.py`

来源：`le-wm/module.py`（SIGReg）、`le-wm/train.py`（`lejepa_forward` 的两项损失）。

```
预测损失 = 预测值与目标逐向量的均方误差          # 目标不截断梯度，梯度双向流回编码器
正则损失 = SIGReg( 全部 T+1 帧的 latent1，重排为 (位置 = 帧×槽位, 样本, D) )
总损失   = 预测损失 + λ · 正则损失                  # λ 默认 0.09（loss.reg_weight）
```

改动栏：

- **正则的位置维扩展为帧数乘 K 并分块计算**：SIGReg 统计量沿样本维计算、沿位置维平均；位置维从帧数变成帧数×K 后中间张量
  `(位置, 样本, 投影, 节点)` 会占几十 GB，因此沿位置分块累加（同一份随机投影复用），并用 `torch.utils.checkpoint` 逐块重算，
  峰值显存只与块大小 `loss.chunk` 有关；与原实现精确等价（测试里与原式逐一比对）。默认超参照用：17 个节点、1024 个投影。
- **只监督最后一步**：LeWM 是教师强制、每个上下文步都监督；这里只监督第 T+1 帧。
- **不加投影头**：不加它编码器后面和预测器后面的 MLP 投影头。
- 前 T−1 个动作不进网络（第 3 节定义如此）。V-JEPA 2 的动作条件预测器只作 token 级按槽位对齐写法的参考，损失不搬。

### 三个仓库都不涉及的部分（取最简单实现，代码注释里已标出）

- `TemporalEncoder`：`torch.nn.TransformerEncoder`（pre-norm、GELU），全注意力；位置/时间编码各自可选可学习或固定正余弦；
  因果掩码按帧划分；读出 `last_frame` 或 K 个可学习查询的一层 `nn.MultiheadAttention` 交叉注意力。
- 合成数据集：xyz 均匀落在包围盒内、其余通道均匀 [0,1]、动作标准正态，帧间点无对应关系。

## 已知限制

- 训练和测试以 CUDA 为准（RTX 5080 Laptop + torch 2.9.0+cu128 + spconv-cu126 2.3.8 上验证：16 个测试全过，`train.py` 200 步跑通）。
  没有 CUDA 时 `tests/` 与 `train.py` 会回退到 CPU（spconv 的 Native 算法在 CPU 上前向、反向都能跑通，但很慢）。
- 普通注意力分支的分块大小取批内最少体素数，同一样本的输出会轻微依赖同批其他样本。
- 不含真实数据加载器、分布式训练、评测脚本与可视化。
