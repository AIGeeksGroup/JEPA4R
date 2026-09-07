# 当前实现与正式启动状态

当前不是三个正式实验均已就绪。原 PPO 已通过真实环境采样、更新、保存和重载检查；专家采样预检为原生 success 0/30，按用户要求禁止开采；世界模型训练驱动已通过明确标记的 CPU 合成测试夹具的优化、保存、恢复和重载，尚无真实专家点云数据训练或规划评测。正式 GPU 依赖环境尚未安装。

## 实际模型

- PPO/WM 共用 `configs/ppo_pointnet_identity.py`，每帧 xyz+RGB `[B,N,6]` → latent1 `[B,512]`；配置 `[6,64,128,512]`，LN1d、前两层 ReLU、最后 Identity（无激活）、全局最大池化。最后激活的变更是用户最新授权；在外部子类中实现，继承官方 forward。
- 输入四帧，获得 latent1 `[B,4,512]`。前三帧送入独立 TemporalEncoder，K=1、T=3，学习时间位置编码，原生 PyTorch 因果时间注意力取最后 token。
- TemporalEncoder token 宽度始终 D=512。独立配置默认为2层、8头、FFN4D=2048、dropout0、GELU、pre-norm；这些是本实现的显式初始选择，不是 LeWM 预测器的代码或超参。
- latent2 `[B,512]` 拼第3个7维单步动作，经 critic LinearMLP `[519,256,256,512]`，得到下一帧 latent1 预测。
- 原 V critic 输入为512视觉+30本体状态，即542；按任务要求改为512+7动作，即519。隐藏层和初始化保持原样，输入和输出边界维度必须适配。
- 目标为同一 PointNet 编码的第4帧，不截断梯度。损失为 MSE +0.09×官方SIGReg；SIGReg输入 `[4,B,512]`，逐帧沿batch计算。
- 用户已纠正时间尺度：采样、encoder 输入、预测和 PPO 决策均为1个控制步。WM 使用连续观测0、1、2和动作2，预测观测3；原官方数据 frameskip5 未作为 encoder 参数继续沿用。

## 需要重点审查的结构限制

1. **末层已按要求改为 Identity。** 当前不再通过末层 ReLU 截断负值，也不通过 tanh 限幅；实际测试可输出 -2/+2。没有增加投影头、改变其余激活、归一化、池化或 SIGReg。表征分布由网络和训练共同决定，是否获得有用表征仍由实验评估。
2. **新增时间聚合与预测器分开。** 已移除原先错误复用的 LeWM Transformer 及512→192→512投影；时间模块只聚合 latent1，不输入动作或 state。完整独立配置随 checkpoint 保存并校验。
3. **预测头256维瓶颈。** 原 critic 隐藏宽度保留256，最后由256投影至512；同样是用户指定架构的容量约束。

## 训练配方与接口

生产配置保留官方LeWM的batch128、100epochs、AdamW lr5e-5/wd1e-3、bf16、梯度裁剪1、workers6、train split0.9、seed3072、T3；按最新要求覆盖frameskip为1。官方 normalizer、窗口划分、Lightning/spt管理、模型导出均复用。调度器参数推导复用匹配的stable_worldmodel脚本；配置保留interval=epoch，而固定SPT版本实测按优化更新推进调度器，未额外改写其行为。

forward返回训练输出与损失；encode返回逐帧emb/latent1及时间聚合latent2；rollout继承固定公开LeWM JEPA语义，动作包含上下文动作，而非新版本stable_worldmodel的strictly-future语义。所有官方源码保持未修改。

模型代码和训练驱动能够运行，不构成真实数据表征质量、收敛或正式规划成功的证据。

Identity 修改不会改变 PPO 的公式、actor/critic 隐藏结构或优化入口。旧 ReLU、tanh 和当前 Identity 虽有相同权重形状，复现时仍必须使用各自保存的配置；历史配置文件保留，当前入口和模型导出统一记录 Identity。

第三阶段加载训练好的 encoder 截至 latent2，完全冻结后按原 PPO 拼接当前 state[30]，两支头保持542维输入。见 STAGE3.md。PPO 只优化策略/价值头；连续三帧点云和当前 state 随观测存入 replay，state 不进入 encoder。
