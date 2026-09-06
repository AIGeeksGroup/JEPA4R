# 4R-JEPA（基于 PPRL 源仓库）

本分支在 [PointPatchRL (PPRL)](https://github.com/balazsgyenes/pprl) 官方仓库（commit 80ebe4f）之上**只新增文件、不改原文件**，实现：

1. **潜空间世界模型（LeWM 方式训练）**：latent1 = PPRL 的 PointGPT encoder 输出（768 维）；
   时间聚合 Transformer（自研，`pprl/models/modules/temporal_transformer.py`）把连续 H=3 帧的 latent1 聚合成 latent2；
   预测器复用 PPRL 价值头的 MLP 结构（`pprl/models/modules/latent_predictor.py`），输入 latent2 ⊕ 动作，输出下一帧 latent1。
   损失 = 预测 MSE（目标不 detach）+ 0.09 × SIGReg（`pprl/models/modules/lewm_sigreg.py`，原文复制自 [LeWM 官方仓库](https://github.com/lucas-maes/le-wm)）。
   训练超参照 LeWM 官方 `config/train/lewm.yaml`（AdamW 5e-5 / wd 1e-3、batch 128、bf16、梯度裁剪 1.0、history 3、num_preds 1、10 epoch）。
   - 数据采集：`scripts/collect_wm_data.py` + `conf/collect_wm_data.yaml`（LeWM 附录 E 的 Cube 协议：10,000 局 × 200 步，带噪声的脚本启发式策略，ManiSkill2 LiftCube-v0）。
   - 训练：`scripts/train_latent_wm.py` + `conf/train_latent_wm.yaml`；数据集 `pprl/data/wm_dataset.py`；模型 `pprl/models/latent_world_model.py`。
   - `data.frameskip` 默认 1（连续帧）。注意：LeWM 官方 frameskip=5 是给其预测器取帧用的，本设计的时间聚合在表征侧，应吃连续帧。
2. **PPO（parllel 实现 + 修正）**：`scripts/train_ppo.py` + `conf/train_ppo.yaml`、`conf/algo/ppo.yaml`、`conf/algo/aux_ppo.yaml`、`conf/model/pointgpt_ppo.yaml`；
   修正文件 `pprl/algos/ppo_fixed.py`（提前停止返回值）、`pprl/models/pg/gaussian_fixed.py`（似然比按维度求和后再 exp）、`pprl/algos/truncation_bootstrap.py`（时限截断 bootstrap）、
   `pprl/models/pg/pointcloud_pg_model.py`（encoder + PPRL PiMlpHead + 价值 MLP）。超参取 ManiSkill2-Learn 官方点云 PPO 配置。
   续训：`scripts/train_ppo_resume.py`（加载放在所有组件构建之后，避免 parllel 构造阶段污染 BatchNorm 统计量）。
3. **冻结世界模型作 encoder 的 PPO**：`scripts/train_ppo_frozen_wm.py` + `conf/train_ppo_frozen_wm.yaml`；
   `pprl/envs/wrappers/history_obs_wrapper.py`（环境输出 H 帧历史，帧间隔从世界模型检查点读取）、
   `pprl/models/pg/frozen_wm_encoder.py`（H 帧 → 冻结 encoder → 冻结时间 Transformer → latent2）、`pprl/models/pg/frozen_encoder.py`（只用 latent1 的备选）。
   除 encoder 外，PPO 其余组件与第 2 项完全相同（重建辅助损失因 encoder 冻结而去掉）。
4. 其他：`scripts/plan_latent_wm.py`（按 LeWM 官方 eval 的 CEM 规划，未使用）、`conf/env/reach.yaml`（sofa_env Reach 配置）、`scripts/train_sac_resume.py`。

运行顺序：采集数据 → `train_latent_wm.py` → `scripts/train_ppo_frozen_wm.sh`（`scripts/after_wm_launch_frozen_ppo.sh` 为等待世界模型结束后自动冒烟并启动 PPO 的链式脚本）。
环境：ManiSkill2 0.5.3 / SAPIEN 2.2.2 / torch 2.8 cu128 / parllel（PPRL 子模块）。
