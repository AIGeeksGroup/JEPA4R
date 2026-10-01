# OGBench-IL

本页记录 JEPA4R 上已完成的 OGBench IL（imitation learning）实验和下游 CEM 诊断。结果按协议分组；double-cube 结果不能写入 single-cube 官方 benchmark 表。

## 核心结论

- **Pretrained WM + BC：50/50 = 100% SR。** 在 double-cube Task-1、legacy six-camera PCD、240 步 native full-episode 协议下，冻结 epoch-10 WM 后训练 BC head，50 条轨迹全部成功。
- **Random WM 对照：0/50 = 0% SR。** 数据、BC head、训练配置、评测 seeds 和环境完全相同；唯一变量是 WM 随机初始化（seed 3072）。该结果支持 pretrained representation 对控制有效，但目前只是单 seed 消融。
- **Bounded strong CEM：0/50 = 0% SR。** 使用合法 action box、horizon 5、300 samples、30 iterations 后，50 条轨迹仍全部截断；问题不能简单归因于 action 越界或搜索预算不足。
- **结果边界：以上都是 double-cube 诊断。** single-cube、front_pixels、200 步的正式 OGBench 线路尚未完成 Expert Gate、PCD Preview Gate、固定任务数据采集和正式 SR。

## 1. 实验边界

| 项目 | 当前协议 |
|---|---|
| 运行环境 | Python 3.10.21，PyTorch 2.8.0+cu128，CUDA 12.8，NVIDIA RTX 5090 |
| 历史执行主机 | 927-5090 |
| world-model checkout | /root/autodl-tmp/workspace/users/lewm/JEPA4R-ogbench-native-sr |
| world-model commit | df0496d86aeb619bcd700a29526e141a8e4b5ab6 |
| 任务 | double-cube Task-1，single_pnp，env_type=double，reward_task_id=1 |
| 观测 | legacy six-camera fused PCD，XYZ+RGB，1200 points |
| PCD 预处理 | center [0.425, 0, 0.2]，XYZ scale 4.0，RGB /255，voxel 0.005，cube fraction 0.25 |
| action | 5-D normalized relative action，环境范围 [-1, 1] |
| episode budget | 240 environment steps |
| 正式单 cube 协议 | cube-single-singletask-task2-v0，front_pixels，200 steps；本页 double 结果不属于该正式线 |

数据和 epoch-10 world-model checkpoint：

    train: /autodl-fs/data/lewm/ogbench-ur5e-task1-10000/full/train.h5
    holdout: /autodl-fs/data/lewm/ogbench-ur5e-task1-10000/full/holdout.h5
    checkpoint: /autodl-fs/data/lewm/ogbench-ur5e-task1-10000/checkpoints/ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_10ep/weights_epoch_10.pt
    checkpoint SHA-256: ce13e4bd3b50ca44aaedd9b109c7f2d5334189ae14a3c82b502bc17108605c17

train.h5 有 10,000 episodes、1,210,000 rows，其中 1,200,000 行 action finite；holdout.h5 有 100 episodes、12,100 rows，其中 12,000 行 action finite。训练文件约为 9,900 expert episodes 加 100 random/failure episodes。

## 2. 结果总表

| 轨道 | 配置 | 评测 | 结果 | run status | 结论范围 |
|---|---|---:|---:|---|---|
| Single-cube formal line | 新 fixed-task 数据、单 front_pixels、200 steps | 官方 full-episode SR | 未运行 | NOT_STARTED | Expert Gate 和 PCD Preview Gate 尚未完成 |
| IL，pretrained WM | frozen epoch-10 WM + 192->256->5 BC head；10 epochs；1.2M train rows；12k holdout rows | seeds 93011..93060，50 episodes | 50/50 = 100% SR；mean 69.86 steps；50 terminated，0 truncated | VERIFIED | double-cube diagnostic |
| IL，random WM control | 同一架构、数据、head、训练超参和评测 seeds；WM 从 config 随机初始化，seed 3072，完全冻结 | 同上 | 0/50 = 0% SR；mean 240 steps；0 terminated，50 truncated | VERIFIED | 单 seed representation ablation |
| Reduced CEM | horizon 2；8 samples；2 iterations；top-k 2；action block 5；500-step wrapper | 50 native episodes | 0/50；全部 truncated | VERIFIED | 低配工程诊断，不是 official single-cube CEM |
| Strong CEM，unbounded | horizon 5；300 samples；30 iterations；top-k 30；240 steps | 5 episodes；seeds 93011..93015 | 0/5；saturation 1124/1200 | FAILED | action 越界，不能作为干净 planner 结果 |
| Strong CEM，bounded gate | 同 300x30；native action box 约束 | 5 episodes；seeds 93011..93015 | 0/5；saturation 0/1200；全部 truncated | VERIFIED run，FAILED task | bounded planner gate |
| Strong CEM，bounded 50 episodes | 与 gate 相同；没有增加单 episode 搜索预算 | 50 episodes；seeds 93011..93060；240 steps | 0/50；全部 truncated；saturation 0/12000；exit code 0 | VERIFIED run，FAILED task | bounded planner diagnostic failure |
| PCD RL / GCIQL | 当前 GCIQL 为 RGB/DINO 轨道，未完成 PCD adapter | 未运行 | 无 RL SR | BLOCKED | 缺 goal-PCD、PCD actor/critic 和 native evaluator |

## 3. IL 详细结果

### 3.1 Pretrained WM

- 冻结 epoch-10 DexArtPointNet WM encoder/projector。
- BC head：Linear(192,256) -> GELU -> Linear(256,5)。
- optimizer：AdamW，learning rate 1e-3，weight decay 1e-4。
- loss：SmoothL1；batch size 512；10 epochs；训练 seed 3072。
- 最佳 holdout loss：0.019787790552775065，epoch 9。
- 评测 seeds：93011..93060。
- 每条轨迹均为 PCD (1200,3)、action (5,)，finite。
- head SHA-256：ffd80f6b55e821339beacdd4d08ed0ed65000c6103532216800ac570fb02ef69。

有效产物：

    training:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_full_bc_20261001_v5/training_summary.json
    head:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_full_bc_20261001_v5/selected_head.pt
    evaluation:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_full_bc_20261001_v6/eval/results.json
    status:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_full_bc_20261001_v6/status.json

注意：v5/status.json 后来记录了一次缺少 MUJOCO_GL=egl 的旧评测失败；训练本身由 training_summary.json 和 checkpoint 验证，成功评测使用独立的 v6。

### 3.2 Random WM control

随机对照不是“加载 pretrained 后重置几个 Linear 层”，而是读取相同 config.json、不加载 checkpoint 权重、固定 WM seed 3072 后完整实例化同一架构，再冻结 encoder/projector。

- 最佳 holdout loss：0.048003953297932944。
- 评测：0/50，全部 240 步截断。
- action saturation：9296 steps；这不是程序崩溃，action 和 PCD 均 finite。
- random WM state SHA-256：4872cfe8aea77c22a657a93d8c68ba03e41cd664bf9b555eabe6799e7acc3ba5。
- BC head SHA-256：f2936f8733dbf32d2784a439e9ed4aff45b2545b5611e4a05ace892e3a666674。

产物：

    run:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_random_wm_20261001/
    random WM state:
    /root/autodl-tmp/lewm_il_features_random_20261001/random_wm_state.pt
    comparison:
    /autodl-fs/data/lewm/ogbench-double-task1-epoch10-sr-20260930/il_random_wm_20261001/control_comparison.json

这是 fixed-seed ablation，不足以报告 random initialization 的 mean/std；至少需要多个独立 WM seeds 才能做随机初始化统计。

## 4. CEM 详细结果

### 4.1 Bounded strong CEM

为排除非法 action 对 planner 的污染，bounded 版本在标准化 action 空间中把 candidate 约束到 native environment action box，并对越界 candidate 加 penalty。配置为：

- horizon 5
- samples 300
- CEM iterations 30
- top-k 30
- action block 5
- receding horizon 1
- native goal PCD
- legacy six-camera PCD

50 条完整 episode 的结果：

- 0/50 success
- 50 条均 truncation at 240 steps
- action 全部 finite，shape (5,)
- action saturation 0/12000
- exit code 0
- elapsed time 1490.40 s

产物：

    /root/autodl-tmp/cem_bounded_strong_50ep_20261001/status.json
    /root/autodl-tmp/cem_bounded_strong_50ep_20261001/results.json
    /root/autodl-tmp/cem_bounded_strong_50ep_20261001/protocol.json

结论：相同单 episode 搜索预算下的 50-episode 诊断仍未解决 double Task-1。下一步应先检查 latent prediction、goal cost alignment、action-block/history 语义，不应只继续增加 samples 或 iterations。

## 5. 明确未完成项

- 未完成 single-cube Expert Gate。
- 未完成 single-camera front_pixels PCD preview 和 cube quota review。
- 未采集正式 single-cube fixed-task 10k episodes。
- 未训练正式 single-cube checkpoint。
- 未得到 official single-cube full-episode SR。
- 未完成 PCD-GCIQL/RL adapter。

## 6. 复现实验原则

所有后续对照必须继承以下固定变量：数据文件、episode-disjoint split、PCD 处理、任务/相机、BC head、训练 epoch/batch/loss/optimizer、评测 seeds、episode budget 和 EGL runtime。每次只改变一个明确变量，并保存 status、日志、配置、退出码、checkpoint/hash 和逐 episode 记录。

本页报告的是 OGBench-IL 当前 double-cube diagnostic 结果，不是 single-cube official benchmark 表。
