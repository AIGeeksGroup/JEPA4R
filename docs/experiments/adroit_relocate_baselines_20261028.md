# 原生 Adroit Relocate：四条 RGB baseline 与最薄接口

结果已加入[仓库README主矩阵](../../README.md#adroit-bc-参考矩阵)，[可移植清单](../results/adroit_relocate_baselines_20261028.json)保留12个seed的原始20轮曲线、实际退出、checkpoint／metrics／validation／policy／CPU重载报告SHA。真实生产代码是 `3f59920a5e4e3bfcc51a33d0b86dcbe4757b3333`；本次发布只整合该已验收薄适配和结果，没有重跑训练或在线SR。

## 结果口径

| 方法 | Seed100 Best／Final | Seed200 Best／Final | Seed300 Best／Final | Best均值±样本SD | Final均值±样本SD |
|---|---:|---:|---:|---:|---:|
| PLDM | 12／0 | 8／0 | 12／4 | 10.67±2.31 | 1.33±2.31 |
| LeJEPA | 16／8 | 16／8 | 16／4 | 16.00±0.00 | 6.67±2.31 |
| VISReg | 48／40 | 44／28 | 28／24 | 40.00±10.58 | 30.67±8.33 |
| V-JEPA 2.1 | 40／16 | 32／20 | 36／20 | 36.00±4.00 | 18.67±2.31 |

均值单位为%，SD为百分点，`ddof=1`。每seed的Best取20轮 `eval/success` 最大值，Final取0-based `eval99`；评测索引为 `5,10,...,95,99`。主表两任务均值使用既有Pen结果与本次Relocate结果的任务均值分别取算术平均，属于描述性汇总，不是pooled SR、置信区间或官方榜成绩。

原BC保持100条官方示范、100epochs、head batch256、MSE＋Adam lr1e-3、每轮25条在线轨迹、`num_cpu=1`、seeds100/200/300。12stage严格串行，完整100条有限loss、20轮评测、真实子进程退出0后才验收。所有12份原生保存策略均通过两次CPU重载、state一致、输出有限和maxabs0检查。

`best_policy.pickle` 按reward及原保存条件产生，不自动等于曲线Best或Final；原生没有 `final_policy`，没有补造。每seed在多轮中复用25个评测初态，`train/*`复用eval轨迹，没有独立500个train-init评测。7,900次BC更新只是原loop预期数量，没有实际optimizer hook计数；没有实测encoder训练前后运行态SHA。

## 数据与原方法

- WM使用已验收的原生Relocate mixed数据：10,000条train、每条200行，共2,000,000个 **pre-action RGB／outgoing action** 行；原action30、observation39、proprio30、qpos/qvel36及随机／branch／失败样本保留。独立150条test未进入预训练或validation。400个train分片及其VDS保持配对；WM相机为 `vil_camera`、256 RGB、MSAA0。不得套用Pen的T+1／末行NaN动作协议。
- BC单独使用官方100条示范的20,000行。保留的官方初态、动作和非视觉字段与已有PCD100一致，RGB由原仿真闭环重渲，`vil_camera`／256／MSAA8；不是WM mixed示范或点云反投影。历史原官方RGB文件缺失，未声称重渲RGB与其逐像素等价。输入pickle SHA为 `187ed87f06dc1cbcc559ba33e7fa16befe4354f8cbfdd3d5b58228b1c2454d04`。
- PLDM原4帧、skip5、B128/BF16、完整原PLDMLoss／两AdamW／调度不变，action30对应150D动作块；原clip随机90/10划分不是episode-disjoint。原seed3072完成10轮、两optimizer各127,260次更新及14,150个validation batch，完整module严格CPU重载、epoch10裸权重与full.model逐tensor一致。梯度为抽查，不宣称全步检查、optimizer/scheduler恢复或不中断RNG等价。
- LeJEPA从原seed0完成10轮、78,120次真实AdamW及forward；原Tiny/14→512 head、四CPU增强view、B256/BF16、0.98 invariance＋0.02 SIGReg、16D训练projector与原AdamW／调度不变。没有validation loop，全部视觉帧保留；1..10原生权重均有限，backbone/head/proj完整严格CPU重载。原权重导出不含optimizer/scheduler，不作weights-only恢复承诺。
- VISReg、V-JEPA 2.1直接复用发布权重，不进行本地预训练。四法保留各自原读出192／512／768／768D，FLARE history3＋30D proprio得到606／1566／2334／2334D输入，输出30D动作；没有PCA、补零或统一宽度。

模型容量、预训练数据、原生分辨率和预算不同，以上SR不建立方法、预训练或容量优势的因果结论；它们也不是RH56、PCD扩展或片段CEM成绩。本轮未补2D LeWM。

## 允许变化的接口

代码仅扩展现有 `scripts/baselines/pen/train_pldm.py`、`train_lejepa.py` 的显式 `--task adroit_relocate`，保留Pen默认值、原参数转发和原生学习调用。两份新YAML `pldm_relocate`／`lejepa_relocate`只选择既有loader及本任务epoch10目录；VISReg／V-JEPA原YAML和loader直接复用。native trainer、模型、loss、增强、采样、优化器、调度和BC runner未改。

`--task`、预期帧数和schema标签是计划元数据，不能代替对真实HDF数据、动作块、边界batch、CUDA更新、checkpoint严格重载的验收。默认只准备计划；显式 `--mode smoke|formal` 才执行，`--dry-run`仍不训练。输出必须使用全新外部目录。

### 外部版本与环境边界

稳定世界模型框架固定 `6f1e499e9cc0c898d326112f485c1062c3d20f24`；EAI固定 `76fe35e87b1937168f1ec4b236e863451883eaf3` 及其需要的子模块／legacy运行环境；V-JEPA源码固定 `204698b45b3712590f06245fbfba32d3be539812`。数据、checkpoint、完整依赖环境、MuJoCo200、许可证和仿真资产均外置，不随此发布复制。

实际WM环境：Python3.10.21、Torch2.8/cu128、NumPy1.26.4、Transformers5.17、stable-pretraining0.1.8、Lightning2.6.6。实际BC环境：Python3.10.21、Torch2.8/cu128、NumPy1.23.4、timm0.6.11、Gym0.23.1、mujoco_py2.0.2.13。BC生产进程在主环境之后追加已有WM site-packages补充Gymnasium/SPT，保留primary NumPy/Torch/timm/Gym/MuJoCo provider；timm0.6.11使用已有纯代码overlay，没有改全局环境或学习loader。真实provider检查和该私有边界的SHA在结果清单中保留；下方示例只是相同追加与转发方式，不另造BC实现。

`LD_LIBRARY_PATH`、MuJoCo路径、许可证及EGL设置必须在解释器启动前配置。不要把WM site-packages放到BC `PYTHONPATH`最前方覆盖NumPy，也不要在Python启动后才修改动态库路径。框架源码须显式加入路径；EAI的 `mujoco_vc/src`、`visual_imitation`、`vc_models/src`及V-JEPA root须解析到上述固定源码。

### 显式任务与数据开关

在仓库根目录设置外部路径，以下先展示计划；运行时先完成本任务smoke／真实更新／重载验收。PLDM完整验收后才运行LeJEPA，两预训练均验收并冻结后才运行BC。

```bash
REPO=$PWD
SWM_ROOT=/path/to/stable-worldmodel
WM_PY=/nix/lewm/your-wm-env/bin/python
TRAIN_RGB=/path/to/relocate/full_rgb_v1/train.h5
OUT_ROOT=/path/to/fresh-relocate-training
export PYTHONPATH="$REPO/lewm_pcd:$SWM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$WM_PY" "$REPO/lewm_pcd/scripts/baselines/pen/train_pldm.py" \
  --framework-root "$SWM_ROOT" --task adroit_relocate \
  --dataset "$TRAIN_RGB" --output "$OUT_ROOT/pldm" \
  --workers 24 --prefetch-factor 2 --mode formal --dry-run

"$WM_PY" "$REPO/lewm_pcd/scripts/baselines/pen/train_lejepa.py" \
  --task adroit_relocate --dataset "$TRAIN_RGB" --output "$OUT_ROOT/lejepa" \
  --workers 24 --prefetch-factor 2 --mode formal --dry-run
```

确认外部版本、数据和GPU smoke后，去掉各自 `--dry-run` 才执行相应正式阶段。24workers／prefetch2用于预训练输入和24路源分片读取；BC已缓存特征的小MLP loader及依赖当前动作的在线未来帧不机械套用24路预取。

### 复用原 BC entry，四方法依次三 seed

下面的checkpoint必须是已完整验收的真实epoch10目录；`STABLEWM_HOME`指发布权重根，而不是新的训练输出根。VISReg权重按作者许可单独获取；V-JEPA文件名保持原发布版本，清单绑定四份实际checkpoint SHA。

```bash
BC_PY=/nix/lewm/your-bc-env/bin/python
EAI_VC_ROOT=/path/to/eai-vc-76fe35
VJEPA2_ROOT=/path/to/vjepa2-204698b
TIMM0611_ROOT=/path/to/existing-timm0611-overlay
export CORTEXBENCH_EAI_VC_ROOT="$EAI_VC_ROOT"
export BC_EXTRA_SITE=/path/to/existing-wm-env/lib/python3.10/site-packages
export BC_ENTRY="$REPO/lewm_pcd/bench/cortexbench/run_bc.py"
export STABLEWM_HOME=/path/to/released-weights-home
export RELOCATE_PLDM_CHECKPOINT="$OUT_ROOT/pldm/checkpoints/pldm_adroit_relocate"
export RELOCATE_LEJEPA_CHECKPOINT="$OUT_ROOT/lejepa/checkpoints/lejepa_adroit_relocate"
export PYTHONPATH="$TIMM0611_ROOT:$REPO/lewm_pcd:$REPO/lewm_pcd/bench/cortexbench:$SWM_ROOT:$EAI_VC_ROOT/cortexbench/mujoco_vc/src:$EAI_VC_ROOT/cortexbench/mujoco_vc/visual_imitation:$EAI_VC_ROOT/vc_models/src:$VJEPA2_ROOT"
BC_DATA_DIR=/path/to/official_bc_rgb100  # contains relocate-v0.pickle
BC_RUN_ROOT=/path/to/fresh-relocate-bc
test ! -e "$BC_RUN_ROOT" || exit 1

for embedding in pldm_relocate lejepa_relocate visreg_vitb vjepa21_vitb; do
  for seed in 100 200 300; do
    "$BC_PY" -c 'import os,runpy,sys; extra=os.environ.get("BC_EXTRA_SITE"); sys.path.extend([extra] if extra else []); entry=os.environ["BC_ENTRY"]; sys.argv=[entry,*sys.argv[1:]]; runpy.run_path(entry,run_name="__main__")' \
      --config-name Adroit_BC_config env=relocate-v0 embedding="$embedding" \
      seed="$seed" data_dir="$BC_DATA_DIR" num_demos=100 epochs=100 \
      eval_num_traj=25 eval_frequency=5 save_frequency=10 num_cpu=1 \
      bc_kwargs.batch_size=256 job_name=policy hydra.job.chdir=true \
      hydra.run.dir="$BC_RUN_ROOT/${embedding}_seed${seed}" \
      wandb.project=local wandb.entity=local || exit $?
  done
done
```

每stage保留命令／环境、日志、状态和实际退出码；完成须核100条有限loss、20次完整eval与实际exit0。loss100先于eval99写出，不能提前判完；每个stage验收后才将后续运行纳入已完成结果。已保存policy需独立CPU双重载并核SHA；原保存条件没有导出时单独披露。原流程、外部运行时和仿真渲染仍可能影响重复运行，本文不承诺单次图像闭环逐位一致。
