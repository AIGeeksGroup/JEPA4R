# 官方2Cube：2D／3D LeWM embedding

服务器：`ssh 101732Test`。

任务根：`/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube2_official_embeddings_v1`。

| 项目 | 内容 |
|---|---|
| 官方数据 | `cube-double-play-v0`，1000 train／100 val；每条1001原始状态，保留真实terminal |
| 模型 | `checkpoints/cube2/2d_epoch10`、`checkpoints/cube2/3d_epoch10`；均为2Cube LeWM第10轮 |
| 结果 | 每模态1,101,100×192 FP32；`2d/cache/` 与 `3d/cache/` 分开 |
| 启动状态 | 2D pilot实际退出0，2D full已真实提交episode；3D自动排队，尚未声称完成或通过GPU门 |
| 加速 | 2D B32／2 renderer／2个有界episode预取；3D B32／16 renderer／32个有界预取；uint8颜色传GPU，保原预处理 |
| 调度 | 唯一持久 `run_serial_v2.sh`；当前实际状态和退出在 `runtime_v2/`，整个wrapper退出在 `wrapper_v2_exit_code` |

`2d/cache/preprocess_selection.json` 已真实验证同32张图像：GPU和CPU预处理后的pixels、latents最大差均0。传输＋预处理＋编码片段中位CPU34.637ms／GPU10.302ms；该短片段不代表全流程速度提升3.36倍。

相机保持原front上抬5度；2Cube点云用自身训练的crop `[[.1,-.5,-.02],[.75,.5,.6]]`，不使用3Cube扩大crop。保1200点、原固定XYZ仿射和RGB/255；2D无空间crop。官方NPZ为state数据，新RGB/PCD复用既有静态qpos/qvel渲染流程，不声称动作或接触动力学重放。

原3Cube代码、数据、模型和cache保持原样；薄适配只物化到任务下 `workspace/`，原shared source/external/reference/env/checkpoints使用只读引用。`materialize_delta_v2.json`（若存在）或 `adapter_source_freeze_v2.json` 与实际源文件记录确切差异／SHA。

初次辅助材料化因缺少一条只读test来源引用退出1，尚未执行encoder；保留在旧 `runtime/`。`materialize_v3.py` 只补该引用并核对11个既有生成文件及6条共享链接，原v2/source未改，使用新 `runtime_v2/` 继续。旧Triple测试只作来源pin引用，不作为Cube2测试运行。

两个完整producer和新cache CPU审计都实际0、进程组退出、最终wrapper退出0之后，才以 `BOTH_CACHES_READY.json` 和两模态 `cache/READY.json` 为就绪标记。当前不宣称完整缓存已完成或已本地备份。

启动参数：`launch_plan_resume_v2.json`；真实目标文件名为 `control_v2/launch_plan.json`。本地独立数据核实见 `DATA_ACCEPTANCE_READY.json`；实际进度见 `root_status.json` 和 `stable_progress_*.json`。原始1000步数据未裁为自采500或Triple600视图；本任务无GCIQL、BC、RL训练或SR。


## 11:20检查后的实际接力

2D完整1100条／1101100行已通过原独立审计，full/audit阶段实际退出0且进程组清空；闭合小证据544890B已本地复制并核SHA。3D原reference/fresh均0、数值bitexact，原GPU预处理耗时只下降3.985%，未达工程5%门；旧报告accepted=false和runtime_v2/outer=1保留。Root明确记录性能准入例外后保持同一GPU路线，不改数值容差、不复测旧门。新增薄policy仅改变两条工程利润准入guard，原模型、数据、预处理和forward未改。

新唯一run_serial_v3.sh以runtime_v3/control_v3接剩余3D：CPU准备profile→pilot（已2002行/实际0）→full→独立audit，末尾publish_ready_v3.py拼接旧已接受2D和新3D真实退出链。旧runtime_v2退出1不会被改成0。新精确参数在launch_plan_release_v3.json，工程决定在performance_release_v1/release_decision.json。最终只认wrapper_v3_exit_code和对应两READY/两audit；尚未宣称3D完整完成。


## 最终完成

两组均已完整1100EP／1101100×192 FP32，各NPY合计845785600B（含1100个NPY header）。2D实际接受报告为2d/independent_audit/report.json；3D为3d/independent_audit_v2/report.json，两组全新latent SHA／shape／dtype／finite及canonical raw ID／真实terminal-next均接受。原3D audit在开始向量扫描前因symlink配置membership失败；audit_final_v2仅对唯一Double配置pin作resolve＋SHA等价绑定，原报告和旧源保持，未重编码或重跑GPU门。

最终实际聚合入口accept_final_v4.py/run_accept_v4.sh，runtime_v4/outer_exit_code和wrapper_v4_exit_code均真实0，两cache/READY.json和BOTH_CACHES_READY.json齐；旧v2/v3失败均保留为1，不伪造全旧阶段通过。2D及3D生产实际出口分别见runtime_v2/2d_full_exit.json、runtime_v3/3d_released_full_exit.json。

源／配置／参数／日志／退出／验收证据已真实本地备份：旧2D metadata包544890B、SHA771f351d6970474d7e26b5a31bf4b3a59c1b6217aebec1fbbea9f429fb00f209；新3D＋final包2126431B／1234members、SHA3c5b41122eb746f226bd7b895e653974dba1cbfe7f13eb2cdb22118bd9b4f273。原始数据、权重和约1.69GB完整embedding向量仍保留在服务器Temp；未把向量伪称已本地备份。Root最终独立复核见ROOT_FINAL_ACCEPTANCE_READBACK_v2.json与ROOT_FINAL_BACKUP_READBACK.json。没有下游GCIQL／BC／RL训练或SR。
