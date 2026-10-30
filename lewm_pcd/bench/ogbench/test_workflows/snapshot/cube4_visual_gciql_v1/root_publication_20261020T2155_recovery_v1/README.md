# 4Cube cached LeWM GCIQL

服务器：ssh103032Test。最小接口、原GCIQL算法、GPU串行。

| 组别 | 状态 |
|---|---|
| G11 RGB | 1M训练＋完整153叶CPU接受0，11checkpoint账本／200验证齐，final-only本地备份完成 |
| G12 PCD | 新PCD缓存full运行；验收后由已arm恢复等待器生成真实v3配置并串行开训 |

两组freshseed0/B1024/1M，current/next=cache192+proprio19（211维），各自原sampleID goal=cache192+四物体XYZ12（204维），不补零。四cube各距目标<=4cm后ALL奖励0，否则-1，mask1-success。原alpha1/Adam3e-4/512x3LN/.99/.9/tau.005、sampler/RNG、JAX默认精度不变。训练不运行encoder，4线程仅查询同batch RAM。

G11在2026-10-20 21:54完成100万更新并通过CPU接受。闭合包42,372,006B／73members，SHA9176e27a689235030bd2a6f4557589173960b120825bb0bb3d3a1455dba5787c；唯一SCP到非OneDrive scratch、成员SHA/CRC/源前后/本地整包校验均0，Root再次核全SHA。11点仅保存最终policy+sidecar负载，其他10点保留远端并有SHA账本。

PCD原v1在工程5%性能门实际失败1：数值逐位相同、实测耗时下降3.9868%，原accepted=false原样保留。Root用新版本仅放行两处性能准入，原renderer/preprocess/model/数值容差不改；没有重测凑PASS。新control_v2剩余4阶段已22:10运行，profile／2002帧pilot实际0；22:25:50 full907／5500EP，907907有效feature rows，GPU99%、无OOM。16renderer／32有界future／B32／FP32XYZ+uint8RGB GPU预处理保持。

当前20/40/80个真实提交窗约983/980/1002行/秒，条件编码结束23:42–23:44，封存和CPU审计另计。下一次23:32检查。G12新等待器control_v2/after_cache_G12_runtime_v2已arm，只有新缓存完整审计/恢复收据通过才产生functional_v2/configs/G12_cube4_pcd.v3.json；当前没有G12训练或新SR。

实时入口：root_status.json、monitor_handoff_v1.json、monitor_checks/20261020T2155_v1/Root_recovery_and_ETA_v2.json。G12最终备份仅用closed_backup_preparation_v3/HANDOFF_v1.json；旧v2 G12 selector不适用。PCD新cache小metadata包仍待闭合后实施；向量4.23GB仍远端批准Temp，不声称已本地备份全部向量。
