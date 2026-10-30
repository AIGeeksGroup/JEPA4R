# 4Cube cached LeWM GCIQL：训练与必要备份完成

服务器：ssh103032Test。两组均freshseed0／B1024／1M更新，训练直接读取各自LeWM192缓存；current211=latent192+proprio19，goal204=同原sampleID latent192+四物体XYZ12，不补零。原GCIQL alpha1/Adam3e-4/512x3LN/.99/.9/tau.005、采样器与JAX默认精度保持。

| 编号 | 模态 | 训练／CPU验收 | Checkpoint／验证 | 最终模型本地备份 |
|---|---|---|---|---|
| G11 | RGB | 1M／完整153叶通过 | 11份／200次 | 已核验 |
| G12 | PCD | 1M／完整153叶通过 | 11份／200次 | 已核验 |

G11训练于10月20日21:54结束，G12于10月21日01:03结束。各自所有role阶段、外层及进程组退出均正常；G12恢复dispatcher三阶段也全部0。CPU报告均明确未执行整段1M host sampler重放。本次未测SR。

PCD缓存5500EP／5505500行FP32D192全部完成独立SHA／finite／canonicalID及真terminal验收，5500小receipt与闭合元数据已本地核验。完整向量仍远端批准Temp。两组11份完整checkpoint均保留远端，本地只收最终policy+sidecar及11份SHA账本，未重传旧WM/raw/cache向量或全部periodic。

三个包均sourcebeforeafter／成员SHA／gzipCRC／唯一SCP非OneDrive scratch／项目wholeSHA-size以及Root独立核验0：

| 包 | 字节 | 成员 | SHA256 |
|---|---:|---:|---|
| G11 final1M | 42372006 | 73 | 9176e27a689235030bd2a6f4557589173960b120825bb0bb3d3a1455dba5787c |
| G12 final1M | 42346081 | 107 | 4ba923919dc49a5b6aff336b635e05de32bca576a84095b8b5fafd7f89612000 |
| PCD closedmetadata | 15361328 | 5643 | b50d765d5bfa72bbc43cc48e04d1e3ea1dbc3226af1f02b55cd7841164df85cf |

原性能5%门失败、正收益3.9868%工程放行、旧聚合退出1和各辅助预飞失败均保留；没有改原accepted=false或数值容差。

最终证据入口：final_closure_v1/final_complete_receipt_v2.json、final_independent_readback_v1/G12_completion_readback_v1.json、各completed_v1/LOCAL_BACKUP_VERIFIED.json。最终监控移除结果另见final_closure_v1/monitor_removal_actual.json；不再自动展开训练或SR。

监控4cube-g11-g12-cached-gciql已由app实际删除，活动automation.toml也已不存在；最终原始删除回执保存在monitor_removal_actual.json。
