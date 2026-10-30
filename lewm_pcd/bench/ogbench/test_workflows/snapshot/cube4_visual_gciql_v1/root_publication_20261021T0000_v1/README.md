# 4Cube cached LeWM GCIQL

服务器：ssh103032Test。复用原GCIQL算法与四角色RAM读取；GPU串行。

| 组别／资产 | 状态 |
|---|---|
| G11 RGB | 1M训练／完整CPU接受／final-only本地备份全部完成 |
| PCD缓存 | 5500EP／5505500行FP32D192，full及独立audit真实0，闭合metadata本地备份完成 |
| G12 PCD | gate2＋独立CPU重载0，23:42:37 freshseed0正式开训；23:55:15已139000步／27val |

两组均B1024／1M／seed0；current211=cache192+proprio19、goal204=各原sampleID cache192+四物体XYZ12，无补零；四cube各距目标<=4cm后ALL奖励0，否则-1，mask1-success。原alpha1/Adam3e-4/512x3LN/.99/.9/tau.005、sampler、JAX默认精度不变。训练不运行encoder，4线程仅同batch RAM查询。每5K原val，step1+每100K共11份。

G12最新3/5/7/12个5K窗约208/206/205/196Hz，条件训练结束2026-10-21 01:04–01:08；最快近期窗约01:03，最终保存与CPU接受另计。下一单次00:52检查。当前训练／最终验收尚未完成，无新SR。

G11本地包42,372,006B／73members，SHA9176e27a689235030bd2a6f4557589173960b120825bb0bb3d3a1455dba5787c。PCD closedmetadata包15,361,328B／5643members，SHAb50d765d5bfa72bbc43cc48e04d1e3ea1dbc3226af1f02b55cd7841164df85cf。两包均源前后／成员SHA／gzipCRC／唯一scratch网络copy／项目wholeSHA-size及Root复核0，勿重包重传。PCD完整向量约4.23GB仍仅远端批准Temp，未本地全量备份。

旧PCD工程5%门实际失败1保留（数值0差、耗时仅下降3.9868%）；新正收益工程放行只改两准入断言，模型／数值容差／资源不改，无重测。原失败和新成功链由recovery_receipt_v2真实绑定。元数据包旧v1静审1未执行，修后v2仅排除旧未运行阶段路径，全部旧记录保留。

当前入口：root_status.json、monitor_handoff_v1.json。G12实际role=control_v2/G12、config=functional_v2/configs/G12_cube4_pcd.v3.json；新v3备份执行接口仅closed_backup_preparation_v3/HANDOFF_v1.json，不能使用旧v2 G12 selector。
