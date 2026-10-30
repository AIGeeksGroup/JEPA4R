# VISReg / V-JEPA Cube GCIQL

服务器：`ssh -p 31435 root@connect.westd.seetacloud.com`。

统一接口支持8组：VISReg ViT-B/16最后层CLS768 × Cube1..4；V-JEPA2.1 ViT-L最后层patch均值1024 × Cube1..4。作者Drive截图已纠正原URL字符，VISReg恢复启用。所有新产物按family和Cube独立目录存放。

流水线：先独立随机初始化诊断 → 所有官方权重smoke → 各组串行执行完整embedding、GCIQL 1M、完整CPU验收、最终checkpoint Native250和独立结果验收。任何实际阶段失败停止，不自动换batch、精度、奖励或重启。随机缓存/配置/checkpoint/日志有random_init_v1_前缀，不得进入正式训练。

输入：current/next = 原生D维视觉特征 + proprio19；actor/value各自目标 = 对应rawID的D维视觉特征 + 每个cube XYZ。D取768或1024，无补零/PCA，当前object GT排除。奖励沿用所有cube各自距目标≤4厘米；训练/采样/GCIQL数学和原生评测主体复用已有代码。

每组fresh seed0、B1024、1M更新，Adam3e-4、alpha1、512×3+LayerNorm、discount.99、expectile.9、tau.005；4个同批RAM特征查询线程、async100，编码时uint8传入GPU。每5K保留原验证，step1及每100K共11份checkpoint，仅最终1M做Native250（各任务50集，Cube1/2/3/4上限200/500/1000/1000步）。不在GCIQL训练中运行encoder。

相机：Cube1原front_pixels；Cube2/3 front上抬5°；Cube4上抬5°并后移10cm。RGB224 AA4整幅视野，训练缓存与在线SR共用同一encoder接口和预处理。官方NPZ每EP保留1001原始观测和真terminal，Cube1/2各1000train+100val、Cube3 3000+300、Cube4 5000+500。

存储：新runs和Native日志放共享长期盘；V-JEPA Cube3完整缓存放根盘新目录，其他缓存放Temp；通过新目录映射保持代码路径不变。旧实验没有移动、删除或覆盖。完整缓存不自动回传本地；阶段参数/退出/QC报告及必要最终模型需按实际完成后备份。

当前验收范围：CPU数据契约、8组维度/奖励/索引/RNG、随机隔离、调度屏障测试已通过。模型GPU诊断、正式训练及SR结果以control中的实际阶段退出和独立报告为准，代码或PID存在不代表通过。
