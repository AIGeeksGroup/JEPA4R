# 4Cube cached LeWM GCIQL

服务器：ssh103032Test。两组使用已迁入并核SHA的4Cube LeWM epoch10权重。

| 编号 | 输入模态 | 当前状态 |
|---|---|---|
| G11 | RGB LeWM192 | B1024门＋独立CPU重载0，seed0正式1M训练中 |
| G12 | PCD LeWM192 | PCD缓存relay已arm，先等G11完整验收；缓存真接受后接原四阶段 |

训练输入current/next=cache192+proprio19（211维）；value/actor各自同采样IDgoal=cache192+四物体XYZ12（204维），不补零。训练只读取缓存，不运行encoder。

沿用1Cube alpha1/Adam3e-4/512×3/LN/discount.99/expectile.9/tau.005、原sampler与JAX默认精度。每组freshseed0、B1024、1M；step1及每100K保存共11份，5K原验证一次。距离奖励要求四物体均在各自目标4cm内。

GPU严格串行。4个worker是同一batch的RAM查询线程；PCD离线编码使用16renderer/32有界future/B32。

当前阶段和下一检查见root_status.json、monitor_handoff_v1.json。21:19:49已观察G11 562100步，近期约211update/s，条件预计2026-10-20 21:54训练结束，CPU接受另计；已设21:44检查。PCD编码与G12尚未完成，当前没有新SR。

权重、原始数据、既有RGB缓存和旧实验保持。迁移stat变化用独立字节证明与sidecar绑定，原每NPY SHA/finite/ID守卫保留。全部新配置、sourcefreeze、实际argv、日志和退出码保存在本目录对应远端命名空间。

21:19:16 G12 after-cache唯一等待器已arm，真实stage=waiting_PCD_actual_closure/NN_importedFalse。它等现PCD八stage、完整5500EP独审及真实退出0后才生成实际G12配置，并调用原四阶段；当前G12配置和训练未产生。
