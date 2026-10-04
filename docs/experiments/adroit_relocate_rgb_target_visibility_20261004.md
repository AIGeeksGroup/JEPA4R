# Relocate RGB target 可见性只读审计

日期：2026-10-04。直接读取实际训练HDF5和官方BC pickle中的像素，未运行仿真、模型、训练或评测，未修改远程文件。

发布证据：[原始像素来源、pickle header、源码和颜色诊断，以及全部材料的 SHA/可用范围](../results/adroit_relocate_rgb_target_sources_20261004.json)；[全部 18 张抽查帧拼图](../figures/adroit_relocate_rgb_target_samples_20261004.png)。未单独发布的原始 PNG 和拼图脚本位于本 Git checkout 的父目录下 `artifacts/relocate_rgb_target_visibility_20261004_readonly_v1/`，索引给出各文件 SHA；它们不是公开下载链接。

**已验证：抽查的WM训练/测试RGB和官方BC demo图像中有target。它显示为绿色半透明球形区域，位置在不同episode/demo之间变化，后期可能被手遮挡。不能将原PCD缺target的结论套到RGB。**

## 实际像素来源与范围

| 来源 | 原始数据 | 抽查范围 |
|---|---|---|
| v6 WM训练 | datasets/train/relocate.h5，pixels=(2000000,256,256,3) | episodes 0,1,2,100,5000,9999，共8帧 |
| v6 WM测试 | datasets/test/relocate.h5，pixels=(30000,256,256,3) | episodes 0,1,149，共5帧 |
| 官方BC demo | /autodl-fs/data/lewm/cortexbench-official-data/adroit/relocate-v0.pickle | 前3条demo，共5帧 |

前两个HDF5根目录均为：

`/root/autodl-tmp/workspace/users/lewm/adroit_full/relocate_official_10000_150_20261001_v1/`

WM train/test和官方demo0均抽取t=0、80、199；其余抽取t=0。全部18张样本已目视检查，存在绿色目标区域或被手部分遮挡后的绿色残留。不同初始样本的目标中心确实出现在不同图像位置。

官方pickle约4.21GB，本次没有完整反序列化。通过pickletools顺序读取前三个images数组的字节payload，每个形状(200,256,256,3)、39,321,600bytes；header检查确认images键和形状，首数组u1 dtype，后续复用同一dtype memo。读取到第三个数组即停止。pixel_sources.json记录字节offset、原始像素SHA和各帧来源。

初次记录的短header窗口不足，产生header warnings；后续official_image_headers.json使用更长窗口验证了三个images键及形状。该验证不执行pickle中的代码。

## target的定义与渲染路径

准确XML源文件：

`/root/autodl-tmp/workspace/users/lewm/cortexbench-official-pinned-src/mj_envs-c1557f5572977085f053df63f4e81f4b4e1fb17c/mj_envs/envs/hand_manipulation_suite/assets/DAPG_relocate.xml:67`

```xml
<site name="target" pos="-0.007 0.0 0.2" size="0.07" rgba="0 1 0 0.125" />
```

target为绿色、alpha0.125的site。原生relocate_v0.py:77-79在reset随机设置其位置，:106在set_env_state恢复target_pos。

实际WM capture的smoke.py:150-173和BC online gym_wrapper.py:51-58均直接sim.render再vertical flip，camera=vil_camera；当前链没有隐藏target site的操作。

在线评测：本轮未新运行renderer；其显示依据是现存渲染源码、相同XML及此前3demo×3帧的stored-vs-render alignment报告。不能由此声称所有在线帧的目标都可见。

## 224输入与目标信息契约

将18张真实样本按BC的PIL BICUBIC缩放到224后，仍都有绿色像素区域；绿色代理计数为98～487像素。该计数采用G>R+10、G>B+10、G>40，是颜色诊断，**不等于GT target mask、精确目标面积或全量可见率**。

BC proprio只有原生观测的前30维手与基座关节状态。原生39维完整观测为：

`hand_qpos(30) + palm-object(3) + palm-target(3) + object-target(3)`

BC截掉后三组关系，没有额外target坐标输入；完整39维用于采集state专家。RGB WM配置obs_key=pixels，observation字段虽被载入，并未作为目标坐标送入图像encoder。

因此当前RGB BC需要通过图像编码获得目标信息。**图像里有target已获得抽样证据；encoder是否保留其位置及物体—目标关系，尚未验证。**

## 原PCD缺target的原因

原几何PCD链只保留segmentation object_type=5，即OBJ_GEOM；target为site，OBJ_SITE=6，因而被过滤。后续PCD target-aware数据补入target points，属于另一数据线。本报告只使用v6原始RGB和官方BC demo，不使用后续PCD preview证明RGB可见性。

## 证据文件

- rgb_target_examples.png：4张真实初始帧，第一行WM训练，第二行官方BC。
- all_stored_samples.png：全部18张样本。
- pixel_sources.json：原始路径、采样索引、seed、原像素SHA。
- official_image_headers.json：有限内存的pickle数组来源验证。
- target_sources.json：XML、环境、capture和wrapper源码快照与SHA。
- green_pixel_diagnostics.json：256/224颜色代理统计。
- build_contact_sheets.py：可复现的本地图像拼图与颜色诊断脚本。

结论边界：只确认抽查图像中目标存在；未统计完整2M帧目标可见率、遮挡率，未运行encoder目标敏感性测试，也未因此确定低SR根因。
