# Dependencies

World model与下游MPC环境保持分离。本次验收环境：Python3.10、PyTorch2.8.0+cu128、stable-pretraining0.1.8、ogbench1.2.0、MuJoCo3.12.0、h5py3.16.0、hdf5plugin7.1.0。环境安装与CUDA真实算子仍须各主机独立验证。

现有服务器 stable-worldmodel源码checkout HEAD为 `5841424ab0fd9623f6bf72f57f77a046a15af8aa`。GCRL、Predictor、GCBC核心文件另与上游 `21446f1ede6d5284e981bd7b47f432b994e6d812`实际逐字节对齐。旧 `lewm_pcd/README.md`的6f1e499说明属于历史环境，不应当当前锁。

stable-worldmodel不复制进本仓库；原项目LICENSE和补丁记录保留：

```bash
git clone https://github.com/galilai-group/stable-worldmodel lewm_pcd/third_party/stable-worldmodel
git -C lewm_pcd/third_party/stable-worldmodel checkout 5841424ab0fd9623f6bf72f57f77a046a15af8aa
pip install -e 'lewm_pcd/third_party/stable-worldmodel[train,env]'
export PYTHONPATH="$PWD/lewm_pcd:$PWD/lewm_pcd/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"
```

原配置中引用 `jepa`、`encoders`等平铺模块，所以需保留 `lewm_pcd` 在PYTHONPATH。checkpoint与config.yaml由使用者提供，不上传模型或公开数据。

官方state GCBC是另外一条 JAX基线，源码固定 OGBench `572788d97c2a6225b8edb7c420e304c04d1b3459`，JAX/Jaxlib0.4.35、Flax0.10.2、Optax0.2.4。它不经过本分支冻结LeWM factory；CUDA插件与logger兼容应在独立环境处理。
