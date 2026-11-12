# source /root/autodl-tmp/lewm234d/lewm_pcd/env.sh   —— 跑本项目任何脚本前先 source
export LEWM_PCD_ROOT=/root/autodl-tmp/lewm234d/lewm_pcd
# LeWM / stable-worldmodel 的缓存根：datasets/  checkpoints/  logs/ 都在这里（数据盘）
export STABLEWM_HOME=/root/autodl-tmp/lewm234d/lewm_pcd_data
# 可选：train.py 读数据的根，默认就用 STABLEWM_HOME；有本地 SSD 时再覆盖
# export LOCAL_DATASET_DIR=/path/to/fast/disk
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=$STABLEWM_HOME/logs/train
# stable-pretraining 的 Lightning 运行目录（last.ckpt 等，默认写 ~/.cache 即系统盘）
export SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache
# export WANDB_MODE=offline
export PYTHONPATH=$LEWM_PCD_ROOT:$LEWM_PCD_ROOT/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}

# 专用 venv 建好前的临时方案：ms2 的 venv 有全部 swm 依赖，但 torch 是 CPU 版，只能采集/调试，不能训练
# export PATH=/root/autodl-tmp/ms2_lewm_model_only_20260906/.venv-worldmodel/bin:$PATH
# export PYTHONPATH=$LEWM_PCD_ROOT/TMP/pydeps:$PYTHONPATH   # pymunk 等 pusht 依赖的垫片
