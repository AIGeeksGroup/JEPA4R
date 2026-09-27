# source bench/cortexbench/env.sh  -- CortexBench (eai-vc) runtime for the scripts in this directory.
# Needs the cortexbench conda env (python 3.10 + torch cu128 + eai-vc stack, see README.md): activate it or set PY.
CB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # lewm_pcd/bench/cortexbench
LEWM_ROOT=$(cd "$CB_ROOT/../.." && pwd)                  # lewm_pcd
export CB_ROOT LEWM_ROOT
export PY=${PY:-python}
export STABLEWM_HOME=${STABLEWM_HOME:-$(cd "$LEWM_ROOT/.." && pwd)/lewm_pcd_data}   # demos, checkpoints, runs
export MUJOCO_PY_MUJOCO_PATH=$LEWM_ROOT/third_party/mujoco/mujoco200                  # MuJoCo 2.0 for mujoco-py
export MUJOCO_PY_MJKEY_PATH=$LEWM_ROOT/third_party/mujoco/mjkey.txt
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$LEWM_ROOT:$CB_ROOT${PYTHONPATH:+:$PYTHONPATH}   # lewm_pcd (module.py, encoders/) + the loaders here
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
