#!/usr/bin/env bash
# Source this file; all source paths are explicit and no existing environment is edited.
export LEWM_CHECKOUT=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ogbench-native-sr-df0496d
export LEWM_PCD_ROOT=$LEWM_CHECKOUT/lewm_pcd
export LEWM_PYTHON=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
export LEWM_JAX_CPU=/root/autodl-tmp/workspace/users/lewm/downstream-smoke-20261005-2133/state/cpu-deps
export LEWM_JAX_GPU=/root/autodl-tmp/workspace/users/lewm/ogbench-official-single-play-20261006/jax_gpu_overlay_v1/packages
export PYTHONPATH=$LEWM_PCD_ROOT:$LEWM_PCD_ROOT/third_party/stable-worldmodel:$LEWM_JAX_GPU:$LEWM_JAX_CPU
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export LD_LIBRARY_PATH=/root/autodl-tmp/conda/envs/jepa4r-smoke/lib:/usr/lib/x86_64-linux-gnu
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1
export PATH=/usr/local/cuda/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
LEWM_NVIDIA_LIBS=/root/autodl-tmp/conda/envs/jepa4r-smoke/lib/python3.10/site-packages/nvidia
LEWM_NVIDIA_LIBRARY_PATH=$(find "$LEWM_NVIDIA_LIBS" -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd: -)
export LD_LIBRARY_PATH=$LEWM_NVIDIA_LIBRARY_PATH:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4