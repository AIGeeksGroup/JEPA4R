#!/usr/bin/env bash
set -u
source /root/autodl-tmp/workspace/users/lewm/coordination/cached_lewm_visual_state_gciql_b1024_1m_109_20261013_v1/runtime_autotune0_v1/activate.sh || exit $?
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
mkdir -p /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/formal_dispatch_v1/M5
/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/core_v1/launch.py --config /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/configs/M5_w4_formal_config.json --output /root/autodl-tmp/workspace/users/lewm/ogbench-official-single-play-20261006/rl3_m5_gciql_seed0_b1024_1m_w4_20261013_v1 --launcher-output /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/formal_launches_v1/M5 --gate-report /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/gpu_gates_final_v1/M5/report.json --gate-sha256 aa935d6da5e4ab3a6a1413449fa0ad134d67a415c41d3b828315bbb6356a235d --execute > /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/formal_dispatch_v1/M5/run.log 2>&1
code=$?
printf "%s\n" "$code" > /root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/formal_dispatch_v1/M5/outer_exit_code
exit "$code"
