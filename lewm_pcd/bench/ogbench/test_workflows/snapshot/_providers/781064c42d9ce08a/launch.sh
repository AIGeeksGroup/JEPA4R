#!/usr/bin/env bash
source /root/autodl-tmp/workspace/users/lewm/coordination/cached_lewm_visual_state_gciql_b1024_1m_109_20261013_v1/runtime_autotune0_v1/activate.sh
export PYTHONUNBUFFERED=1
recovery=/root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/native1m_v1/aux/native_only_recovery_v1
/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python -u "$recovery/resume_native.py" > "$recovery/run.log" 2>&1
result=$?
printf '%s\n' "$result" > "$recovery/outer_actual_exit_code"
exit "$result"
