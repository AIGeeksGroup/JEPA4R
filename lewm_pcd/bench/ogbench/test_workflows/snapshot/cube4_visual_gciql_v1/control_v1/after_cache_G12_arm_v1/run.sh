#!/usr/bin/env bash
set -eu
RUN=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/control_v1/after_cache_G12_arm_v1
exec 9>"$RUN/serial.lock"
flock -n 9 || exit 73
[ ! -e /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/control_v1/after_cache_G12_runtime_v1 ] && [ ! -e "$RUN/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$RUN/wrapper_exit_code"; exit "$rc"' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/control_v1/after_cache_G12.py --q /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1 --pcd-case /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_official_embeddings_v1/pcd_training_camera_v1
