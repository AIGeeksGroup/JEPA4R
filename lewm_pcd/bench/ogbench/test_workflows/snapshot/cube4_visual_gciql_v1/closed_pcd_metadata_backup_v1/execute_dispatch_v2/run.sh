#!/usr/bin/env bash
set -eu
RUN=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/closed_pcd_metadata_backup_v1/execute_dispatch_v2
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$RUN/outer_exit_code"; exit "$rc"' EXIT
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/closed_pcd_metadata_backup_v1/metadata_backup_v2.py --execute
