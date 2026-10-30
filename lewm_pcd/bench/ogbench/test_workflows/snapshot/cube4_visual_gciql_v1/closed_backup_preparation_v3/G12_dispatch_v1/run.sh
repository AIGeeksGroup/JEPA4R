#!/usr/bin/env bash
set -eu
RUN=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/closed_backup_preparation_v3/G12_dispatch_v1
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$RUN/outer_exit_code"; exit "$rc"' EXIT
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_visual_gciql_v1/closed_backup_preparation_v3/closed_G12.py --group G12 --execute
