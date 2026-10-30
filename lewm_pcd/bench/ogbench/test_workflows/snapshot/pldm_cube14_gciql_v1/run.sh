#!/usr/bin/env bash
set +e
queue="$1"
case "$queue" in smoke|full) ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/pldm_cube14_gciql_v1/run.py --queue "$queue" --execute
code=$?
printf '%s\n' "$code" > "/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/pldm_cube14_gciql_v1/control/$queue/wrapper_exit_code"
exit "$code"
