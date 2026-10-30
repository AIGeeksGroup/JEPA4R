#!/usr/bin/env bash
set +e
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/runner.py --execute
code=$?
printf "%s\n" "$code" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/fp16_bulk_w8_v1/wrapper_exit_code
exit "$code"
