#!/bin/bash
set +e
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/run_serial_w8.py --plan-directory /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/formal_w8 --execute
rc=$?
printf "%s\n" "$rc" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/dispatch_w8_v1/wrapper_exit_code
exit "$rc"
