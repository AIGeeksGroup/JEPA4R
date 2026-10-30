#!/bin/bash
set -u
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/bootstrap.py --release /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/ROOT_RELEASE_v1.json
rc=$?
printf "%s\n" "$rc" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/dispatch_v1/wrapper_exit_code
exit "$rc"
