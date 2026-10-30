#!/bin/bash
set +e
export LD_LIBRARY_PATH=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/runtime/lib:/usr/lib/x86_64-linux-gnu
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/bootstrap.py --release /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/ROOT_RELEASE_v2.json --runtime-directory /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/bootstrap_runtime_v2
rc=$?
printf "%s\n" "$rc" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/control/dispatch_v2/wrapper_exit_code
exit "$rc"
