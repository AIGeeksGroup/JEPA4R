#!/usr/bin/env bash
set -u
CASE=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube2_official_embeddings_v1
PY=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python
"$PY" -B "$CASE/resume_relay_v2.py"
rc=$?
if [ "$rc" -eq 0 ]; then
    "$PY" -B "$CASE/publish_ready_v2.py"
    rc=$?
fi
printf '%s\n' "$rc" > "$CASE/wrapper_v2_exit_code"
exit "$rc"
