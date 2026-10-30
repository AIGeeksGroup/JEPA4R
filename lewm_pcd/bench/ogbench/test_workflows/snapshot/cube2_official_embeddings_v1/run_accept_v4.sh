#!/usr/bin/env bash
set -u
CASE=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube2_official_embeddings_v1
PY=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python
export CUDA_VISIBLE_DEVICES=''
"$PY" -B "$CASE/accept_final_v4.py"
rc=$?
printf '%s\n' "$rc" > "$CASE/wrapper_v4_exit_code"
exit "$rc"
