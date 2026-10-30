#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
PY="$ROOT/env_runtime/bin/python"
CONTROL="$ROOT/cube2_pldm_gciql_v1/control_v1"
exec "$PY" -B "$CONTROL/run_pipeline.py" --execute
