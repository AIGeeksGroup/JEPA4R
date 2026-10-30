#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
CASE="$ROOT/cube3_pldm_gciql_v1/control_recovery_v3"
PY="$ROOT/env_runtime/bin/python"
if [ "$#" -eq 0 ]; then exec "$PY" -B "$CASE/serial.py"; fi
[ "$#" -eq 1 ] && [ "$1" = --execute ] || exit 2
exec 9>"$CASE/serial.lock"
flock -n 9 || exit 73
[ ! -e "$CASE/runtime_v1" ] && [ ! -e "$CASE/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$CASE/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$PY" -B "$CASE/serial.py" --execute
