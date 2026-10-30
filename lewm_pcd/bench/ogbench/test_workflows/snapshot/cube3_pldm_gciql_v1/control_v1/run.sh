#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
CONTROL="$ROOT/cube3_pldm_gciql_v1/control_v1"
PY="$ROOT/env_runtime/bin/python"
if [ "$#" -eq 0 ]; then
  exec "$PY" -B "$CONTROL/serial.py"
fi
[ "$#" -eq 1 ] && [ "$1" = --execute ] || exit 2
exec 9>"$CONTROL/serial.lock"
flock -n 9 || exit 73
[ ! -e "$CONTROL/runtime_v1" ] && [ ! -e "$CONTROL/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$CONTROL/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_THREEFRY_PARTITIONABLE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
"$PY" -B "$CONTROL/serial.py" --execute
