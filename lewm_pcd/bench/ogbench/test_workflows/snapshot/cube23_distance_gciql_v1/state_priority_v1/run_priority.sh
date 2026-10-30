#!/usr/bin/env bash
set -eu
PRIORITY=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube23_distance_gciql_v1/state_priority_v1
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
exec 9>"$PRIORITY/serial.lock"
flock -n 9 || exit 73
[ ! -e "$PRIORITY/runtime" ] && [ ! -e "$PRIORITY/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$PRIORITY/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_THREEFRY_PARTITIONABLE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
case " ${XLA_FLAGS-} " in
  *" --xla_gpu_autotune_level=0 "*) ;;
  *) export XLA_FLAGS="${XLA_FLAGS-} --xla_gpu_autotune_level=0" ;;
esac
"$ROOT/env_runtime/bin/python" -B "$PRIORITY/run_priority.py" --execute
