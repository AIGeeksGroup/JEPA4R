#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/runtime/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_PLATFORMS=cpu MUJOCO_GL=egl
trap 'rc=$?; printf "%s\n" "$rc" > "$HERE/wrapper_exit_code"' EXIT
"$ROOT/env_runtime/bin/python" -B "$HERE/run_serial.py"
