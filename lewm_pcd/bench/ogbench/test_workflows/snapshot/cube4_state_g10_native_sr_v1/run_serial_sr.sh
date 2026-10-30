#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
SR="$ROOT/cube4_state_g10_native_sr_v1"
exec 9>"$SR/serial.lock"
flock -n 9 || exit 73
[ ! -e "$SR/runtime" ] && [ ! -e "$SR/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$SR/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
"$ROOT/env_runtime/bin/python" -B "$SR/run_serial_sr.py" --plan "$SR/launch_plan.json" --execute
