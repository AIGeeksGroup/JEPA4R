#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
CASE="$ROOT/cube4_visual_gciql_v1"
RUN="$CASE/control_v1/G11"
exec 9>"$RUN/serial.lock"
flock -n 9 || exit 73
[ ! -e "$RUN/runtime" ] && [ ! -e "$RUN/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$RUN/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$ROOT/env_runtime/bin/python" -B "$CASE/control_v1/run_role.py" --directory "$RUN"
