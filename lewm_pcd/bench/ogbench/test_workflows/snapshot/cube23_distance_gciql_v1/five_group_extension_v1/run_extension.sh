#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
CASE="$ROOT/cube23_distance_gciql_v1/five_group_extension_v1"
exec 9>"$CASE/serial.lock"
flock -n 9 || exit 73
[ ! -e "$CASE/runtime" ] && [ ! -e "$CASE/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$CASE/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$ROOT/env_runtime/bin/python" -B "$CASE/run_extension.py" --execute
