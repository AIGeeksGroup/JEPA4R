#!/usr/bin/env bash
set -eu
ROOT=/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1
CASE="$ROOT/cube4_official_embeddings_v1"
exec 9>"$CASE/control_v2/serial.lock"
flock -n 9 || exit 73
[ ! -e "$CASE/training_camera_v2/runtime_v2" ] && [ ! -e "$CASE/control_v2/wrapper_exit_code" ] || exit 73
trap 'rc=$?; trap - EXIT; set -o noclobber; printf "%s\n" "$rc" > "$CASE/control_v2/wrapper_exit_code"; exit "$rc"' EXIT
source "$ROOT/activate.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$ROOT/env_runtime/bin/python" -B "$CASE/control_v2/run_relay.py"
