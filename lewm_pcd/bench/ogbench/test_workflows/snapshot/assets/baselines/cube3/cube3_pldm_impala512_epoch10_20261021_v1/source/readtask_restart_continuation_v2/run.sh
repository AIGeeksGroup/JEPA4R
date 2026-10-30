#!/bin/bash
set -u
ROOT=/root/autodl-tmp/workspace/users/lewm/triple_cube_pldm_lejepa_b128_10ep_10155090_20261017_v1
DATA_ROOT=/root/autodl-tmp/workspace/users/lewm/triple_cube_portable_20261016_v1
OUT=$1; PARENT=$2; TRIAL=$3; RELEASE=$4
if [ -e "$OUT" ] || [ -e "$OUT.launch.log" ] || [ -e "$OUT.outer_exit_code" ]; then exit 2; fi
mkdir -p "$(dirname "$OUT")"
(
 source "$DATA_ROOT/activate.sh" || exit $?
 /root/autodl-tmp/conda/envs/triple-wm/bin/python -B "$ROOT/readtask_restart_continuation_v2/continuation.py" --mode full --root "$ROOT" --output "$OUT" --parent-proof "$PARENT" --trial-report "$TRIAL" --release "$RELEASE"
) > "$OUT.launch.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$OUT.outer_exit_code"
exit "$RC"
