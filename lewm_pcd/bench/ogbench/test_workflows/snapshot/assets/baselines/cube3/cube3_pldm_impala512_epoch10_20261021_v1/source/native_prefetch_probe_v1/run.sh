#!/bin/bash
set -u
ROOT=/root/autodl-tmp/workspace/users/lewm/triple_cube_pldm_lejepa_b128_10ep_10155090_20261017_v1
DATA_ROOT=/root/autodl-tmp/workspace/users/lewm/triple_cube_portable_20261016_v1
OUT=$1; PARENT=$2; RELEASE=$3
if [ -e "$OUT" ] || [ -e "$OUT.launch.log" ] || [ -e "$OUT.outer_exit_code" ]; then exit 2; fi
mkdir -p "$(dirname "$OUT")"
(
 source "$DATA_ROOT/activate.sh" || exit $?
 /root/autodl-tmp/conda/envs/triple-wm/bin/python -B "$ROOT/native_prefetch_probe_v1/trial.py" --mode aba --root "$ROOT" --output "$OUT" --parent-proof "$PARENT" --release "$RELEASE"
) > "$OUT.launch.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$OUT.outer_exit_code"
exit "$RC"
