#!/usr/bin/env bash
set -euo pipefail
P=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
A=/root/autodl-tmp/workspace/users/lewm/coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1/native1m_v1/aux/final_audit_v1
AUDITOR=$A/audit_rl3_profiles_native195.py
test "$#" -ge 7
CONFIG=$1; CONFIG_SHA=$2; DRIVER=$3; PLAN=$4; PLAN_SHA=$5; OUTPUT=$6
shift 6
LAUNCH=${OUTPUT}_launch
mkdir "$LAUNCH"
trap 'rc=$?; printf "%s\n" "$rc" > "$LAUNCH/outer_exit_code"; date -u > "$LAUNCH/finished_at"' EXIT
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$P" - "$AUDITOR" <<'PY'
import hashlib,sys
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()=='fdeacc9c6dc25c4ce557f3734b2ded95c1606db3702d89703a30ee000b1b2b1f'
PY
ARGV=("$P" "$AUDITOR" --mode full --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA" --driver "$DRIVER" --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA" --output "$OUTPUT")
for EXIT_FILE in "$@"; do ARGV+=(--required-zero-exit "$EXIT_FILE"); done
printf '%q ' "${ARGV[@]}" > "$LAUNCH/argv.txt"
date -u > "$LAUNCH/started_at"
"${ARGV[@]}" > "$LAUNCH/audit.log" 2>&1
