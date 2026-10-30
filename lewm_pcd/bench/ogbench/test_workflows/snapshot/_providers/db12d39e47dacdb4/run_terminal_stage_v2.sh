#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
T=/root/autodl-tmp/workspace/users/lewm/coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/terminal_v1
MODE=${1:?pcd or rgb}; case "$MODE" in pcd|rgb) ;; *) exit 2;; esac
P=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
A="$T/stages/${MODE}_$(date -u +%Y%m%dT%H%M%S)_pid$$"
mkdir -p "$A"
trap 'code=$?; printf "%s\n" "$code" > "$A/outer_exit_code"; date -u > "$A/finished_at"' EXIT
date -u > "$A/started_at"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Source/plan are Root-reviewed and immutable before explicit GPU stage dispatch.
"$P" - "$T/source_freeze_v2.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
for p,h in json.loads(Path(sys.argv[1]).read_text())['source_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
PY
"$P" "$T/terminal_producer.py" --plan "$T/terminal_plan.$MODE.v2.json" --execute > "$A/producer.log" 2>&1
