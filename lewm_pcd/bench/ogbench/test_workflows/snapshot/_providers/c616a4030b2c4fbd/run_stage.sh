#!/usr/bin/env bash
set -euo pipefail
W=/root/autodl-tmp/workspace/users/lewm
N=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
B=$W/ogbench-official-single-play-20261006
S=$W/downstream-smoke-20261005-2133/state/official-source
CONFIG=$1
STAGE=${2:-describe}
case "$STAGE" in describe|gate|aggregate) test "$#" -le 2;; task) test "$#" = 3; case "$3" in 1|2|3|4|5) ;; *) exit 2;; esac;; *) exit 2;; esac
P=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
PROFILE=$("$P" - "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());p=Path(c['runtime_profile'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==c.get('runtime_profile_sha256','e36f92e198ba0e51fce7a9c4806dccd29c9530b8067b983c1d8ecaf841f9eca2')
print(p)
PY
)
source "$PROFILE"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$N:$S/impls:$S:$LEWM_JAX_CPU:$LEWM_PCD_ROOT:$LEWM_PCD_ROOT/third_party/stable-worldmodel:$B/official_lewm_gcbc_20261006:$B/official_state_gciql_launch_109_20261009_v1/native_eval_prepare:$W/coordination/pcd_frozen_lewm_gciql_100k_sr_109_20261010_v1:$B"
mapfile -t PATHS < <("$LEWM_PYTHON" - "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
for p,h in c['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print(c['train_root']);print(c['native_output']);print(c['reference_replay']);print(c.get('policy_kind','cached-state-xyz'))
PY
)
test "${#PATHS[@]}" = 4
A="$N/aux/stage_launches/$(date -u +%Y%m%dT%H%M%S)_pid$$"
mkdir -p "$A"
trap 'rc=$?; printf "%s\n" "$rc" > "$A/exit_code"; date -u > "$A/finished_at"' EXIT
cp "$CONFIG" "$A/config.json"
ARGV=("$LEWM_PYTHON" "$N/official_state_sr.py" --policy-kind="${PATHS[3]}" --algorithm=gciql --config "$CONFIG" --reference-replay "${PATHS[2]}" --train-root "${PATHS[0]}" --output "${PATHS[1]}" --stage "$STAGE")
if [ "$STAGE" = task ]; then ARGV+=(--task "$3"); fi
printf '%q ' "${ARGV[@]}" > "$A/argv.txt"
date -u > "$A/started_at"
"${ARGV[@]}" > "$A/run.log" 2>&1
