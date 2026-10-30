#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/env_runtime/bin/python"
LIB="$ROOT/runtime/lib:/usr/lib/x86_64-linux-gnu"
RUN="$HERE/runtime/formal_v1"

if [[ -e "$RUN" ]]; then
  echo "refusing existing formal runtime: $RUN" >&2
  exit 2
fi
mkdir -p "$RUN"
printf '%s\n' "$$" > "$RUN/launcher_pid"
printf '%s\n' '{"stage":"starting","completed_groups":[],"actual_exit_code":null}' > "$RUN/status.json.tmp"
mv "$RUN/status.json.tmp" "$RUN/status.json"
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$RUN/outer_exit_code"
  "$PY" - "$RUN/status.json" "$rc" <<'PY'
import json,sys,time
p=sys.argv[1]; rc=int(sys.argv[2])
try: d=json.load(open(p))
except Exception: d={}
d.update(actual_exit_code=rc,completed=time.time())
tmp=p+'.tmp'; json.dump(d,open(tmp,'w'),indent=2); open(tmp,'a').write('\n')
import os; os.replace(tmp,p)
PY
}
trap finish EXIT

for group in G05 G06; do
  phase="$RUN/$group"
  mkdir "$phase"
  printf '{"group":"%s","argv":["%s","-u","%s","--group","%s","--mode","full","--output","%s"],"cwd":"%s","started":%s,"environment":{"MUJOCO_GL":"egl","JAX_PLATFORMS":"cpu"}}\n' \
    "$group" "$PY" "$HERE/visual_sr.py" "$group" "$HERE/full/${group}_final1m_v1" "$HERE" "$(date +%s)" > "$phase/actual_argv.json"
  printf '{"stage":"running","group":"%s","started":%s}\n' "$group" "$(date +%s)" > "$RUN/status.json.tmp"
  mv "$RUN/status.json.tmp" "$RUN/status.json"
  set +e
  env LD_LIBRARY_PATH="$LIB" MUJOCO_GL=egl JAX_PLATFORMS=cpu \
    "$PY" -u "$HERE/visual_sr.py" --group "$group" --mode full --output "$HERE/full/${group}_final1m_v1" \
    > "$phase/run.log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$phase/child_pid"
  wait "$child"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$phase/actual_exit_code"
  if [[ $rc -ne 0 ]]; then
    printf '{"stage":"failed","group":"%s","actual_exit_code":%s}\n' "$group" "$rc" > "$RUN/status.json.tmp"
    mv "$RUN/status.json.tmp" "$RUN/status.json"
    exit "$rc"
  fi
  printf '{"stage":"group_complete","group":"%s","completed":%s,"actual_exit_code":0}\n' "$group" "$(date +%s)" > "$RUN/status.json.tmp"
  mv "$RUN/status.json.tmp" "$RUN/status.json"
done
printf '%s\n' '{"stage":"both_groups_complete","completed_groups":["G05","G06"],"actual_exit_code":0}' > "$RUN/status.json.tmp"
mv "$RUN/status.json.tmp" "$RUN/status.json"
exit 0
