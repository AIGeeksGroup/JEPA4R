#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || $1 != --episodes ]]; then
  echo "usage: bash run_ur5e_cem_nine_tasks.sh --episodes 1|5|100" >&2
  exit 2
fi
EPISODES=$2
case "$EPISODES" in
  1|5|100) ;;
  *) echo "episodes must be 1, 5, or 100" >&2; exit 2 ;;
esac

ROOT=/root/autodl-tmp/workspace/users/lewm/JEPA4R-ur5e-multitask/lewm_pcd
RUNTIME=/root/autodl-tmp/workspace/users/lewm/ogbench-ur5e-multitask
PY=/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python
OUT="$RUNTIME/cem_dexart_100"
mkdir -p "$OUT"
exec 9>"$OUT/.queue.lock"
if ! flock -n 9; then
  echo "another UR5e CEM queue owns the lock" >&2
  exit 1
fi

export MUJOCO_GL=egl
export STABLEWM_HOME="$RUNTIME"
export LEWM_PCD_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
cd "$ROOT"

tasks=(
  cube_double_task1_single_pnp
  cube_double_task2_double_pnp1
  cube_double_task3_double_pnp2
  cube_double_task4_swap
  cube_double_task5_stack
  cube_triple_task1_single_pnp
  cube_triple_task2_triple_pnp
  cube_triple_task3_pnp_from_stack
  cube_triple_task4_cycle
)

printf 'RUNNING stage=%s time=%s\n' "$EPISODES" "$(date -u +%FT%TZ)" >"$OUT/queue.status"
for task in "${tasks[@]}"; do
  mkdir -p "$OUT/$task"
  printf 'RUNNING stage=%s task=%s time=%s\n' "$EPISODES" "$task" "$(date -u +%FT%TZ)" >"$OUT/queue.status"
  if ! "$PY" -m scripts.eval.evaluate_ur5e_fixed_pcd \
    --task "$task" --episodes "$EPISODES" --runtime "$RUNTIME" \
    --output "$OUT/$task" >"$OUT/$task/run_${EPISODES}.log" 2>&1; then
    printf 'FAILED stage=%s task=%s time=%s\n' "$EPISODES" "$task" "$(date -u +%FT%TZ)" >"$OUT/queue.status"
    exit 1
  fi
  if ! "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["completed"] == int(sys.argv[2]), d' \
    "$OUT/$task/summary.json" "$EPISODES"; then
    printf 'FAILED_SUMMARY stage=%s task=%s time=%s\n' "$EPISODES" "$task" "$(date -u +%FT%TZ)" >"$OUT/queue.status"
    exit 1
  fi
done
printf 'ACCEPTED stage=%s tasks=9 time=%s\n' "$EPISODES" "$(date -u +%FT%TZ)" >"$OUT/queue.status"
