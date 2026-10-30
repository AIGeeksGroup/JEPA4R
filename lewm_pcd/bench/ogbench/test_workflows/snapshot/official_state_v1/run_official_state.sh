#!/usr/bin/env bash
# Preview by default. This wrapper does not download data or alter official code.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /absolute/cube-triple-play-v0.npz /absolute/new_save_dir [--execute]" >&2
  exit 2
fi
TRAIN_NPZ="$1"
SAVE_DIR="$2"
[[ "$TRAIN_NPZ" = /* && "$SAVE_DIR" = /* ]]
[[ "$(basename -- "$TRAIN_NPZ")" = cube-triple-play-v0.npz ]]
[[ -f "$TRAIN_NPZ" && -f "${TRAIN_NPZ%.npz}-val.npz" ]]
[[ ! -e "$SAVE_DIR" ]]
if [[ $# = 2 ]]; then
  printf 'Prepared only: official cube-triple-play-v0; state46/goal46; alpha1; B1024; 1M; eval50.\nTrain: %s\nOutput: %s\n' "$TRAIN_NPZ" "$SAVE_DIR"
  exit 0
fi
[[ "$3" = --execute ]]
source "$TASK_ROOT/activate.sh"
export CUBE_OFFICIAL_TRAIN_NPZ="$TRAIN_NPZ"
cd "$TASK_ROOT/source/ogbench/impls"
# The only switch forwards an explicit existing dataset path to the original loader.
exec python -B -c 'import functools, os, runpy, ogbench; ogbench.make_env_and_datasets = functools.partial(ogbench.make_env_and_datasets, dataset_path=os.environ["CUBE_OFFICIAL_TRAIN_NPZ"]); runpy.run_path("main.py", run_name="__main__")' \
  --env_name=cube-triple-play-v0 --eval_episodes=50 \
  --agent=agents/gciql.py --agent.alpha=1.0 --save_dir="$SAVE_DIR"
