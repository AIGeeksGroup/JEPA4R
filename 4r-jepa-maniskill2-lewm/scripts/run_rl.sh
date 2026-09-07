#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
exec "${RL_LAUNCH_PYTHON:-python3}" "$TASK_RL_ROOT/scripts/run_ppo.py" "$@"
