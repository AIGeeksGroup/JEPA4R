#!/usr/bin/env bash
# Linux x86_64 / Python 3.10. Install only into project-local environments.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
cd "$TASK_RL_ROOT"
mkdir -p .cache/pip artifacts vendor
rl_python="${RL_PYTHON:-python3.10}"
if ! command -v "$rl_python" >/dev/null 2>&1; then
  echo "Python 3.10 is required. Set RL_PYTHON to its executable." >&2
  exit 1
fi
"$rl_python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
"$rl_python" scripts/bootstrap_sources.py --group rl
if [[ ! -x .venv-rl/bin/python ]]; then
  "$rl_python" -m venv --without-pip .venv-rl
fi
.venv-rl/bin/python -c 'import sys; from pathlib import Path; assert sys.version_info[:2] == (3, 10); assert Path(sys.prefix).resolve() == Path(".venv-rl").resolve(), "Refusing a non-local environment"'
if ! .venv-rl/bin/python -m pip --version >/dev/null 2>&1; then
  if ! .venv-rl/bin/python -m ensurepip --upgrade; then
    # For distributions without ensurepip, another interpreter's pip installs
    # into this venv only. Set RL_BOOTSTRAP_PYTHON if python3 has no pip.
    rl_bootstrap_python="${RL_BOOTSTRAP_PYTHON:-python3}"
    "$rl_bootstrap_python" -m pip --python "$TASK_RL_ROOT/.venv-rl/bin/python" \
      install pip==26.2.1 --index-url https://pypi.org/simple
  fi
fi
.venv-rl/bin/python -m pip install \
  pip==26.2.1 setuptools==69.5.1 wheel==0.48.0 --index-url https://pypi.org/simple
.venv-rl/bin/python -m pip install \
  'https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp310-cp310-linux_x86_64.whl' \
  'https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp310-cp310-linux_x86_64.whl' \
  --index-url https://pypi.org/simple
.venv-rl/bin/python -m pip install -r requirements-rl.lock \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
.venv-rl/bin/python -m pip check
