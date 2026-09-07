#!/usr/bin/env bash
# CUDA lives in a sibling venv; the validated CPU environment remains intact.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
cd "$TASK_RL_ROOT"
if [[ ! -x .venv-rl/bin/python ]]; then
  echo "Run bash scripts/setup_rl.sh first." >&2
  exit 1
fi
.venv-rl/bin/python scripts/bootstrap_sources.py --group rl
if [[ ! -x .venv-rl-gpu/bin/python ]]; then
  .venv-rl/bin/python -m venv --without-pip .venv-rl-gpu
fi
.venv-rl-gpu/bin/python - <<'PY'
import sys
from pathlib import Path
assert sys.version_info[:2] == (3, 10), "Python 3.10 required"
assert Path(sys.prefix).resolve() == Path('.venv-rl-gpu').resolve(), 'Refusing a non-local environment'
PY
.venv-rl/bin/python -m pip --python "$TASK_RL_ROOT/.venv-rl-gpu/bin/python" install pip==25.3 --index-url https://pypi.org/simple
.venv-rl-gpu/bin/python - <<'PY'
import subprocess, sysconfig
from pathlib import Path
root=Path.cwd()
source=subprocess.check_output([str(root/'.venv-rl/bin/python'),'-c',"import sysconfig; print(sysconfig.get_paths()['purelib'])"],text=True).strip()
(Path(sysconfig.get_paths()['purelib'])/'rl_dependencies.pth').write_text(source+'\n')
PY
.venv-rl-gpu/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128
.venv-rl-gpu/bin/python -c 'import torch; assert torch.version.cuda is not None; print("CUDA build:", torch.version.cuda)'
echo "GPU PPO environment ready. Start with bash scripts/run_rl.sh train."
