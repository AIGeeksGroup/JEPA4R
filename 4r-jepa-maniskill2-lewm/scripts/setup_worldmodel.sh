#!/usr/bin/env bash
# Isolated setup; run setup_rl.sh first. GPU dependencies use a sibling env.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/rl_env.sh"
cd "$TASK_RL_ROOT"
mode="${1:-cpu}"
if [[ "$mode" != cpu && "$mode" != gpu ]]; then
  echo "Usage: $0 [cpu|gpu]" >&2
  exit 2
fi
if [[ ! -x .venv-rl/bin/python ]]; then
  echo "Run bash scripts/setup_rl.sh first." >&2
  exit 1
fi
.venv-rl/bin/python scripts/bootstrap_sources.py --group all
if [[ "$mode" == gpu && ! -x .venv-worldmodel/bin/python ]]; then
  echo "Run bash scripts/setup_worldmodel.sh cpu first." >&2
  exit 1
fi
wm_venv=.venv-worldmodel
if [[ "$mode" == gpu ]]; then
  wm_venv=.venv-worldmodel-gpu
fi
if [[ ! -x "$wm_venv/bin/python" ]]; then
  .venv-rl/bin/python -m venv --without-pip "$wm_venv"
fi
"$wm_venv/bin/python" - "$wm_venv" <<'PY'
import sys
from pathlib import Path
assert sys.version_info[:2] == (3, 10), "Python 3.10 required"
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), "Refusing a non-local environment"
PY
# Bootstrap pip before adding read-only dependency paths from sibling envs.
.venv-rl/bin/python -m pip --python "$TASK_RL_ROOT/$wm_venv/bin/python" \
  install pip==25.3 --index-url https://pypi.org/simple
"$wm_venv/bin/python" - "$mode" <<'PY'
import subprocess
import sys
import sysconfig
from pathlib import Path
root = Path.cwd()
siblings = [".venv-rl"] if sys.argv[1] == "cpu" else [".venv-worldmodel", ".venv-rl"]
paths = [subprocess.check_output(
    [str(root / sibling / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
    text=True,
).strip() for sibling in siblings]
(Path(sysconfig.get_paths()["purelib"]) / "task_dependencies.pth").write_text("\n".join(paths) + "\n")
PY
if [[ "$mode" == gpu ]]; then
  "$wm_venv/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128
  echo "GPU environment ready: $wm_venv. Source scripts/wm_env.sh before use."
  exit 0
fi
"$wm_venv/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0+cpu torchvision==0.23.0+cpu
"$wm_venv/bin/python" scripts/download_wm_dependencies.py --install
# Preserve the archived NumPy compatibility combination. pip check is
# diagnostic here: ManiSkill2 declares numpy<1.24; WM Lance needs >=1.24.
if ! "$wm_venv/bin/python" -m pip check > artifacts/wm_pip_check.txt 2>&1; then
  cat artifacts/wm_pip_check.txt >&2
  "$wm_venv/bin/python" - <<'PY'
from pathlib import Path
expected = "mani-skill2 0.5.3 has requirement numpy<1.24, but you have numpy 1.24.4."
issues = [line.strip() for line in Path("artifacts/wm_pip_check.txt").read_text().splitlines() if line.strip()]
if issues != [expected]:
    raise SystemExit("Unexpected dependency conflicts; inspect artifacts/wm_pip_check.txt")
PY
  echo "The documented ManiSkill2/NumPy metadata conflict is recorded in artifacts/wm_pip_check.txt." >&2
fi
echo "CPU integration environment ready: $wm_venv"
