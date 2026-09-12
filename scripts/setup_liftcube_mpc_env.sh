#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
env_dir=/nix/lewm/4d-jepa-mpc
log_root=/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment
python_bin="${MPC_BOOTSTRAP_PYTHON:-/nix/lewm/4d-jepa-worldmodel/bin/python}"
checkpoint="${MPC_CHECKPOINT:-/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500/best.pt}"
checkpoint_sha256="${MPC_CHECKPOINT_SHA256:-38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32}"

mkdir -p "$log_root"
log_file="$log_root/install-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log_file") 2>&1

echo "repo_dir=$repo_dir"
echo "env_dir=$env_dir"
echo "bootstrap_python=$python_bin"
cd "$repo_dir"

"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" wheel
"$env_dir/bin/python" -m pip install -r requirements-mpc.txt
"$env_dir/bin/python" scripts/check_liftcube_mpc_env.py \
  --checkpoint "$checkpoint" \
  --expected-checkpoint-sha256 "$checkpoint_sha256"
