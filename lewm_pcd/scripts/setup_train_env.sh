#!/usr/bin/env bash
# Idempotent CUDA training env on the data disk. Run detached:
#   setsid nohup bash scripts/setup_train_env.sh > TMP/env_setup.log 2>&1 < /dev/null &
# Packages are installed with --no-deps from the pinned list of the working
# collection venv (pip's resolver rejects some of those pins on paper although
# they coexist fine in practice); `pip check` at the end reports what it dislikes.
set -uo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
ENV=/root/autodl-tmp/conda/envs/lewm_pcd
REQ=$P/TMP/req_train_env.txt
PIP="$ENV/bin/python -m pip install -q --root-user-action=ignore"
echo "=== $(date) setup_train_env ==="
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" python=3.10 || exit 1
$PIP --upgrade pip
echo "=== torch cu128 ==="
$PIP torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 || exit 1
echo "=== pinned deps, --no-deps (bulk, then per-package fallback) ==="
if ! $PIP --no-deps -r "$REQ"; then
  : > "$P/TMP/env_setup_failed.txt"
  while read -r pkg; do
    [ -z "$pkg" ] && continue
    $PIP --no-deps "$pkg" || echo "$pkg" >> "$P/TMP/env_setup_failed.txt"
  done < "$REQ"
  echo "per-package fallback done; failures: $(wc -l < "$P/TMP/env_setup_failed.txt")"; cat "$P/TMP/env_setup_failed.txt"
fi
echo "=== stable-worldmodel editable (6f1e499), no deps ==="
$PIP --no-deps -e "$P/third_party/stable-worldmodel" || exit 1
echo "=== pip check (informational) ==="
"$ENV/bin/python" -m pip check | head -20 || true
echo "=== verify ==="
"$ENV/bin/python" -c "import torch, stable_worldmodel, ogbench, mujoco, lance, h5py, lightning, stable_pretraining, pymunk, hydra, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| ogbench ok | swm', stable_worldmodel.__file__)" || exit 1
echo "=== $(date) DONE ==="
