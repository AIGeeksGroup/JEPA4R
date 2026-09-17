#!/usr/bin/env bash
set -euo pipefail

P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$P/env_franka.sh"

PY=${PYTHON:-python}
TRAINED_RUN=franka_pcd_ms2_10ep_seed3072
RANDOM_RUN=franka_pcd_ms2_random_init_seed3072
RANDOM_WEIGHTS=weights_random_init.pt
DATA_NAME=ogbench/franka_full/cube_franka_single_expert_pcd.rechunk20
SWEEP_STATUS="$STABLEWM_HOME/logs/franka_epoch_cem_sweep.status"
STATUS="$STABLEWM_HOME/logs/franka_random_init_cem.status"
LOCK="$STABLEWM_HOME/logs/franka_random_init_cem.lock"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="$STABLEWM_HOME/logs/eval/${RANDOM_RUN}_cem100_$STAMP"
mkdir -p "$LOG_ROOT" "$(dirname "$STATUS")"

exec 9> "$LOCK"
flock -n 9 || {
    echo "Another Franka random-init CEM is active" >&2
    exit 6
}

write_status() {
    local stage=$1 tmp
    tmp=$(mktemp "${STATUS}.tmp.XXXXXX")
    printf 'stage=%s time=%s pid=%s log_root=%s\n' \
        "$stage" "$(date --iso-8601=seconds)" "$$" "$LOG_ROOT" > "$tmp"
    mv "$tmp" "$STATUS"
}

on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        write_status "FAILED_exit_code_$rc"
    fi
}
trap on_exit EXIT

write_status CREATING_RANDOM_CHECKPOINT
cd "$P"
CUDA_VISIBLE_DEVICES= "$PY" - "$STABLEWM_HOME" "$TRAINED_RUN" "$RANDOM_RUN" "$RANDOM_WEIGHTS" <<'PY'
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

home, trained_run, random_run, filename = sys.argv[1:]
home = Path(home)
source_config = home / "checkpoints" / trained_run / "config.yaml"
target_dir = home / "checkpoints" / random_run
target_dir.mkdir(parents=True, exist_ok=True)
target_weights = target_dir / filename
target_config = target_dir / "config.json"
manifest_path = target_dir / "random_init_manifest.json"

seed = 3072
if not target_weights.exists():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = OmegaConf.load(source_config)
    model = hydra.utils.instantiate(cfg.model)

    weights_tmp = target_weights.with_suffix(target_weights.suffix + ".tmp")
    torch.save(model.state_dict(), weights_tmp)
    os.replace(weights_tmp, target_weights)

    config = OmegaConf.to_container(cfg.model, resolve=True)
    config_tmp = target_config.with_suffix(target_config.suffix + ".tmp")
    config_tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(config_tmp, target_config)

sha256 = hashlib.sha256(target_weights.read_bytes()).hexdigest()
epoch10 = home / "checkpoints" / trained_run / "weights_epoch_10.pt"
assert sha256 != hashlib.sha256(epoch10.read_bytes()).hexdigest()
manifest = {
    "checkpoint": str(target_weights),
    "kind": "random_initialized_world_model",
    "seed": seed,
    "sha256": sha256,
    "source_architecture_config": str(source_config),
    "training_steps": 0,
}
manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(manifest_tmp, manifest_path)
print(json.dumps(manifest, sort_keys=True))
PY

write_status WAITING_FOR_EPOCH_SWEEP
while true; do
    if grep -q '^stage=COMPLETE ' "$SWEEP_STATUS" 2>/dev/null; then
        break
    fi
    if grep -q '^stage=FAILED' "$SWEEP_STATUS" 2>/dev/null; then
        echo "Epoch CEM sweep failed; random-init control will not start" >&2
        exit 3
    fi
    sleep 30
done

write_status EVALUATING
started=$(date +%s)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
    "$PY" eval.py \
    --config-name=cube_franka_pcd \
    policy="$RANDOM_RUN/$RANDOM_WEIGHTS" \
    eval.dataset_name="$DATA_NAME" eval.num_eval=100 \
    output.filename="franka_random_init_cem100_${STAMP}_results.txt" video=false \
    hydra.run.dir="$LOG_ROOT/hydra" \
    2>&1 | tee "$LOG_ROOT/cem100.log"
elapsed=$(( $(date +%s) - started ))

"$PY" scripts/eval/summarize_cem_result.py "$LOG_ROOT/cem100.log" \
    --episodes 100 --checkpoint "$RANDOM_RUN/$RANDOM_WEIGHTS" \
    --evaluation-seconds "$elapsed" --out "$LOG_ROOT/cem100.json"
"$PY" - "$LOG_ROOT/cem100.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
assert record["completed"] == 100
assert record["episodes"] == 100
assert 0 <= record["successes"] <= 100
assert 0.0 <= record["success_rate"] <= 100.0
PY

write_status COMPLETE
