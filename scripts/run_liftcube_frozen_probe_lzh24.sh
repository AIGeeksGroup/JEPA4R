#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/lzh_24/project/JEPA-4R/worktrees/liftcube-frozen-probe
OUTPUT=/public/home/lzh_24/project/JEPA-4R/runs/liftcube_frozen_probe_20260909_v1
CHECKPOINT=/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt
DATA=/public/home/lzh_24/project/JEPA-4R/data/compact_rgb_proprio/LiftCube-v0
PYTHON=/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python
DEVICE=${DEVICE:-cuda:0}

mkdir -p "$OUTPUT"
printf '%q ' "$PYTHON" "$ROOT/scripts/eval_liftcube_frozen_probe.py" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA" \
  --contact-sidecar "$OUTPUT/contact_sidecar.npz" \
  --output-dir "$OUTPUT" \
  --device "$DEVICE" \
  --batch-size 8 \
  --alignment-tolerance-m 1e-4 \
  --ridge-alpha 1e-3 \
  --logistic-l2 1e-3 \
  --logistic-max-iter 500 > "$OUTPUT/command.txt"
printf '\n' >> "$OUTPUT/command.txt"

"$PYTHON" "$ROOT/scripts/eval_liftcube_frozen_probe.py" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA" \
  --contact-sidecar "$OUTPUT/contact_sidecar.npz" \
  --output-dir "$OUTPUT" \
  --device "$DEVICE" \
  --batch-size 8 \
  --alignment-tolerance-m 1e-4 \
  --ridge-alpha 1e-3 \
  --logistic-l2 1e-3 \
  --logistic-max-iter 500 \
  2>&1 | tee "$OUTPUT/run.log"
