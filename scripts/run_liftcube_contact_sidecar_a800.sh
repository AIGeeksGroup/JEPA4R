#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/users/lewm/liftcube-frozen-probe
OUTPUT=/workspace/users/lewm/runs/liftcube_contact_labels_20260909_v1
SOURCE=/workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100
COMPACT=/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0
PYTHON=/nix/lewm/4d-jepa-mpc/bin/python

mkdir -p "$OUTPUT"
printf '%q ' "$PYTHON" "$ROOT/scripts/generate_liftcube_contact_sidecar.py" \
  --h5 "$SOURCE/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5" \
  --json "$SOURCE/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.json" \
  --compact-root "$COMPACT" \
  --output-dir "$OUTPUT" \
  --alignment-tolerance-m 1e-4 > "$OUTPUT/command.txt"
printf '\n' >> "$OUTPUT/command.txt"

"$PYTHON" "$ROOT/scripts/generate_liftcube_contact_sidecar.py" \
  --h5 "$SOURCE/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5" \
  --json "$SOURCE/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.json" \
  --compact-root "$COMPACT" \
  --output-dir "$OUTPUT" \
  --alignment-tolerance-m 1e-4 \
  2>&1 | tee "$OUTPUT/run.log"
