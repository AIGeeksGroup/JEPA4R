#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Runs the missing jobs of every paper table, then writes outputs/report.md.
# Extra arguments are passed to each reproduce call, e.g. --stage train_wm or --where env=scene.
for spec in experiments/tab1_ogbench.yaml experiments/tab2_cortexbench.yaml experiments/tab3_recon.yaml \
            experiments/tab4_ablations.yaml experiments/params.yaml experiments/twin.yaml; do
  python -m jepa4r reproduce "$spec" --run "$@"
done
python -m jepa4r report
