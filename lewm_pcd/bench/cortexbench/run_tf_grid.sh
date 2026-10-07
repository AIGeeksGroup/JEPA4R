#!/usr/bin/env bash
# CortexBench TriFinger with the TriFinger LeWMs (run_tf_bc.py): reach_cube and move_cube x lewm_tf_pix / lewm_tf_pcd x
# seeds 100 / 200 / 300 (the MetaWorld CortexBench seeds); the official settings otherwise (bc_default: lr 1e-4, batch
# 32; reach 100 epochs evaluated every 5, move 1000 epochs evaluated every 25; 25 train + 25 test demos per evaluation).
# Demo home $STABLEWM_HOME/datasets/cortexbench/trifinger: the official split files (trifinger_vc's assets/bc_demos,
# copied when missing) and demos (assets/data/trifinger-demos: `run_tf_bc.py prep` downloads them when missing, then
# caches the demo point clouds; a no-op when cached). Then the runs, all at once; logs and metrics.jsonl in
# $STABLEWM_HOME/logs/cortexbench/trifinger/<encoder>/<task>/s<seed>{,.log}. A finished run (all its evaluations in
# metrics.jsonl) is skipped; an unfinished run directory is refused (metrics.jsonl would be appended to). Encoders:
# run_tf_bc.py's TF_CKPT_PIX / TF_CKPT_PCD / TF_EPOCH. In the lewm_tf python:
#   STABLEWM_HOME=... PY=<lewm_tf python> TF_EPOCH=10 bash bench/cortexbench/run_tf_grid.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # lewm_pcd
L=$PWD
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_tf python}"
DEMOS=$STABLEWM_HOME/datasets/cortexbench/trifinger
RUNS=$STABLEWM_HOME/logs/cortexbench/trifinger
declare -A EVALS=([reach_cube]=21 [move_cube]=41)  # evaluations of a finished run (summarize_tf.py)
todo=()
for task in move_cube reach_cube; do for rep in lewm_tf_pix lewm_tf_pcd; do for s in 100 200 300; do
  r=$RUNS/$rep/$task/s$s
  n=$(cat "$r/metrics.jsonl" 2>/dev/null | wc -l)
  [ "$n" -ge "${EVALS[$task]}" ] && continue
  [ ! -e "$r" ] || { echo "refusing to run: $r exists unfinished ($n of ${EVALS[$task]} evaluations)"; exit 1; }
  todo+=("$task $rep $s")
done; done; done
[ ${#todo[@]} -gt 0 ] || { echo "$(date +%F_%T) all runs finished"; exit 0; }
mkdir -p "$DEMOS/assets" "$RUNS"
[ -d "$DEMOS/assets/bc_demos" ] || cp -r "$L/third_party/eai-vc/cortexbench/trifinger_vc/assets/bc_demos" "$DEMOS/assets/"
cd "$DEMOS"
export PYTHONPATH=$L:$L/bench/cortexbench:$L/third_party/stable-worldmodel:$L/third_party/eai-vc/cortexbench/trifinger_vc/src:$L/third_party/eai-vc/vc_models/src \
       OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled HYDRA_FULL_ERROR=1
$PY $L/bench/cortexbench/run_tf_bc.py prep > "$RUNS/prep.log" 2>&1 || { echo "prep FAILED (see $RUNS/prep.log)"; exit 1; }
for job in "${todo[@]}"; do
  read -r task rep s <<< "$job"
  r=$RUNS/$rep/$task/s$s
  mkdir -p "$(dirname "$r")"
  nice -n 5 $PY $L/bench/cortexbench/run_tf_bc.py task=$task algo.pretrained_rep=$rep algo.freeze_pretrained_rep=true \
    rep_to_policy=none no_wandb=True seed=$s hydra.run.dir=$r hydra.job.chdir=False > $r.log 2>&1 \
    && echo "$(date +%F_%T) $rep $task s$s done" || echo "$(date +%F_%T) $rep $task s$s FAILED ($r.log)" &
done
wait
echo "$(date +%F_%T) ALL_DONE"
