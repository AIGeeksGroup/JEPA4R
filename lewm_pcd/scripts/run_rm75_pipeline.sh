#!/usr/bin/env bash
# The three reference runs on the RM75-B + EG2-4C2 (envs/cube_rm75_env.py), on freshly collected data:
#   1/5 point-cloud collection (config ogb_rm75_pcd: 20 shards at once) -> merge -> rechunk -> eval columns
#   2/5 pixel collection (config ogb_rm75_pix: 20 shards, 8 at a time -- a pixel shard grows to ~5 GB RAM) -> merge -> eval columns
#   3/5 random-policy evaluation on both datasets: sanity check of env + eval path (the 2000-episode trial gave 58-64 %,
#       the UR5e 10000-episode data 42.8 %); trainings start only if both rates are within 40..75 %
#   4/5 trainings, all with the repo's own configs: official 2D (data=ogb_rm75 model=lewm), 3D and 4D with one encoder (ENC)
#   5/5 per-epoch eval loop for the three runs (paper protocol, GPU only) -> logs/eval/success_rates_cem10.csv
# Idempotent: existing datasets / checkpoint dirs are skipped. Detached:
#   setsid nohup bash scripts/run_rm75_pipeline.sh > $STABLEWM_HOME/logs/rm75_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
PY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python
SHARDS=${SHARDS:-20}; MAXPAR_PIX=${MAXPAR_PIX:-8}; ENC=${ENC:-dexart}; NUM_TRAJ=${NUM_TRAJ:-10000}
case "$ENC" in ms2) M3=lewm_pcd;; *) M3=lewm_pcd_$ENC;; esac     # 3D model config per encoder (config/train/model/)
DS=$STABLEWM_HOME/datasets/ogbench
PCD=$DS/cube_rm75_single_expert_pcd.h5; PIX=$DS/cube_rm75_single_expert.h5
RUNS="cube_rm75_pix cube_rm75_pcd_${ENC} cube_rm75_pcd_agg_${ENC}"
log() { echo "$(date +%F_%T) $*"; }
rate() { grep -oE "'success_rate': [0-9.]+" "$1" | tail -1 | grep -oE "[0-9.]+$"; }
cd "$P"; mkdir -p "$STABLEWM_HOME/logs/eval"

if [ ! -f "$PCD" ]; then
  log "1/5 point-cloud collection: $SHARDS shards x $((NUM_TRAJ / SHARDS)) episodes -> $PCD"
  $PY scripts/data/collect_cube_pcd_sharded.py --shards "$SHARDS" --num-traj "$NUM_TRAJ" --seed 3072 \
      --out ogbench/cube_rm75_single_expert_pcd.raw.h5 -- --config-name=ogb_rm75_pcd || { log "pcd collection FAILED"; exit 1; }
  log "1/5 rechunk (20 rows per chunk, as the UR5e dataset) + eval columns"
  $PY scripts/data/rechunk_h5.py "$DS/cube_rm75_single_expert_pcd.raw.h5" "$PCD" || { log "rechunk FAILED"; exit 1; }
  rm -f "$DS/cube_rm75_single_expert_pcd.raw.h5"
  $PY scripts/data/add_eval_columns.py "$PCD" || { log "add_eval_columns FAILED"; exit 1; }
else log "1/5 point-cloud dataset exists: $PCD"; fi

if [ ! -f "$PIX" ]; then
  log "2/5 pixel collection: $SHARDS shards x $((NUM_TRAJ / SHARDS)) episodes, $MAXPAR_PIX at a time -> $PIX"
  $PY scripts/data/collect_cube_pcd_sharded.py --shards "$SHARDS" --num-traj "$NUM_TRAJ" --seed 3072 --max-parallel "$MAXPAR_PIX" \
      --script collect_cube_pix.py --out ogbench/cube_rm75_single_expert.h5 -- --config-name=ogb_rm75_pix || { log "pixel collection FAILED"; exit 1; }
  $PY scripts/data/add_eval_columns.py "$PIX" || { log "add_eval_columns FAILED"; exit 1; }
else log "2/5 pixel dataset exists: $PIX"; fi
$PY - "$PCD" "$PIX" <<'PY'
import sys, h5py, os
for p in sys.argv[1:]:
    with h5py.File(p) as f:
        print(f"{os.path.basename(p)}: {f['action'].shape[0]} rows, {f['ep_len'].shape[0]} episodes, {os.path.getsize(p)/1e9:.1f} GB, ep_idx {'ep_idx' in f}, columns {len(f)}")
PY

log "3/5 random-policy evaluation (50 starts, seed 42) on both datasets"
for kind in pcd pix; do
  L=$STABLEWM_HOME/logs/eval/rm75_random_${kind}.log
  [ -n "$(rate "$L" 2>/dev/null)" ] || MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY eval.py --config-name=cube_rm75_${kind}.yaml policy=random \
      device=cuda solver.device=cuda hydra.run.dir="$STABLEWM_HOME/logs/eval/hydra_rm75_random_${kind}" > "$L" 2>&1
  log "random policy, $kind: $(rate "$L") %"
done
R_PCD=$(rate "$STABLEWM_HOME/logs/eval/rm75_random_pcd.log"); R_PIX=$(rate "$STABLEWM_HOME/logs/eval/rm75_random_pix.log")
ok() { awk -v v="${1:-nan}" 'BEGIN{exit !(v >= 40 && v <= 75)}'; }
ok "$R_PCD" && ok "$R_PIX" || { log "random-policy rates out of the expected 40..75 % band (pcd $R_PCD, pix $R_PIX): stopping before training"; exit 2; }

log "4/5 trainings: $RUNS"
for spec in "lewm cube_rm75_pix ogb_rm75" "$M3 cube_rm75_pcd_${ENC} ogb_rm75_pcd" "lewm_pcd_agg cube_rm75_pcd_agg_${ENC} ogb_rm75_pcd --config-name=lewm_agg model/encoder=${ENC}"; do
  set -- $spec; model=$1; run=$2; data=$3; shift 3
  if [ -d "$STABLEWM_HOME/checkpoints/$run" ]; then log "  $run exists, not relaunched"; continue; fi
  DATA=$data bash scripts/train/run_train_detached.sh "$model" "$run" "$@" || { log "  launch of $run FAILED"; exit 1; }
done

log "5/5 per-epoch eval loop (paper protocol, GPU only)"
RUNS="$RUNS" CSV=$STABLEWM_HOME/logs/eval/success_rates_cem10.csv SUFFIX=_cem10 GPU_ONLY=1 GPU_MIN_FREE_MB=8000 \
  setsid nohup bash scripts/eval/auto_eval_checkpoints.sh >> "$STABLEWM_HOME/logs/eval/auto_eval_cem10.log" 2>&1 < /dev/null &
log "pipeline launched; results -> $STABLEWM_HOME/logs/eval/success_rates_cem10.csv"
