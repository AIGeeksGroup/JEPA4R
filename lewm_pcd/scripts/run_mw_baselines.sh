#!/usr/bin/env bash
# Baselines on the five CortexBench MetaWorld tasks, trained on the same play data as LeWM (the datasets of
# config/train/data/mw_<task>.yaml) with LeWM's budget (10 epochs, epoch 10 scored), then scored by bench/cortexbench
# (frozen encoder + behaviour cloning, seeds 100/200/300):
#   LeJEPA      scripts/baselines/train_lejepa_mw.py                                 -> checkpoints/lejepa_mw_<task>
#   PLDM        the framework's scripts/train/pldm.py, data=ogb on the task's dataset -> checkpoints/pldm_mw_<task>
#   V-JEPA 2.1  ViT-B as released, frozen, no training (model vjepa21_vitb; needs third_party/vjepa2 and the weights,
#               see bench/cortexbench/README.md)
# Two training lanes (LeJEPA, PLDM) run at the same time, each over the five tasks in turn. A CortexBench lane scores
# V-JEPA 2.1 first (2 slots: 4 ran out of GPU memory next to the two trainings), then every model as soon as its
# epoch 10 exists (4 slots, 3 seeds). Idempotent: finished trainings and CortexBench runs are skipped. Detached:
#   STABLEWM_HOME=<checkout>/lewm_pcd_data PY=<lewm_pcd python> CB_PY=<cortexbench python> \
#     setsid nohup bash scripts/run_mw_baselines.sh > $STABLEWM_HOME/logs/mw_baselines.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."  # lewm_pcd
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data home (<checkout>/lewm_pcd_data)}"
: "${PY:?set PY to the lewm_pcd python}"; : "${CB_PY:?set CB_PY to the cortexbench python}"
L=$PWD
F=$L/third_party/stable-worldmodel
TASKS=(binpick assembly button drawer hammer)
declare -A CB_TASK=([binpick]=bin-picking [assembly]=assembly [button]=button-press-topdown [drawer]=drawer-open [hammer]=hammer)
export MUJOCO_GL=egl HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       SPT_CACHE_DIR=$STABLEWM_HOME/checkpoints/_spt_cache WANDB_MODE=offline WANDB_DIR=$STABLEWM_HOME/logs/train
LOGS=$STABLEWM_HOME/logs
DONE_DIR=$LOGS/cortexbench/baselines_done
mkdir -p "$LOGS/train" "$DONE_DIR"
log() { echo "$(date +%F_%T) $*"; }

for t in "${TASKS[@]}"; do
  [ -f "$STABLEWM_HOME/datasets/metaworld/mw_$t.h5" ] || { log "missing dataset metaworld/mw_$t.h5 (run scripts/data/run_mw_collection.sh)"; exit 1; }
done

trained() { [ -f "$STABLEWM_HOME/checkpoints/$1/weights_epoch_10.pt" ]; }

train_lejepa() {  # task
  local t=$1 r=lejepa_mw_$1
  trained "$r" && return 0
  log "training $r"
  PYTHONPATH=$L:$F $PY scripts/baselines/train_lejepa_mw.py --task "mw_$t" > "$LOGS/train/train_$r.log" 2>&1 \
    && log "trained $r" || log "training $r FAILED (see $LOGS/train/train_$r.log)"
}

train_pldm() {  # task
  local t=$1 r=pldm_mw_$1
  trained "$r" && return 0
  log "training $r"
  mkdir -p "$LOGS/train/$r"
  ( cd "$LOGS/train/$r" && PYTHONPATH=$L:$F $PY "$F/scripts/train/pldm.py" data=ogb data.dataset.name="metaworld/mw_$t.h5" \
      '~data.dataset.keys_to_merge' subdir="$r" output_model_name="$r" trainer.max_epochs=10 > "$LOGS/train/train_$r.log" 2>&1 ) \
    && log "trained $r" || log "training $r FAILED (see $LOGS/train/train_$r.log)"
}

lane() {  # training function
  local t
  for t in "${TASKS[@]}"; do $1 "$t"; done
}

cortexbench() {  # model tasks(comma list) slots
  [ -f "$DONE_DIR/$1" ] && return 0
  log "CortexBench: $1 on $2"
  ( export PY=$CB_PY && cd bench/cortexbench && source env.sh \
      && $PY grid.py --models "$1" --tasks "$2" --slots "$3" >> "$LOGS/cortexbench/grid.log" 2>&1 ) \
    && touch "$DONE_DIR/$1" && log "CortexBench $1 finished" || log "CortexBench $1 FAILED (see $LOGS/cortexbench/grid.log)"
}

cb_lane() {  # V-JEPA 2.1 first, then every trained model; every 5 minutes until the trainings are over
  local all m t pending
  all=$(IFS=,; echo "${CB_TASK[binpick]},${CB_TASK[assembly]},${CB_TASK[button]},${CB_TASK[drawer]},${CB_TASK[hammer]}")
  cortexbench vjepa21_vitb "$all" 2
  while true; do
    for m in lejepa pldm; do for t in "${TASKS[@]}"; do
      trained "${m}_mw_$t" && cortexbench "${m}_mw_$t" "${CB_TASK[$t]}" 4
    done; done
    if [ -f "$DONE_DIR/.trainings_over" ]; then
      pending=0
      for m in lejepa pldm; do for t in "${TASKS[@]}"; do
        trained "${m}_mw_$t" && [ ! -f "$DONE_DIR/${m}_mw_$t" ] && pending=1
      done; done
      [ $pending = 0 ] && break
    fi
    sleep 300
  done
  ( export PY=$CB_PY && cd bench/cortexbench && source env.sh && $PY summarize.py > /dev/null )
}

rm -f "$DONE_DIR/.trainings_over"
cb_lane & CB=$!
lane train_lejepa & LJ=$!
lane train_pldm & PL=$!
wait $LJ $PL
touch "$DONE_DIR/.trainings_over"
log "both training lanes finished"
wait $CB
log "ALL_DONE (CortexBench tables: $LOGS/cortexbench/results/scores.md)"
