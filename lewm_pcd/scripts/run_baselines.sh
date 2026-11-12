#!/usr/bin/env bash
# Paper baselines that live in the framework (scripts untouched): PLDM, GCBC, GCIQL, GCIVL on our cube dataset,
# 10 epochs each, trained ONE AT A TIME (pixel trainings together exceed the 90 GB memory cap); an eval loop runs
# alongside on the GPU and evaluates every checkpoint with the framework's own eval scripts (eval_wm.py / eval_ff.py,
# CEM 10 iterations for the world model). Detached:
#   setsid nohup bash scripts/run_baselines.sh > $STABLEWM_HOME/logs/train/baselines.log 2>&1 < /dev/null &
set -uo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd; source "$P/env.sh"; F=$P/third_party/stable-worldmodel; PY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python
export MUJOCO_GL=egl
log() { echo "$(date +%F_%T) $*"; }
cd "$P"
CSV=$STABLEWM_HOME/logs/eval/success_rates_cem10.csv SUFFIX=_cem10 GPU_ONLY=1 GPU_MIN_FREE_MB=6000 LAUNCH_MAX_ANON_GB=45 KILL_ANON_GB=70 \
RUNS="pldm_cube gcbc_cube" setsid nohup bash scripts/eval/auto_eval_checkpoints.sh >> "$STABLEWM_HOME/logs/eval/auto_eval_cem10.log" 2>&1 < /dev/null &
log "eval loop started for the baselines"
cd "$F"
train() { name=$1; shift; log "TRAIN $name: $*"; "$@" > "$STABLEWM_HOME/logs/train/launch_${name}_$(date +%Y%m%d_%H%M%S).log" 2>&1; log "TRAIN $name exited with $? (checkpoints: $(ls $STABLEWM_HOME/checkpoints/$name/weights_epoch_*.pt 2>/dev/null | wc -l))"; }
train pldm_cube  $PY scripts/train/pldm.py data=ogb subdir=pldm_cube output_model_name=pldm_cube trainer.max_epochs=10
train gcbc_cube  $PY scripts/train/gcbc.py  data=ogb output_model_name=gcbc_cube wm.action_dim=5 +wandb.enable=false trainer.max_epochs=10
# (eval not possible as released: gciql.py saves the whole hydra cfg as config.json, load_pretrained cannot rebuild the model)
#train gciql_cube $PY scripts/train/gciql.py dataset_name=ogbench/cube_single_expert.h5 output_model_name=gciql_cube +subdir=gciql_cube dinowm.action_dim=5 +wandb.enable=false trainer.max_epochs=10
#train gcivl_cube $PY scripts/train/gcivl.py dataset_name=ogbench/cube_single_expert.h5 output_model_name=gcivl_cube +subdir=gcivl_cube dinowm.action_dim=5 +wandb.enable=false trainer.max_epochs=10
log "all baseline trainings finished"
