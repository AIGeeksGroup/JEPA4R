"""One CortexBench MuJoCo imitation run: eai-vc's cortexbench/mujoco_vc/visual_imitation/hydra_launcher.py +
train_loop.py (commit 76fe35e, third_party/eai-vc), unchanged on disk, with three runtime patches that do not change the
reported numbers:

1. Embedding configs: an embedding named NAME is read from conf/model/NAME.yaml next to this file when that file exists
   (the LeWM / JEPA4R encoders and the extra baselines), otherwise from vc_models' own conf/model (VC-1). The official
   load_pretrained_model does the loading either way.
2. Logging: `setup_wandb` returns a logger that appends every `wandb_run.log(...)` record to metrics.jsonl in the
   hydra run dir (wandb is disabled here). The records are the ones the official loop logs, e.g. eval/highest_success.
3. The "train/*" rollouts: after every evaluation the official loop rolls the policy out a second time from the
   training demos' initial states (Adroit, DMC) or from seed 54321 (MetaWorld) and logs them as train/* only. They are
   replaced by the evaluation paths just computed. The evaluation rollouts are untouched: mjrl's do_rollout reseeds the
   env and numpy with base_seed + episode before every episode, and nothing in BC training draws from numpy or from the
   policy rollouts, so the evaluation episodes, the trained policy and eval/* are the same; only the logging-only
   train/* values change (they repeat eval/*). This halves the rollout time.
4. Adroit fresh evaluation (the Relocate runs: --config-name Adroit_BC_config with
   ADROIT_FRESH_EVAL=1): the seed-54321 rollouts are not replaced by the evaluation paths.

    source env.sh && $PY run_bc.py --config-name Metaworld_BC_config env=... embedding=... seed=100 data_dir=... \\
        hydra.run.dir=...
"""

import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the eai-vc checkout: CORTEXBENCH_EAI_VC_ROOT (set by the Relocate workflow: eai-vc with its train-loop patch), else
# JEPA4R_EAI_VC (MetaWorld, Scene, the RM75-B tasks), else third_party/eai-vc
EAI_VC = (
    os.environ.get("CORTEXBENCH_EAI_VC_ROOT")
    or os.environ.get("JEPA4R_EAI_VC")
    or os.path.join(HERE, "..", "..", "..", "third_party", "eai-vc")
)
VI = os.path.join(EAI_VC, "cortexbench", "mujoco_vc", "visual_imitation")
# <CONF_HOME>/conf/model/NAME.yaml: the embedding configs (a run's own, or configs/policy/bc of this repository)
CONF_HOME = os.environ.get("JEPA4R_BC_CONF_HOME", os.path.join(HERE, "..", "..", "..", "configs", "policy", "bc"))
sys.path.insert(0, VI)

import mujoco_vc.gym_wrapper as gw
import mujoco_vc.model_loading as ml
import numpy as np
import train_loop

_load = ml.load_pretrained_model  # the official one, or run_bc3d.py's point-cloud-aware wrapper of it


def load_pretrained_model(embedding_name, *args, **kwargs):
    if not os.path.exists(os.path.join(CONF_HOME, "conf", "model", embedding_name + ".yaml")):
        return _load(embedding_name, *args, **kwargs)
    # the loader reads <vc_models_dir_path>/conf/model/NAME.yaml
    saved, ml.vc_models_dir_path = ml.vc_models_dir_path, CONF_HOME
    try:
        return _load(embedding_name, *args, **kwargs)
    finally:
        ml.vc_models_dir_path = saved


for mod in (ml, gw, train_loop):
    mod.load_pretrained_model = load_pretrained_model


class JsonlRun:
    def __init__(self, path):
        self.f = open(path, "a")

    def log(self, data=None, step=None, **kwargs):
        rec = {}
        for k, v in (data or {}).items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = str(v)
        if step is not None:
            rec["_step"] = int(step)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def finish(self):
        self.f.close()


train_loop.setup_wandb = lambda config: JsonlRun(os.path.join(os.getcwd(), "metrics.jsonl"))

_sample_paths = train_loop.sample_paths
_last = {}
_config_name = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--config-name=")), None)
if "--config-name" in sys.argv:
    _config_name = sys.argv[sys.argv.index("--config-name") + 1]


def sample_paths(*args, **kwargs):
    if (
        kwargs.get("base_seed") == 54321
        and "paths" in _last
        and not (  # MetaWorld train/* rollouts: logging only
            _config_name == "Adroit_BC_config" and os.environ.get("ADROIT_FRESH_EVAL") == "1"
        )
    ):
        return _last["paths"]
    paths = _sample_paths(*args, **kwargs)
    _last["paths"] = paths
    return paths


def rollout_from_init_states(*args, **kwargs):  # Adroit / DMC train/* rollouts: logging only
    return _last["paths"]


train_loop.sample_paths = sample_paths
train_loop.rollout_from_init_states = rollout_from_init_states
np.seterr(all="ignore")

sys.argv = [os.path.join(VI, "hydra_launcher.py")] + sys.argv[1:]
runpy.run_path(os.path.join(VI, "hydra_launcher.py"), run_name="__main__")
