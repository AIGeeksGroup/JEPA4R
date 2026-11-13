# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/cortexbench/run_bc.py (verbatim except: imports / paths adapted to the package)
"""Final Relocate adapter around the original EAI-VC BC loop.

Local model configurations select the paper encoders. Metrics are logged as
JSONL; the existing train/* bookkeeping repeats evaluator paths. The native
eval/* rollouts and optimization use the recorded CortexBench protocol.
"""

import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EAI_VC = os.environ.get('CORTEXBENCH_EAI_VC_ROOT', os.path.join(HERE, '..', '..', '..', '..', '..', 'third_party', 'eai-vc'))
VI = os.path.join(EAI_VC, 'cortexbench', 'mujoco_vc', 'visual_imitation')
sys.path.insert(0, VI)

import mujoco_vc.gym_wrapper as gw  # noqa: E402
import mujoco_vc.model_loading as ml  # noqa: E402
import numpy as np  # noqa: E402
import train_loop  # noqa: E402

_load = ml.load_pretrained_model  # the official one, or run_bc3d.py's point-cloud-aware wrapper of it


def load_pretrained_model(embedding_name, *args, **kwargs):
    if not os.path.exists(os.path.join(HERE, 'conf', 'model', embedding_name + '.yaml')):
        return _load(embedding_name, *args, **kwargs)
    saved, ml.vc_models_dir_path = ml.vc_models_dir_path, HERE  # the loader reads <vc_models_dir_path>/conf/model/NAME.yaml
    try:
        return _load(embedding_name, *args, **kwargs)
    finally:
        ml.vc_models_dir_path = saved


for mod in (ml, gw, train_loop):
    mod.load_pretrained_model = load_pretrained_model


class JsonlRun:
    def __init__(self, path):
        self.f = open(path, 'a')

    def log(self, data=None, step=None, **kwargs):
        rec = {}
        for k, v in (data or {}).items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = str(v)
        if step is not None:
            rec['_step'] = int(step)
        self.f.write(json.dumps(rec) + '\n')
        self.f.flush()

    def finish(self):
        self.f.close()


train_loop.setup_wandb = lambda config: JsonlRun(os.path.join(os.getcwd(), 'metrics.jsonl'))

_sample_paths = train_loop.sample_paths
_last = {}
_config_name = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--config-name=')), None)
if '--config-name' in sys.argv:
    _config_name = sys.argv[sys.argv.index('--config-name') + 1]


def sample_paths(*args, **kwargs):
    if kwargs.get('base_seed') == 54321 and 'paths' in _last and not (
            _config_name == 'Adroit_BC_config' and os.environ.get('ADROIT_FRESH_EVAL') == '1'):
        return _last['paths']
    paths = _sample_paths(*args, **kwargs)
    _last['paths'] = paths
    return paths


def rollout_from_init_states(*args, **kwargs):  # Adroit / DMC train/* rollouts: logging only
    return _last['paths']


train_loop.sample_paths = sample_paths
train_loop.rollout_from_init_states = rollout_from_init_states
np.seterr(all='ignore')

sys.argv = [os.path.join(VI, 'hydra_launcher.py')] + sys.argv[1:]
runpy.run_path(os.path.join(VI, 'hydra_launcher.py'), run_name='__main__')
