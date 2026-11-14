"""One CortexBench imitation run on the RM75 port of a MetaWorld task (env rm75-<task>-v2-goal-observable, TASKS):
the official hydra_launcher + train_loop (eai-vc 76fe35e, unchanged on disk) with run_bc.py's runtime patches (embedding
configs from conf/model, metrics.jsonl logging, the logging-only train/* rollouts reusing the evaluation paths) and
run_bc3d.py's point-cloud input for a point-cloud embedding, and with the env constructed differently for the RM75:
RM75Proxy, a gym env that forwards reset / step / seed / frames to rm75_server.py, which runs rm75_env.RM75CBEnv in the
MuJoCo 3 python (MuJoCo 3 + metaworld 3.0; the harness's env has MuJoCo 2.0) given by MW_PY. Everything around it is the
official loop: MuJoCoPixelObsWrapper (its get_image returns the proxy's frame: 256x256 RGB, or the (1200, 6) point cloud
for a point-cloud embedding), FrozenEmbeddingWrapper (FLARE over 3 frames + proprioception obs[:4]), mjrl GymEnv, BC 100
epochs, 20 evaluations of 25 episodes, success = info['success'] at the last of 500 steps. Demos: make_rm75_demos.py.

    source env.sh && MW_PY=<MuJoCo 3 python> $PY run_bc_rm75.py --config-name Metaworld_BC_config \
        env=rm75-bin-picking-v2-goal-observable embedding=lewm_mw_binpick_rm75 seed=100 \
        data_dir=$STABLEWM_HOME/datasets/cortexbench/metaworld-rm75-expert/ wandb.project=local wandb.entity=local \
        hydra.run.dir=<dir>          (3D: embedding=jepa4r_mw_binpick_rm75, data_dir=.../metaworld-rm75-expert-pcd/)
"""

import json
import os
import pickle
import runpy
import struct
import subprocess
import sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
VI = os.path.join(os.environ["JEPA4R_EAI_VC"], "cortexbench", "mujoco_vc", "visual_imitation")  # the eai-vc checkout
CONF_HOME = os.environ.get("JEPA4R_BC_CONF_HOME", HERE)  # <CONF_HOME>/conf/model/NAME.yaml: the embedding configs
sys.path.insert(0, VI)
MW_PY = os.environ.get("MW_PY", "")  # the MuJoCo 3 python, for rm75_server.py

import gym
import hydra
import mujoco_vc.gym_wrapper as gw
import mujoco_vc.model_loading as ml
import mujoco_vc.supported_envs as se
import numpy as np
import train_loop
from gym.spaces import Box
from mjrl.utils.gym_env import GymEnv
from omegaconf import OmegaConf
from PIL import Image

TASKS = {
    "rm75-bin-picking-v2-goal-observable": "bin-picking",
    "rm75-button-press-topdown-v2-goal-observable": "button-press-topdown",
    "rm75-drawer-open-v2-goal-observable": "drawer-open",
    "rm75-assembly-v2-goal-observable": "assembly",
}
for name in TASKS:
    se.ENV_TO_SUITE[name] = "metaworld"  # the same dict gym_wrapper imported
EMB = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("embedding=")), "")
CONF = os.path.join(CONF_HOME, "conf", "model")
MODE = "rgb"
if os.path.exists(os.path.join(CONF, EMB + ".yaml")) and OmegaConf.load(
    os.path.join(CONF, EMB + ".yaml")
)._target_.endswith("_pcd"):
    MODE = "pcd"


def _arr(t):
    dt, shape, b = t
    return np.frombuffer(b, dtype=dt).reshape(shape).copy()


class RM75Proxy(gym.Env):
    """gym (0.23) env backed by rm75_server.py; spec / _get_obs as the harness uses them for metaworld."""

    def __init__(self, env_name):
        self.spec = namedtuple("spec", ["id", "max_episode_steps"])(env_name, 500)
        self.observation_space = Box(-np.inf, np.inf, (39,), np.float64)
        self.action_space = Box(-1.0, 1.0, (4,), np.float32)
        self._p = subprocess.Popen(
            [MW_PY, os.path.join(HERE, "rm75_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(os.path.join(os.getcwd(), "rm75_server.log"), "a"),
            cwd=HERE,
        )
        self._call("init", TASKS[env_name])
        self._obs = None

    def _call(self, cmd, args=None):
        b = pickle.dumps((cmd, args), protocol=4)
        self._p.stdin.write(struct.pack("<Q", len(b)) + b)
        self._p.stdin.flush()
        n = struct.unpack("<Q", self._p.stdout.read(8))[0]
        return pickle.loads(self._p.stdout.read(n))

    def seed(self, seed=None):
        self._call("seed", int(seed or 0))
        return [seed]

    def reset(self):
        self._obs = _arr(self._call("reset"))
        return self._obs

    def step(self, action):
        o, r, d, info = self._call("step", np.asarray(action, dtype=np.float64).tobytes())
        self._obs = _arr(o)
        return self._obs, r, d, info

    def _get_obs(self):  # get_proprioception: env.unwrapped._get_obs()[:4] (also before the first reset)
        if self._obs is None:
            self._obs = _arr(self._call("obs"))
        return self._obs

    def frame(self):
        return _arr(self._call(MODE))

    def close(self):
        try:
            self._call("close")
        except Exception:
            pass


_official_get_image = gw.MuJoCoPixelObsWrapper.get_image


def get_image(self):
    if isinstance(self.env, RM75Proxy):
        return self.env.frame()
    return _official_get_image(self)


gw.MuJoCoPixelObsWrapper.get_image = get_image
_official_constructor = gw.env_constructor


def env_constructor(
    env_name,
    pixel_based=True,
    device="cuda",
    image_width=256,
    image_height=256,
    camera_name=None,
    embedding_name="resnet50",
    history_window=1,
    fuse_embeddings=None,
    render_gpu_id=-1,
    seed=123,
    add_proprio=False,
    *args,
    **kwargs,
):
    if env_name not in TASKS:
        return _official_constructor(
            env_name,
            pixel_based,
            device,
            image_width,
            image_height,
            camera_name,
            embedding_name,
            history_window,
            fuse_embeddings,
            render_gpu_id,
            seed,
            add_proprio,
            *args,
            **kwargs,
        )
    e = RM75Proxy(env_name)  # the official metaworld branch, with the proxy for the metaworld env
    e.seed(seed)
    e = gw.MuJoCoPixelObsWrapper(env=e, width=image_width, height=image_height, camera_name=camera_name, device_id=0)
    e = gw.FrozenEmbeddingWrapper(
        env=e,
        embedding_name=embedding_name,
        suite="metaworld",
        history_window=history_window,
        fuse_embeddings=fuse_embeddings,
        device=device,
        seed=seed,
        add_proprio=add_proprio,
    )
    e = GymEnv(e)
    e.set_seed(seed)
    return e


gw.env_constructor = env_constructor
train_loop.env_constructor = env_constructor

_official_load = ml.load_pretrained_model


def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    """Embeddings configured in ./conf/model (the LeWM encoders); a point-cloud one gets the model's own point-cloud
    preprocessing (run_bc3d.py), an image one the official PIL wrapper."""
    path = os.path.join(CONF, embedding_name + ".yaml")
    if not os.path.exists(path):
        return _official_load(embedding_name, input_type, *args, **kwargs)
    model, dim, transforms, metadata = hydra.utils.call(OmegaConf.load(path))
    model = model.eval()
    if metadata.get("input") == "pcd":
        return model, dim, (lambda pcd: transforms(pcd).unsqueeze(0)), metadata
    return model, dim, (lambda img: transforms(Image.fromarray(img)).unsqueeze(0)), metadata


for mod in (ml, gw, train_loop):
    mod.load_pretrained_model = load_pretrained_model


class JsonlRun:  # run_bc.py's logger: every wandb_run.log record to metrics.jsonl
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


def sample_paths(*args, **kwargs):  # run_bc.py: the logging-only MetaWorld train/* rollouts reuse the evaluation paths
    if kwargs.get("base_seed") == 54321 and "paths" in _last:
        return _last["paths"]
    paths = _sample_paths(*args, **kwargs)
    _last["paths"] = paths
    return paths


train_loop.sample_paths = sample_paths
np.seterr(all="ignore")
if __name__ == "__main__":  # importable (diagnostics) without launching a run
    if not MW_PY:
        sys.exit("set MW_PY to the MuJoCo 3 python (MuJoCo 3 + metaworld 3.0): it runs the RM75 env (rm75_server.py)")
    sys.argv = [os.path.join(VI, "hydra_launcher.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(VI, "hydra_launcher.py"), run_name="__main__")
