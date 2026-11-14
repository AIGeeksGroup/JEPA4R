"""One CortexBench imitation run on a task whose env needs MuJoCo 3, while the official harness runs MuJoCo 2.0
(mujoco-py): the RM75-B twin's MetaWorld tasks (env rm75-<task>-v2-goal-observable) and OGBench Scene's five evaluation
tasks (scene-task{1..5}-v0); the env name selects the suite. The official hydra_launcher + train_loop (eai-vc 76fe35e,
unchanged on disk) with run_bc.py's runtime patches (embedding configs from conf/model, metrics.jsonl logging, the
logging-only train/* rollouts reusing the evaluation paths) and run_bc3d.py's point-cloud input for a point-cloud
embedding; the env is EnvProxy, a gym env that forwards reset / step / seed / frames to env_server.py, which runs
rm75_env.RM75CBEnv or scene_env.SceneCBEnv in the MuJoCo 3 python given by MW_PY. Everything around it is the official
loop: MuJoCoPixelObsWrapper (its get_image returns the proxy's frame, or the (1200, 6) point cloud for a point-cloud
embedding), FrozenEmbeddingWrapper (FLARE over 3 frames + proprioception), mjrl GymEnv, BC 100 epochs, 20 evaluations of
25 episodes, success = info['success'] at the last step.
  RM75-B twin  256x256 RGB from top_cap2, proprioception obs[:4] (the official get_proprioception), 500-step episodes;
               demos: make_rm75_demos.py
  Scene        224x224 RGB from front_pixels, at most 750 steps (scene-v0; an episode ends at the goal, so the last
               step's success is OGBench's success rate); proprioception: env_kwargs.add_proprio=True,
               env_kwargs.proprio_key=gripper_proprio and SCENE_PROPRIO_DIMS=6, so the evaluation input and the demos'
               proprioception both keep the first 6 of scene_env.proprio_of's 8 entries (effector xyz, cos / sin yaw,
               gripper opening: the robot's own state, without the two locks); demos: make_scene_demos.py
Started by the train_policy stage for the twin's MetaWorld tasks and for env=scene (jepa4r.policy.bc.train;
run_command gives its command line).
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
MW_PY = os.environ.get("MW_PY", "")  # the MuJoCo 3 python, for env_server.py

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

# env name -> the server's init argument; episode length, proprioception and action sizes; the server's environment
SUITES = {
    "rm75": dict(
        tasks={
            "rm75-bin-picking-v2-goal-observable": "bin-picking",
            "rm75-button-press-topdown-v2-goal-observable": "button-press-topdown",
            "rm75-drawer-open-v2-goal-observable": "drawer-open",
            "rm75-assembly-v2-goal-observable": "assembly",
        },
        horizon=500,
        obs_dim=39,  # metaworld's _get_obs(); get_proprioception reads its first 4 entries
        act_dim=4,
        drop_cuda_devices=False,
    ),
    "scene": dict(
        tasks={f"scene-task{k}-v0": k for k in range(1, 6)},  # = scene_env.TASKS (not imported: it needs MuJoCo 3)
        horizon=750,
        obs_dim=8,  # scene_env.proprio_of
        act_dim=5,
        drop_cuda_devices=True,
    ),
}
ENV = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("env=")), "")
SUITE = "scene" if ENV.startswith("scene-") else "rm75"
S = SUITES[SUITE]
TASKS, HORIZON, PROPRIO_DIM, ACT_DIM = S["tasks"], S["horizon"], S["obs_dim"], S["act_dim"]
for name in TASKS:
    se.ENV_TO_SUITE[name] = "metaworld"  # success and demo reading as MetaWorld's
EMB = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("embedding=")), "")
CONFS = [os.path.join(CONF_HOME, "conf", "model")]


def conf_path(name):
    for d in CONFS:
        if os.path.exists(os.path.join(d, name + ".yaml")):
            return os.path.join(d, name + ".yaml")
    return None


MODE = "rgb"
if conf_path(EMB) and OmegaConf.load(conf_path(EMB))._target_.endswith("load_jepa4r"):
    MODE = "pcd"


def _arr(t):
    dt, shape, b = t
    return np.frombuffer(b, dtype=dt).reshape(shape).copy()


class EnvProxy(gym.Env):
    """gym (0.23) env backed by env_server.py; spec / _get_obs as the harness uses them for metaworld."""

    def __init__(self, env_name):
        self.spec = namedtuple("spec", ["id", "max_episode_steps"])(env_name, HORIZON)
        self.observation_space = Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float64)
        self.action_space = Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self._p = subprocess.Popen(
            [MW_PY, os.path.join(HERE, "env_server.py"), SUITE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(os.path.join(os.getcwd(), f"{SUITE}_server.log"), "a"),
            cwd=HERE,
            # the Scene server starts without CUDA_VISIBLE_DEVICES (as in the paper's Scene runs), the RM75 one with it
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            if S["drop_cuda_devices"]
            else None,
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

    def _get_obs(self):  # the proprioception the harness reads (also before the first reset)
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
    if isinstance(self.env, EnvProxy):
        return self.env.frame()
    return _official_get_image(self)


gw.MuJoCoPixelObsWrapper.get_image = get_image
_official_proprio = gw.get_proprioception


# Scene: the first entries used; 6 = effector xyz, cos / sin yaw, gripper
PROPRIO_USE = int(os.environ.get("SCENE_PROPRIO_DIMS", PROPRIO_DIM))


def get_proprioception(env, suite):
    # the proxy's obs is the 8 entries; the first PROPRIO_USE (not called without add_proprio)
    if isinstance(env, EnvProxy):
        return env._get_obs()[:PROPRIO_USE]
    return _official_proprio(env, suite)


_official_precompute = train_loop.precompute_features


def precompute_features(paths, *args, **kwargs):
    """The demos' proprioception (env_infos['gripper_proprio'], 8 entries) cut to the first PROPRIO_USE too, as in the
    evaluation."""
    if PROPRIO_USE < PROPRIO_DIM:
        for path in paths:
            path["env_infos"]["gripper_proprio"] = path["env_infos"]["gripper_proprio"][:, :PROPRIO_USE]
    paths = _official_precompute(paths, *args, **kwargs)
    print(
        f"policy input dim {paths[0]['features'].shape[1]} (proprio_key {kwargs.get('proprio_key')}, first "
        f"{PROPRIO_USE} entries)",
        flush=True,
    )
    return paths


if SUITE == "scene":
    gw.get_proprioception = get_proprioception
    train_loop.precompute_features = precompute_features
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
    e = EnvProxy(env_name)  # the official metaworld branch, with the proxy as the env
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
    """An encoder from conf/model; a point-cloud encoder with its own point-cloud preprocessing (run_bc3d.py), an image
    encoder with the official PIL wrapping."""
    path = conf_path(embedding_name)
    if path is None:
        return _official_load(embedding_name, input_type, *args, **kwargs)
    model, dim, transforms, metadata = hydra.utils.call(OmegaConf.load(path))
    model = model.eval()
    if metadata.get("input") == "pcd":
        return model, dim, (lambda pcd: transforms(pcd).unsqueeze(0)), metadata
    return model, dim, (lambda img: transforms(Image.fromarray(img)).unsqueeze(0)), metadata


for mod in (ml, gw, train_loop):
    mod.load_pretrained_model = load_pretrained_model


class JsonlRun:  # run_bc.py's logging: every wandb_run.log record into metrics.jsonl
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


def sample_paths(*args, **kwargs):  # run_bc.py: MetaWorld's logging-only train/* rollouts reuse the evaluation paths
    if kwargs.get("base_seed") == 54321 and "paths" in _last:
        return _last["paths"]
    paths = _sample_paths(*args, **kwargs)
    _last["paths"] = paths
    return paths


train_loop.sample_paths = sample_paths
np.seterr(all="ignore")
if __name__ == "__main__":
    if not MW_PY:
        sys.exit("set MW_PY to the MuJoCo 3 python (MuJoCo 3 with metaworld 3.0 / OGBench): it runs env_server.py")
    sys.argv = [os.path.join(VI, "hydra_launcher.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(VI, "hydra_launcher.py"), run_name="__main__")
