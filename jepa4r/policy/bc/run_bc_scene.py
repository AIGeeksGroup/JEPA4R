"""One CortexBench imitation run on an OGBench scene evaluation task (scene-task{1..5}-v0), built as run_bc_rm75.py: the
official hydra_launcher + train_loop (eai-vc 76fe35e, unchanged on disk) with run_bc.py's runtime patches (embedding
configs from conf/model, metrics.jsonl logging, the logging-only train/* rollouts reusing the evaluation paths) and
run_bc3d.py's point-cloud input for a point-cloud embedding; the env is SceneProxy, a gym env that forwards reset / step
/ seed / frames to scene_server.py, which runs scene_env.SceneCBEnv in the lewm_mw python (MuJoCo 3 + OGBench) given by
MW_PY. Everything around it is the official loop: MuJoCoPixelObsWrapper (its get_image returns the proxy's frame:
224x224 RGB, or the (1200, 6) point cloud for a point-cloud embedding), FrozenEmbeddingWrapper (FLARE over 3 frames [+
proprioception]), mjrl GymEnv, BC 100 epochs, 20 evaluations of 25 episodes, success = info['success'] at the last step
of the episode (it ends at the goal, so this is OGBench's success rate); at most 750 steps (scene-v0). Proprioception:
the source's scene_grid.py --proprio hand6 (the default) passes env_kwargs.add_proprio=True,
env_kwargs.proprio_key=gripper_proprio and SCENE_PROPRIO_DIMS=6, so the evaluation input and the demos' proprioception
both keep the first 6 of scene_env.proprio_of's 8 entries (effector xyz, cos / sin yaw, gripper opening: the robot's own
state, without the two locks); --proprio none passes add_proprio=False, proprio_key=None. Demos: make_scene_demos.py.
With ENC_SOCKET set, the frozen encoder runs in enc_server.py (one GPU process batching for every job, instead of dozens
of jobs each holding a CUDA context and the GPU switching between them); the preprocessing stays in this process with
the same config, BC and the evaluation run on the CPU (device=cpu).

    source env.sh && MW_PY=<lewm_mw python> SCENE_PROPRIO_DIMS=6 $PY run_bc_scene.py \\
        --config-name Metaworld_BC_config env=scene-task1-v0 embedding=lewm_scene_pix seed=100 \\
        data_dir=$STABLEWM_HOME/datasets/cortexbench/scene-expert/ camera=front_pixels \\
        env_kwargs.add_proprio=True env_kwargs.proprio_key=gripper_proprio wandb.project=local wandb.entity=local \\
        hydra.run.dir=<dir>          (3D: embedding=lewm_scene_pcd, data_dir=.../scene-expert-pcd/)
"""

import json
import os
import pickle
import runpy
import socket
import struct
import subprocess
import sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
VI = os.path.join(os.environ["JEPA4R_EAI_VC"], "cortexbench", "mujoco_vc", "visual_imitation")  # the eai-vc checkout
CONF_HOME = os.environ.get("JEPA4R_BC_CONF_HOME", HERE)  # <CONF_HOME>/conf/model/NAME.yaml: the embedding configs
sys.path.insert(0, VI)
MW_PY = os.environ.get("MW_PY", "")  # the lewm_mw python, for scene_server.py

import gym
import hydra
import mujoco_vc.gym_wrapper as gw
import mujoco_vc.model_loading as ml
import mujoco_vc.supported_envs as se
import numpy as np
import torch
import train_loop
from gym.spaces import Box
from mjrl.utils.gym_env import GymEnv
from omegaconf import OmegaConf
from PIL import Image

TASKS = {f"scene-task{k}-v0": k for k in range(1, 6)}  # = scene_env.TASKS (not imported: that side needs MuJoCo 3)
HORIZON, PROPRIO_DIM, ACT_DIM = 750, 8, 5
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
if conf_path(EMB) and OmegaConf.load(conf_path(EMB))._target_.endswith("_pcd"):
    MODE = "pcd"


def _arr(t):
    dt, shape, b = t
    return np.frombuffer(b, dtype=dt).reshape(shape).copy()


class SceneProxy(gym.Env):
    """gym (0.23) env backed by scene_server.py; spec / _get_obs as the harness uses them for metaworld."""

    def __init__(self, env_name):
        self.spec = namedtuple("spec", ["id", "max_episode_steps"])(env_name, HORIZON)
        self.observation_space = Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float64)
        self.action_space = Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self._p = subprocess.Popen(
            [MW_PY, os.path.join(HERE, "scene_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(os.path.join(os.getcwd(), "scene_server.log"), "a"),
            cwd=HERE,
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
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

    def _get_obs(self):
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
    if isinstance(self.env, SceneProxy):
        return self.env.frame()
    return _official_get_image(self)


gw.MuJoCoPixelObsWrapper.get_image = get_image
_official_proprio = gw.get_proprioception


# the first entries used; 6 = effector xyz, cos / sin yaw, gripper
PROPRIO_USE = int(os.environ.get("SCENE_PROPRIO_DIMS", PROPRIO_DIM))


def get_proprioception(env, suite):
    # the proxy's obs is the 8 entries; the first PROPRIO_USE (not called without add_proprio)
    if isinstance(env, SceneProxy):
        return env._get_obs()[:PROPRIO_USE]
    return _official_proprio(env, suite)


gw.get_proprioception = get_proprioception
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
    e = SceneProxy(env_name)  # the official metaworld branch, with the proxy as the env
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
ENC_SOCKET = os.environ.get("ENC_SOCKET", "")


class RemoteEncoder(torch.nn.Module):
    """Stand-in for the frozen encoder: sends the preprocessed input to enc_server.py and returns the same encoder's
    output (no parameters, so .to() does nothing)."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # '@name' = abstract namespace
        self.sock.connect("\0" + ENC_SOCKET[1:] if ENC_SOCKET.startswith("@") else ENC_SOCKET)

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            b = self.sock.recv(n - len(buf))
            if not b:
                raise ConnectionError("enc_server closed the connection")
            buf += b
        return bytes(buf)

    def forward(self, x):
        b = pickle.dumps((self.name, x.detach().cpu().numpy().astype(np.float32)), protocol=4)
        self.sock.sendall(struct.pack("<Q", len(b)) + b)
        out = pickle.loads(self._recv(struct.unpack("<Q", self._recv(8))[0]))
        return torch.from_numpy(out).to(x.device)


def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    """An encoder from conf/model; a point-cloud encoder with its own point-cloud preprocessing (run_bc3d.py), an image
    encoder with the official PIL wrapping."""
    path = conf_path(embedding_name)
    if path is None:
        return _official_load(embedding_name, input_type, *args, **kwargs)
    model, dim, transforms, metadata = hydra.utils.call(OmegaConf.load(path))
    model = RemoteEncoder(embedding_name) if ENC_SOCKET else model.eval()
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
        sys.exit("set MW_PY to the lewm_mw python (MuJoCo 3 + OGBench): it runs the scene env (scene_server.py)")
    sys.argv = [os.path.join(VI, "hydra_launcher.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(VI, "hydra_launcher.py"), run_name="__main__")
