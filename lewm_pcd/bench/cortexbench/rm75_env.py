"""CortexBench's MetaWorld env on the RM75 (runs in the lewm_mw env, MuJoCo 3): what the official harness's
env_constructor + MuJoCoPixelObsWrapper give for the Sawyer (metaworld <task>-v2-goal-observable with
_freeze_rand_vec False, a new layout per seed, the hand at the task's own start, 500-step episodes, 256x256 frames
from top_cap2 with multisampling), here with the RM75 env of envs/metaworld/rm75:
* reset(): MWTaskEnv.reset with the seed set by seed(): the layout is drawn from that seed (a different layout than
  metaworld's own seed(s) + reset() gives for the same s), the hand at the task's own start;
* the steps are the RM75 metaworld env's (envs/metaworld/rm75/robot.py: clipped 4-D action, IK, servos), so step(a)
  returns metaworld's own (obs, reward, info) with info['success'];
* rgb(): 256x256 top_cap2, multisampled (the renderer the 2D LeWM data used, at CortexBench's size);
* pcd(): (1200, 6) float32 xyz (m) + rgb 0..255 of the same camera, envs.metaworld.MWPointCloudWrapper with the 3D
  LeWM data's settings (1200 points, 5 mm voxels, 25 % cube points; depth / segmentation without multisampling),
  rendered at 256x256 as the Sawyer harness's 3D (pcd_fork.py). Its sampling RNG is seeded by seed().
Used by make_rm75_demos.py (the demos) and rm75_server.py (the evaluation env behind run_bc_rm75.py's proxy)."""
from __future__ import annotations

import os
import sys

LEWM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # lewm_pcd
sys.path.insert(0, LEWM)
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np  # noqa: E402

from envs.metaworld import MWPointCloudWrapper  # noqa: E402
from envs.metaworld.rm75 import RM75BinPickEnv  # noqa: E402

TASKS = {'bin-picking': RM75BinPickEnv}
SIZE = 256
HORIZON = 500


class RM75CBEnv:
    def __init__(self, task: str):
        env = TASKS[task](width=SIZE, height=SIZE, randomize_hand=False, episode_len=HORIZON, force_type=0)
        self.pcw = MWPointCloudWrapper(env, num_points=1200, voxel_size=0.005, cube_fraction=0.25)
        self.u = env
        self._seed = 0
        self.obs = None

    def seed(self, s):
        self._seed = int(s)

    def reset(self):
        obs, info = self.pcw.reset(seed=self._seed)  # MWTaskEnv.reset: layout from the seed, the task's own hand start
        self.obs = np.asarray(obs, dtype=np.float64)
        return self.obs

    def step(self, a):
        obs, reward, _, _, info = self.u._mw.step(np.asarray(a, dtype=np.float64))  # metaworld's own step and info
        self.obs = np.asarray(obs, dtype=np.float64)
        return self.obs, float(reward), False, {k: float(v) for k, v in info.items()}

    def rgb(self):
        return self.u.render()

    def pcd(self):
        p = self.pcw._point_cloud()
        return np.concatenate([p['pcd_xyz'], p['pcd_rgb'].astype(np.float32)], axis=1).astype(np.float32)

    def close(self):
        self.u.close()
