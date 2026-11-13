# Moved from cortexbench@9331494:lewm_pcd/bench/cortexbench/rm75_env.py (verbatim except: imports / paths adapted to the package)
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
  LeWM data's settings (1200 points, 5 mm voxels, 25 % object points, the task's re-placed fixtures; depth /
  segmentation without multisampling), rendered at 224x224 as the 3D LeWM data and the Sawyer harness's 3D
  (pcd_fork.MWPointClouds): the env is made at 224, the frames come from a 256 renderer of their own. Its sampling RNG
  is seeded by seed().
Used by make_rm75_demos.py (the demos) and rm75_server.py (the evaluation env behind run_bc_rm75.py's proxy)."""
from __future__ import annotations

import os
import sys

LEWM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # the repository (jepa4r/)
sys.path.insert(0, LEWM)
os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from jepa4r.envs.metaworld import MWPointCloudWrapper  # noqa: E402
from jepa4r.envs.twins.metaworld_rm75 import RM75AssemblyEnv, RM75BinPickEnv, RM75ButtonPressEnv, RM75DrawerEnv  # noqa: E402

# task -> (env, fixtures: bodies a reset re-places, in the point-cloud foreground as in the task's collection config)
TASKS = {'bin-picking': (RM75BinPickEnv, ()), 'button-press-topdown': (RM75ButtonPressEnv, ('box',)),
         'drawer-open': (RM75DrawerEnv, ('drawer',)), 'assembly': (RM75AssemblyEnv, ('peg',))}
SIZE = 256      # CortexBench's frames
PCD_SIZE = 224  # the point cloud's depth camera: the 3D LeWM data's (and pcd_fork.MWPointClouds')
HORIZON = 500


class RM75CBEnv:
    def __init__(self, task: str):
        cls, fixtures = TASKS[task]
        env = cls(width=PCD_SIZE, height=PCD_SIZE, randomize_hand=False, episode_len=HORIZON, force_type=0)
        self.pcw = MWPointCloudWrapper(env, num_points=1200, voxel_size=0.005, cube_fraction=0.25, fixtures=fixtures)
        self.u = env
        self._rgb = mujoco.Renderer(env._model, SIZE, SIZE)  # multisampled, as MWTaskEnv.render's renderer
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
        self._rgb.update_scene(self.u._data, camera=self.u._cam)  # MWTaskEnv.render at SIZE
        return self._rgb.render()

    def pcd(self):
        p = self.pcw._point_cloud()
        return np.concatenate([p['pcd_xyz'], p['pcd_rgb'].astype(np.float32)], axis=1).astype(np.float32)

    def close(self):
        self._rgb.close()
        self.u.close()
