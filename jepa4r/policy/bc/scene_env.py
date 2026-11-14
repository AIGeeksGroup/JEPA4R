"""OGBench scene's five evaluation tasks as an env of CortexBench's (eai-vc) imitation harness (runs in the lewm_mw env,
MuJoCo 3 + OGBench); built as rm75_env.py: the harness (MuJoCo 2.0) reaches it through run_bc_scene.py's proxy and the
scene_server.py subprocess, and the env is the repo's (envs.ogb_scene_env.OGBSceneEnv, mode='task') with its point-cloud
wrapper (envs.ogb_pcd.ScenePointCloudWrapper).
* reset(): OGBench's task-mode reset (the task_id fixed) with the seed set by seed(): random arm, the cube / drawer /
  window within +-1 cm of the task's initial state;
* step(a): OGBench's step; the episode ends at the goal (done = info['success'], OGBench's default terminate_at_goal),
  so the harness's "success at the last step" is OGBench's success rate; at most 750 steps (scene-v0's
  max_episode_steps);
* rgb(): front_pixels 224x224, anti-aliased (OGBSceneEnv.render: the renderer and the size of the 2D LeWM data);
* pcd(): (1200, 6) float32 xyz (m) + rgb 0..255, the 3D LeWM data's recipe (ScenePointCloudWrapper's point cloud), its
  sampling RNG seeded with the seed + 7919 at every reset; a cloud is drawn only when the harness asks for a frame;
* proprioception (proprio_of): [effector xyz, cos yaw, sin yaw, gripper opening, button 0 (drawer lock), button 1
  (window lock)], 8 entries; run_bc_scene.py feeds the policy the first SCENE_PROPRIO_DIMS of them (the source's
  scene_grid.py --proprio: hand6 = the first 6, the robot's own state; none = no proprioception).
Used by make_scene_demos.py (the demos) and scene_server.py (the evaluation env)."""

from __future__ import annotations

import os
import sys

# the repository (jepa4r/)
LEWM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, LEWM)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from jepa4r.envs.ogbench.ogb_pcd import SEED_OFFSET, ScenePointCloudWrapper
from jepa4r.envs.ogbench.ogb_scene_env import OGBSceneEnv
from jepa4r.envs.ogbench.ogb_scene_expert import HORIZON

TASKS = {f"scene-task{k}-v0": k for k in range(1, 6)}


def proprio_of(u) -> np.ndarray:
    oi = u.compute_ob_info()
    yaw = float(oi["proprio/effector_yaw"][0])
    return np.concatenate(
        [
            oi["proprio/effector_pos"],
            [np.cos(yaw), np.sin(yaw)],
            oi["proprio/gripper_opening"],
            u._cur_button_states.astype(np.float64),
        ]
    ).astype(np.float64)


class SceneCBEnv:
    def __init__(self, task_id: int):
        self.task_id = int(task_id)
        self.genv = OGBSceneEnv(mode="task", episode_len=HORIZON)
        self.pcw = ScenePointCloudWrapper(self.genv)
        self.env = self.genv._env  # OGBench's gymnasium env (the expert's)
        self.u = self.genv._u
        self._seed = 0
        self.obs = None
        # OGBench builds the model at the first reset; the harness reads proprioception before its first reset
        self.reset()

    def seed(self, s):
        self._seed = int(s)

    def reset(self):
        s = self._seed
        self.genv.reset(seed=s, options=dict(task_id=self.task_id))
        # no cloud at reset: drawn when the harness asks for a frame
        self.pcw._rng = np.random.default_rng(s + SEED_OFFSET)
        self.obs = proprio_of(self.u)
        return self.obs

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        _, reward, _, _, info = self.genv.step(a)
        self.obs = proprio_of(self.u)
        success = float(bool(info["success"]))
        return self.obs, float(reward), bool(success), {"success": success}

    def rgb(self):
        return self.genv.render()

    def pcd(self):
        p = self.pcw._point_cloud()
        return np.concatenate([p["pcd_xyz"], p["pcd_rgb"].astype(np.float32)], axis=1).astype(np.float32)

    def close(self):
        self.genv.close()
