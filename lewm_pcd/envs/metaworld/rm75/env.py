"""envs.metaworld's MetaWorld task envs (top_cap2 camera, three-part collection plan, eval hooks) on the RM75 metaworld
env (robot.py) instead of the Sawyer one. Only what depends on the robot changes:

* the metaworld env is robot.make(task);
* rows record the RM75 controller's state instead of the Sawyer's mocap: ``ee_target`` (the hand target the actions
  move) and ``q_cmd`` (the 7 joint targets of the servos);
* set_state restores qpos / qvel and that controller state (the Sawyer version restores the mocap body).
"""

from __future__ import annotations

import mujoco
import numpy as np
from gymnasium.spaces import Box

from ..env import MWBinPickEnv, MWTaskEnv
from .robot import make


class RM75TaskEnv(MWTaskEnv):
    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        episode_len: int = 250,
        mix=(0.5, 0.3, 0.2),
        force_type: int | None = None,
        noise_max: float = 0.2,
        branch_tail_prob: float = 0.5,
        burst_len=(10, 30),
        randomize_hand: bool = True,
        goal_tol_hand: float = 0.04,
        goal_tol_obj: float = 0.03,
        render_mode: str = 'rgb_array',
    ):
        # MWTaskEnv.__init__ with the RM75 metaworld env (make: goal-observable, _freeze_rand_vec False, seeded_rand_vec)
        mw = make(self.TASK)
        self._mw = mw
        self._model, self._data = mw.model, mw.data
        self.render_mode = render_mode
        self._renderer = mujoco.Renderer(self._model, height, width)
        self._renderer_ss = None
        mujoco.mj_forward(self._model, self._data)
        self._cam = self._top_cap2()
        self.render()
        self._view = self.camera_pose() + (height, width)

        self.action_space = Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (39,), np.float64)
        self.episode_len = int(episode_len)
        self.mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
        self.force_type = force_type
        self.noise_max = float(noise_max)
        self.branch_tail_prob = float(branch_tail_prob)
        self.burst_len = tuple(int(b) for b in burst_len)
        self.randomize_hand = randomize_hand
        self.goal_tol_hand, self.goal_tol_obj = float(goal_tol_hand), float(goal_tol_obj)
        self._goal = None
        self._t = 0
        self._plan = (0, 0.0, -1, 0)

    def _info(self, obs):
        d, mw = self._data, self._mw
        t, noise, b0, blen = self._plan
        info = {
            'qpos': d.qpos.copy(),
            'qvel': d.qvel.copy(),
            'ee_target': np.asarray(mw._ee_target, dtype=np.float64).copy(),
            'q_cmd': np.asarray(mw._q_ik, dtype=np.float64).copy(),
            'hand_pos': np.asarray(obs[:3], dtype=np.float64).copy(),
            'gripper': np.float64(obs[3]),
            'obj_pos': np.asarray(obs[4:7], dtype=np.float64).copy(),
            'obj_quat': np.asarray(obs[7:11], dtype=np.float64).copy(),
        }
        info.update(self._task_info())
        info.update({
            'sample_type': np.int64(t),
            'rand_action': np.int64(self._rand_action()),
            'noise_std': np.float64(noise),
            'branch_t': np.int64(b0),
            'branch_len': np.int64(blen),
        })
        return info

    def set_state(self, qpos, qvel, ee_target=None, q_cmd=None):
        d, mw = self._data, self._mw
        d.qpos[:] = np.asarray(qpos, dtype=np.float64)
        d.qvel[:] = np.asarray(qvel, dtype=np.float64)
        mujoco.mj_forward(self._model, d)
        mw._q_ik = (np.asarray(q_cmd, dtype=np.float64) if q_cmd is not None else d.qpos[mw._arm_q]).copy()
        mw._ee_target = (np.asarray(ee_target, dtype=np.float64) if ee_target is not None
                         else d.site_xpos[mw._hand_site]).copy()
        d.ctrl[mw._arm_act] = mw._q_ik  # the servos hold the recorded command, no jump at the next step
        mw._prev_obs = mw._get_curr_obs_combined_no_goal()


class RM75BinPickEnv(MWBinPickEnv, RM75TaskEnv):
    """MWBinPickEnv (place targets on the bins' floors) on the RM75: MRO MWBinPickEnv -> RM75TaskEnv -> MWTaskEnv."""
