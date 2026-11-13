# Moved from cortexbench@9331494:lewm_pcd/envs/metaworld/env.py (verbatim)
"""MetaWorld tasks of CortexBench as stable-worldmodel envs: three-part data collection (scripts/data/collect_mw.py) and the
planning evaluation (config/eval/mw_<task>_*.yaml). ``MWTaskEnv`` holds what all tasks share, a subclass per task the
rest (``MWBinPickEnv`` here, ``assembly.MWAssemblyEnv``):

* Task: the metaworld 3.0 ``<task>-v3-goal-observable`` env with ``_freeze_rand_vec=False`` as CortexBench's eai-vc gym
  wrapper sets it (a new layout every reset), drawn from the reset seed (``seeded_rand_vec``).
* Camera: CortexBench renders MetaWorld from ``top_cap2``, a camera that only exists in its metaworld fork
  (suraj-nair-1/metaworld@4e896a5, xyz_base.xml: ``mode='targetbody' target='tablelink' pos='0 0.9 0.6'``, default
  fovy). metaworld 3.0 has no such camera, so it is reproduced as a free camera at the same position looking at
  the same body; ``camera_pose()`` returns the pose the renderer actually used (for back-projecting depth).
* Collection plan: every reset draws the episode's sample type (0 backbone, 1 branch, 2 random), its noise level and,
  for a branch, the step the random actions start at and how long they last (to the end = tail, or a burst).
  The task's three-part policy follows the plan; every row records whether its action is random (``rand_action``),
  so the three parts can be told apart in the data. The backbone chains sub-tasks: whenever the policy completes a
  placement it asks the env for the next target (``new_place_target``).
* Evaluation: ``set_state`` / ``set_goal_state`` are the eval.py callables; ``terminated`` = hand and object both
  within tolerance of the goal row's hand and object positions.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box

TOP_CAP2_POS = np.array([0.0, 0.9, 0.6])        # CortexBench fork, xyz_base.xml
BINS = {                                          # where place targets are drawn: the middle of each bin's floor,
    'left': ((-0.17, 0.68), (-0.06, 0.72)),       # inside top_cap2's view (the cube spawns at x -0.21..-0.03)
    'right': ((0.06, 0.68), (0.17, 0.72)),        # the goal bin (bin_goal at x 0.12)
}
HAND_INIT = ((-0.20, 0.55, 0.10), (0.20, 0.80, 0.22))  # bin-picking: random start pose of the hand, kept only if in view
VIEW_MARGIN = 16  # px: "in view" = the hand's projection is this far inside the 224 x 224 image
SAMPLE_TYPES = ('backbone', 'branch', 'random')


class MWTaskEnv(gym.Env):
    """What every MetaWorld task shares. A subclass sets TASK and HAND_INIT and implements _reset_target and
    _task_info (the task's info keys: its target and object state)."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 80}
    TASK: str = ''
    HAND_INIT = HAND_INIT

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
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS

        mw = ENVS[f'{self.TASK}-v3-goal-observable']()
        mw._freeze_rand_vec = False  # CortexBench: a new layout every reset
        mw.seeded_rand_vec = True    # ... drawn from the reset seed, not from the global np.random
        self._mw = mw
        self._model, self._data = mw.model, mw.data
        self.render_mode = render_mode
        self._renderer = mujoco.Renderer(self._model, height, width)
        self._renderer_ss = None  # single-sample renderer for the point cloud, made on first use
        mujoco.mj_forward(self._model, self._data)
        self._cam = self._top_cap2()
        self.render()
        self._view = self.camera_pose() + (height, width)  # fixed camera: pose for in_view()

        self.action_space = Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, mw.observation_space.shape, np.float64)
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

    # ------------------------------------------------------------------ camera
    def _top_cap2(self) -> mujoco.MjvCamera:
        target = self._data.body('tablelink').xpos.copy()
        d = target - TOP_CAP2_POS
        dist = float(np.linalg.norm(d))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
        cam.elevation = float(np.degrees(np.arcsin(d[2] / dist)))
        return cam

    def _single_sample_renderer(self) -> mujoco.Renderer:
        """A second renderer without multisampling, for the point cloud (depth, segmentation and the points' colours), as
        the CortexBench 3D demos (bench/cortexbench/pcd_fork.py): with multisampling an edge pixel's id colour is the
        average of two ids' colours and decodes to an unrelated geom. MuJoCo reads offsamples when a context is made, so
        it is 0 for this one only; the RGB frames keep CortexBench's multisampling."""
        if self._renderer_ss is None:
            q = self._model.vis.quality
            saved, q.offsamples = q.offsamples, 0
            self._renderer_ss = mujoco.Renderer(self._model, *self._view[3:])
            q.offsamples = saved
        return self._renderer_ss

    def render(self, camera=None, depth: bool = False, segmentation: bool = False, multisample: bool = True):
        r = self._renderer if multisample else self._single_sample_renderer()
        if depth:
            r.enable_depth_rendering()
        elif segmentation:
            r.enable_segmentation_rendering()
        r.update_scene(self._data, camera=self._cam if camera in (None, 'top_cap2') else camera)
        img = r.render()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        return img

    def camera_pose(self):
        """Pose of the camera used by the last render: position, rotation (columns = camera x, y, z; the camera looks
        along -z, OpenGL convention as MuJoCo's cam_xmat) and vertical fovy in degrees."""
        s = self._renderer.scene  # scene.camera[0/1] are the stereo eyes (+-ipd/2); a mono render uses their average
        c = mujoco.mjv_averageCamera(s.camera[0], s.camera[1])
        fwd, up = np.array(c.forward, dtype=np.float64), np.array(c.up, dtype=np.float64)
        right = np.cross(fwd, up)
        return np.array(c.pos, dtype=np.float64), np.stack([right, up, -fwd], axis=1), float(self._model.vis.global_.fovy)

    def in_view(self, p, margin: float = VIEW_MARGIN) -> bool:
        """Whether world point p projects at least ``margin`` px inside the top_cap2 image."""
        pos, R, fovy, H, W = self._view
        pc = (np.asarray(p, dtype=np.float64) - pos) @ R
        z = -pc[2]
        if z <= 0:
            return False
        f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
        u, v = pc[0] / z * f + W / 2, -pc[1] / z * f + H / 2
        return margin <= u <= W - margin and margin <= v <= H - margin

    # ------------------------------------------------------------------- plan
    def _draw_plan(self, rng):
        t = int(self.force_type) if self.force_type is not None else int(rng.choice(3, p=self.mix))
        noise = float(rng.uniform(0.0, self.noise_max)) if t < 2 else 0.0
        b0, blen = -1, 0
        if t == 1:
            b0 = int(rng.integers(15, self.episode_len - 40))
            if rng.uniform() < self.branch_tail_prob:
                blen = self.episode_len  # random from b0 to the end of the episode (3DPWM-style tail)
            else:
                blen = int(rng.integers(self.burst_len[0], self.burst_len[1] + 1))  # a burst, then the oracle again
        return t, noise, b0, blen

    def _rand_action(self) -> bool:
        t, _, b0, blen = self._plan
        return t == 2 or (t == 1 and b0 <= self._t < b0 + blen)

    # ------------------------------------------------------------ task hooks
    def _reset_target(self, rng, obs):
        raise NotImplementedError

    def _task_info(self) -> dict:
        raise NotImplementedError

    def new_place_target(self):
        raise NotImplementedError

    # -------------------------------------------------------------- gym API
    def _obj_pos(self):
        return self._mw._get_pos_objects().copy()

    def _info(self, obs):
        d = self._data
        t, noise, b0, blen = self._plan
        info = {
            'qpos': d.qpos.copy(),
            'qvel': d.qvel.copy(),
            'mocap_pos': d.mocap_pos[0].copy(),
            'mocap_quat': d.mocap_quat[0].copy(),
            'hand_pos': np.asarray(obs[:3], dtype=np.float64).copy(),
            'gripper': np.float64(obs[3]),
            'obj_pos': np.asarray(obs[4:7], dtype=np.float64).copy(),
            'obj_quat': np.asarray(obs[7:11], dtype=np.float64).copy(),
        }
        info.update(self._task_info())
        info.update({
            'sample_type': np.int64(t),
            'rand_action': np.int64(self._rand_action()),  # 0 / 1 (bool scalars are dropped by the collector)
            'noise_std': np.float64(noise),
            'branch_t': np.int64(b0),
            'branch_len': np.int64(blen),
        })
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if self.randomize_hand:  # a random start pose of the hand, inside the camera's view
            p = rng.uniform(self.HAND_INIT[0], self.HAND_INIT[1])
            while not self.in_view(p, 2 * VIEW_MARGIN):
                p = rng.uniform(self.HAND_INIT[0], self.HAND_INIT[1])
            self._mw.hand_init_pos = p
        # metaworld's reset() ignores its seed and draws the layout (reset_model) from np_random before reseeding:
        # give it a generator derived from this reset's seed
        self._mw.np_random = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        obs, _ = self._mw.reset()
        self._t = 0
        self._goal = None
        self._plan = self._draw_plan(rng)
        self._reset_target(rng, obs)
        info = self._info(obs)
        info['task_success'] = np.float64(0.0)
        return np.asarray(obs, dtype=np.float64), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, _, _, mw_info = self._mw.step(a)
        self._t += 1
        info = self._info(obs)
        info['task_success'] = np.float64(mw_info['success'])
        terminated = False
        if self._goal is not None:
            hand_ok = np.linalg.norm(info['hand_pos'] - self._goal[0]) < self.goal_tol_hand
            obj_ok = np.linalg.norm(info['obj_pos'] - self._goal[1]) < self.goal_tol_obj
            terminated = bool(hand_ok and obj_ok)
        return np.asarray(obs, dtype=np.float64), float(reward), terminated, False, info

    # ------------------------------------------------------------ eval hooks
    def set_state(self, qpos, qvel, mocap_pos=None, mocap_quat=None):
        d = self._data
        d.qpos[:] = np.asarray(qpos, dtype=np.float64)
        d.qvel[:] = np.asarray(qvel, dtype=np.float64)
        if mocap_pos is not None:  # the hand is welded to the mocap body: restore it too, or the hand snaps back
            d.mocap_pos[0] = np.asarray(mocap_pos, dtype=np.float64)
        if mocap_quat is not None:
            d.mocap_quat[0] = np.asarray(mocap_quat, dtype=np.float64)
        mujoco.mj_forward(self._model, d)
        self._mw._prev_obs = self._mw._get_curr_obs_combined_no_goal()

    def set_goal_state(self, hand_pos, obj_pos):
        self._goal = (np.asarray(hand_pos, dtype=np.float64), np.asarray(obj_pos, dtype=np.float64))

    def close(self):
        self._renderer.close()
        if self._renderer_ss is not None:
            self._renderer_ss.close()
        self._mw.close()


class MWBinPickEnv(MWTaskEnv):
    """bin-picking: pick the cube up from one bin and place it in a bin. Place targets are drawn on the bins' floors
    (the other bin with probability 0.7); info adds ``place_target``."""

    TASK = 'bin-picking'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._place_target = np.zeros(3)

    def _draw_target(self, rng, cube_xy):
        here = 'left' if cube_xy[0] < 0 else 'right'
        other = 'right' if here == 'left' else 'left'
        name = other if rng.uniform() < 0.7 else here
        (x0, y0), (x1, y1) = BINS[name]
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), 0.02])

    def new_place_target(self):
        """Called by the policy when a pick-and-place is done: the next target (another bin with probability 0.7)."""
        self._place_target = self._draw_target(self.np_random, self._obj_pos())

    def _reset_target(self, rng, obs):
        self._place_target = self._draw_target(rng, obs[4:6])

    def _task_info(self):
        return {'place_target': self._place_target.copy()}
