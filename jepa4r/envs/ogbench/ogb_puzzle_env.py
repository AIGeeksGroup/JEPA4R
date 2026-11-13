# Moved from cortexbench@9331494:lewm_pcd/envs/ogb_puzzle_env.py (verbatim)
"""OGBench's own ``puzzle-3x3-v0`` (the ogbench package; stable-worldmodel has no puzzle env) behind the repo's
stable-worldmodel interface, ``swm/OGBenchPuzzle3x3-v0``, for the puzzle world-model data (scripts/data/collect_ogb.py);
the larger boards the same way with ``env_type`` ('4x4': ``swm/OGBenchPuzzle4x4-v0``, 16 buttons; '4x5':
``swm/OGBenchPuzzle4x5-v0``, 20 buttons; OGBench's puzzle-4x4-v0 / puzzle-4x5-v0, everything else as for 3x3; the
3x3 defaults are unchanged).

Collection policy as OGBench's official puzzle-3x3-noisy-v0 (data_gen_scripts/commands.sh and generate_manipspace.py):
  * mode='data_collection': the 9 button colours random at each episode start, random arm;
  * the closed-loop oracle ButtonMarkovOracle(min_norm=0.4, gripper_always_closed=True), held by the env: press a
    random button, move the arm to a random spot, pick the next one (the env gives it a new button after each press);
  * noise level xi ~ U(0, 0.1) drawn per episode, Gaussian action noise N(0, [xi, xi, xi, 3 xi, 10 xi]); each step is a
    uniform random action with probability 0.2 (--p_random_action=0.2; PuzzleNoisyPolicy); actions clipped to [-1, 1];
  * 1001 steps per episode (--max_episode_steps=1001).
All draws as the official script makes them, from numpy's global RNG (seeded per episode): at the episode start the
oracle's reset, then xi (the official script draws xi first; the same distributions), then per step whether the action
is random (drawn by the env when the row starts, so the row can record it) and the action noise. One env per process (num_envs 1): env pools would share the global RNG. The random
actions are OGBench's action_space.sample(): its action_space property builds a new, unseeded Box on every access, so
the random actions (and with them the rows) are not reproducible from the seed, as in the official collection.
The only difference from the official data is the observation:
  * opaque arm (the materials of ob_type='states'; the official visual datasets make the arm translucent, but a
    translucent arm is still whole in the depth image, so the point cloud's colours and shape would disagree);
  * one camera, front_pixels (the official visual datasets' camera), 224x224 (``Camera``):
      2D  pixels: multisampled (anti-aliased) RGB, render();
      3D  pcd_xyz / pcd_rgb: the same camera's depth image back-projected (rendered without multisampling, so edge
          pixels do not decode to wrong geom ids) by envs/ogb_pcd.PuzzlePointCloudWrapper (``point_cloud``), with the
          rules of envs/pcd_wrapper.py: geoms welded to the world (table, walls, button bases, robot base) are
          background; foreground = the moving robot + the buttons' moving parts (the caps, which carry the colour); crop
          box, 5 mm voxels, 1200 points, 25 % of them from the buttons (all visible button points when there are fewer).
Every row also stores the ground truth for probes and evaluation: the 9 button states, effector position / yaw, gripper
opening, qpos / qvel, each button cap's visible pixel count (0 when the arm hides it), whether the step's action is
random, the episode's noise level and the oracle's current button.
This module imports nothing from the repo, so the DQC code (bench/dqc), whose own packages are named envs and utils,
can load it by its path.
"""
from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'egl')
import gymnasium
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401  registers puzzle-3x3-v0
from gymnasium.spaces import Box
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from stable_worldmodel.policy import BasePolicy

ENV_NAME = 'puzzle-3x3-v0'  # the default board; make_env(env_type=...) for the others
CAMERA = 'front_pixels'
IMG = 224
EPISODE_STEPS = 1001
NOISE_MAX = 0.1         # generate_manipspace.py default --noise
P_RANDOM_ACTION = 0.2   # commands.sh's --p_random_action for puzzle-*-noisy
MIN_NORM = 0.4          # generate_manipspace.py default --min_norm
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

# point clouds: the parameters of the repo's OGBench cube / MetaWorld
NUM_POINTS = 1200
VOXEL = 0.005
OBJECT_FRACTION = 0.25
CROP_BOX = np.array([[0.10, -0.50, -0.02], [0.75, 0.50, 0.60]])  # envs/pcd_wrapper.py's DEFAULT_CROP_BOX


def make_env(max_episode_steps=EPISODE_STEPS, env_type='3x3'):
    return gymnasium.make(f'puzzle-{env_type}-v0', ob_type='states', mode='data_collection', terminate_at_goal=False,
                          max_episode_steps=max_episode_steps, width=IMG, height=IMG, visualize_info=False)


class Camera:
    """front_pixels with two renderers: anti-aliased for 2D, without for the point cloud (depth, segmentation, the
    points' colours)."""

    def __init__(self, model: mujoco.MjModel, num_buttons: int = 9):
        self.model = model
        self.cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
        self.fovy = float(model.cam_fovy[self.cid])
        self.r2d = mujoco.Renderer(model, IMG, IMG)
        saved = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0  # read when a renderer is created; affects only the point-cloud one
        try:
            self.rflat = mujoco.Renderer(model, IMG, IMG)
        finally:
            model.vis.quality.offsamples = saved
        f = 0.5 * IMG / np.tan(np.deg2rad(self.fovy) / 2)
        v, u = np.mgrid[0:IMG, 0:IMG]
        self.xn = ((u + 0.5 - IMG / 2) / f).reshape(-1)
        self.yn = (-(v + 0.5 - IMG / 2) / f).reshape(-1)
        # foreground / button geoms
        fg = np.zeros(model.ngeom, bool)
        obj = np.zeros(model.ngeom, bool)
        self.cap_geom = np.array([model.geom(f'btngeom_{i}').id for i in range(num_buttons)])
        for g in range(model.ngeom):
            body = model.geom_bodyid[g]
            if model.body_weldid[body] == 0:  # welded to the world: floor, walls, button bases, robot base
                continue
            root = model.body_rootid[body]
            if model.body_mocapid[root] >= 0:
                continue
            fg[g] = True
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ''
            obj[g] = name.startswith('buttonbox_')
        self.fg, self.obj = fg, obj

    def close(self):
        self.r2d.close()
        self.rflat.close()

    def render(self, data: mujoco.MjData):
        """The 2D frame and the point cloud's renders: (pixels, rgb, depth, seg, cap_px, R, t)."""
        self.r2d.update_scene(data, camera=CAMERA)
        pixels = self.r2d.render().copy()
        return (pixels,) + self.render_flat(data)

    def render_flat(self, data: mujoco.MjData):
        """What the point cloud needs (colour without anti-aliasing, depth, segmentation), the caps' visible pixels
        and the camera pose: (rgb, depth, seg, cap_px, R, t)."""
        r = self.rflat
        r.update_scene(data, camera=CAMERA)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=CAMERA)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=CAMERA)
        seg2 = r.render().copy()
        r.disable_segmentation_rendering()
        seg = np.where(seg2[..., 1] == GEOM, seg2[..., 0], -1).reshape(-1)
        cap_px = np.array([(seg == g).sum() for g in self.cap_geom], dtype=np.int32)
        R = data.cam_xmat[self.cid].reshape(3, 3).copy()
        t = data.cam_xpos[self.cid].copy()
        return rgb, depth, seg, cap_px, R, t


def point_cloud(cam: Camera, rgb, depth, seg, R, t, rng: np.random.Generator):
    """Back-project the depth -> foreground -> crop -> voxel dedup -> stratified sample (buttons 25 %) -> shuffle. As
    envs/pcd_wrapper.py's _sample."""
    valid = seg >= 0
    valid &= cam.fg[np.where(valid, seg, 0)]
    idx = np.flatnonzero(valid)
    n = NUM_POINTS
    if idx.size == 0:
        return np.zeros((n, 3), np.float32), np.zeros((n, 3), np.uint8)
    z = depth.reshape(-1)[idx]
    p_cam = np.stack([cam.xn[idx] * z, cam.yn[idx] * z, -z], axis=1)
    pts = p_cam @ R.T + t
    cols = rgb.reshape(-1, 3)[idx]
    is_obj = cam.obj[seg[idx]]
    m = np.all((pts >= CROP_BOX[0]) & (pts <= CROP_BOX[1]), axis=1)
    pts, cols, is_obj = pts[m], cols[m], is_obj[m]
    if len(pts):
        q = np.floor(pts / VOXEL).astype(np.int32)
        _, keep = np.unique(q, axis=0, return_index=True)
        pts, cols, is_obj = pts[keep], cols[keep], is_obj[keep]
    if len(pts) == 0:
        return np.zeros((n, 3), np.float32), np.zeros((n, 3), np.uint8)
    obj_idx, rob_idx = np.flatnonzero(is_obj), np.flatnonzero(~is_obj)
    n_obj = min(int(round(OBJECT_FRACTION * n)), obj_idx.size)
    if rob_idx.size == 0:
        sel = rng.choice(obj_idx, n, replace=obj_idx.size < n)
    else:
        n_rob = n - n_obj
        sel = np.concatenate([rng.choice(obj_idx, n_obj, replace=False),
                              rng.choice(rob_idx, n_rob, replace=rob_idx.size < n_rob)])
    sel = rng.permutation(sel)
    return pts[sel].astype(np.float32), cols[sel].astype(np.uint8)


# -------------------------------------------------------------------------------------------------- env
class OGBPuzzleEnv(gymnasium.Env):
    """swm/OGBenchPuzzle3x3-v0 (env_type '4x4' / '4x5': swm/OGBenchPuzzle4x4-v0 / 4x5-v0), see the module docstring. The OGBench env is ``_env`` (gymnasium.make'd, as OGBench's
    own scripts use it; its unwrapped PuzzleEnv is ``_u``); the oracle (``agent``) acts on ``ob`` / ``ob_info``, the
    latest observation and info of OGBench's reset / step."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, episode_len: int = EPISODE_STEPS, render_mode: str = 'rgb_array', env_type: str = '3x3', **kwargs):
        self.render_mode = render_mode
        self.env_type = env_type
        self._env = make_env(max_episode_steps=episode_len, env_type=env_type)
        self._u = self._env.unwrapped
        self.action_space = Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, self._env.observation_space.shape, np.float32)
        self.cam = None
        self.agent = None
        self.xi = 0.0
        self.rand = False
        self.ob, self.ob_info = None, None

    def _row_info(self) -> dict:
        u, oi = self._u, self._u.compute_ob_info()
        return {
            'button_states': u._cur_button_states.astype(np.int8).copy(),
            'effector_pos': np.asarray(oi['proprio/effector_pos'], np.float32),
            'effector_yaw': np.asarray(oi['proprio/effector_yaw'], np.float32).reshape(1),
            'gripper_opening': np.asarray(oi['proprio/gripper_opening'], np.float32).reshape(1),
            'qpos': u._data.qpos.astype(np.float32).copy(),
            'qvel': u._data.qvel.astype(np.float32).copy(),
            'rand_action': np.int8(self.rand),
            'noise_xi': np.float32(self.xi),
            'target_button': np.int8(self.ob_info['privileged/target_button']),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # this env's own np_random (stable-worldmodel draws its episode ids from it); unused here
        np.random.seed(seed)  # the oracle (ButtonMarkovOracle.reset's end pose) uses the global np.random, as the official script
        self.ob, self.ob_info = self._env.reset(seed=seed)
        if self.cam is None or self.cam.model is not self._u._model:  # OGBench builds the model at reset
            if self.cam is not None:
                self.cam.close()
            self.cam = Camera(self._u._model, self._u._num_buttons)
        self.agent = ButtonMarkovOracle(env=self._env, min_norm=MIN_NORM, gripper_always_closed=True)
        self.agent.reset(self.ob, self.ob_info)
        self.xi = np.random.uniform(0, NOISE_MAX)  # the episode's noise level (official: drawn once per episode)
        self.rand = np.random.rand() < P_RANDOM_ACTION  # whether the first action is random (official: drawn per step)
        return np.asarray(self.ob, np.float32), self._row_info()

    def step(self, action):
        self.ob, reward, terminated, truncated, self.ob_info = self._env.step(action)
        if self.agent.done:  # the button is pressed: a new random button (as the official script; ob / ob_info stay the step's)
            agent_ob, agent_info = self._u.set_new_target()
            self.agent.reset(agent_ob, agent_info)
        self.rand = np.random.rand() < P_RANDOM_ACTION  # whether the next action is random
        return np.asarray(self.ob, np.float32), float(reward), terminated, truncated, self._row_info()

    def render(self):
        self.cam.r2d.update_scene(self._u._data, camera=CAMERA)
        return self.cam.r2d.render().copy()

    def close(self):
        if self.cam is not None:
            self.cam.close()
        self._env.close()


# ----------------------------------------------------------------------------------------------- policy
class PuzzleNoisyPolicy(BasePolicy):
    """The official noisy collection's action rule for OGBPuzzleEnv: a random row takes OGBench's
    action_space.sample(); otherwise the env's oracle plus N(0, [xi, xi, xi, 3 xi, 10 xi]) from numpy's global RNG."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'noisy_oracle'

    def get_action(self, info_dict, **kwargs):
        envs = [e.unwrapped for e in self.env.envs]
        actions = np.zeros((len(envs), 5), dtype=np.float32)
        for i, env in enumerate(envs):
            if env.rand:
                a = env._env.action_space.sample()
            else:
                a = np.array(env.agent.select_action(env.ob, env.ob_info))
                xi = env.xi
                a = a + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], a.shape)
            actions[i] = np.clip(a, -1, 1).astype(np.float32)
        return actions
