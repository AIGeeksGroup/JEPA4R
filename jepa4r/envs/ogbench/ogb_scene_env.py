# Moved from cortexbench@9331494:lewm_pcd/envs/ogb_scene_env.py (verbatim)
"""OGBench's own ``scene-v0`` (the ogbench package, not stable-worldmodel's ``swm/OGBScene-v0`` port) behind the repo's
stable-worldmodel interface, ``swm/OGBenchScene-v0``, for the scene world-model data (scripts/data/collect_ogb.py) and
the CortexBench evaluation tasks (bench/cortexbench/scene_env.py).

Collection (``mode='data_collection'``): one object task per env (``task``: cube, button, drawer or window), each
collected on its own with MetaWorld's three-part plan (envs/metaworld/env.py, policy.py). At every reset(seed):
  * OGBench's reset (random arm, the cube at a random spot on the table, the two buttons (locks) each random, drawer and
    window openings uniform); the hand start is kept only inside the camera image (as MetaWorld);
  * the current task's object keeps OGBench's random initial state; every other task is set to its "done" or "not done"
    look, 1/2 each: drawer open (-0.16) / closed (0), window open (0.2) / closed (0), the locks as OGBench drew them,
    the cube on the table (OGBench's spot) / in the drawer (a random spot inside; out of sight when the drawer is closed);
  * the episode's plan as MetaWorld's: backbone 0.5 / branch 0.3 / random 0.2; backbone and branch get an action-noise
    std ~ U(0, 0.2) on all 5 action entries; a branch starts uniform random actions at a step drawn from [15, L - 40):
    half of them to the episode end, half for U[10, 30] steps and then back to the backbone.
The backbone is OGBench's closed-loop oracle of the task (the Markov ones, OGBench's scene collection parameters:
min_norm 0.4, cube max_step 100), held by the env; after each goal the env gives it a new goal of the same task (the
branch of OGBench's set_new_target for that task: the cube to a random table spot, into the drawer with probability 0.3
when the drawer is open; a random button toggled; drawer / window: open -> closed, closed -> open). Random actions
(SceneThreePartPolicy): uniform in [-1, 1]^5; when three more steps of the action would take the hand out of the image,
the outward components are flipped (as MetaWorld). 200 steps per episode (MetaWorld's length). OGBench's health check
(``healthy``, applied by the collection script): an episode whose cube goes too far right (y >= 0.29) or too far left
outside the drawer (y <= -0.3 off the drawer's height) is collected again with another seed.
The seed fixes everything: numpy's global RNG (OGBench's oracles draw from it on reset), OGBench's own reset, the
episode RNG (``rng``: the other tasks' looks, the plan, the policy's noise and random actions) and the point-cloud
sampling (envs/ogb_pcd.py). OGBench's oracles draw from numpy's global RNG, so one env per process (num_envs 1).
Every row records the state before its action: the plan (sample_type, rand_action, noise_std, branch_t, branch_len,
oracle_restart = the backbone restarts from the current state after a random burst), the episode seed, the two locks,
drawer / window openings, the cube (position, in the drawer), the effector (position, yaw), the gripper, qpos / qvel and
whether the current goal is reached (target_success).

Evaluation (``mode='task'``): OGBench's task mode, reset(seed, options={'task_id': k}) and step as OGBench's.

Observations: one camera, front_pixels, 224 x 224, the opaque arm of ob_type='states' (``Camera``): render() is the
anti-aliased RGB of the 2D data; envs/ogb_pcd.ScenePointCloudWrapper back-projects the same camera's depth (rendered
without anti-aliasing) into the 3D data's point cloud (``point_cloud``): geoms welded to the world (table, walls, button
bases, drawer cabinet, window frame and the fixed pane, robot base) are background; foreground = the moving arm + cube +
the two button caps + the drawer's moving part + the sliding pane; the window glass (alpha 0.1, nearly invisible in the
image) is left out. Crop box, 5 mm voxels, 1200 points; 25 % for the objects, split evenly over the 5 objects (an
object short of points leaves its share to the others; all object points when they are short together), the rest from
the arm.
"""
from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'egl')
import gymnasium
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401  registers scene-v0
from gymnasium.spaces import Box
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from stable_worldmodel.policy import BasePolicy

ENV_NAME = 'scene-v0'
CAMERA = 'front_pixels'
IMG = 224
TASKS = ('cube', 'button', 'drawer', 'window')

# MetaWorld's three-part plan (envs/metaworld/env.py defaults, scripts/data/config/mw_binpick.yaml)
EPISODE_STEPS = 200
MIX = (0.5, 0.3, 0.2)
NOISE_MAX = 0.2
BRANCH_TAIL_PROB = 0.5
BURST_LEN = (10, 30)
VIEW_MARGIN = 16          # px
LOOKAHEAD = 3             # random actions: flipped when 3 more steps would leave the image
ACTION_SCALE = np.array([0.05, 0.05, 0.05])  # OGBench: action 1 = the effector target moves 5 cm
MIN_NORM = 0.4            # OGBench generate_manipspace.py default

# OGBench scene geometry (scene_env.py / drawer.xml)
DRAWER_OPEN, DRAWER_CLOSED = -0.16, 0.0
WINDOW_OPEN, WINDOW_CLOSED = 0.2, 0.0
P_PUT_IN_DRAWER = 0.3
CUBE_Z_IN_DRAWER = 0.066  # cube centre height on the drawer floor (OGBench task4 / task5 goal z)

# point clouds: as the repo's OGBench cube / puzzle
NUM_POINTS = 1200
VOXEL = 0.005
OBJECT_FRACTION = 0.25
CROP_BOX = np.array([[0.10, -0.55, -0.02], [0.75, 0.50, 0.60]])  # puzzle's box, y floor at -0.55 to hold the closed drawer
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
OBJ_NAMES = ('cube', 'button0', 'button1', 'drawer', 'window')
ROOT2OBJ = {'object_0': 1, 'buttonbox_0': 2, 'buttonbox_1': 3, 'drawer_base': 4, 'window': 5}


def make_env(max_episode_steps=EPISODE_STEPS, mode='data_collection'):
    """OGBench's scene-v0 as the collection (data_collection) or the evaluation (task) uses it; not ending at the goal."""
    return gymnasium.make(ENV_NAME, ob_type='states', mode=mode, terminate_at_goal=False,
                          max_episode_steps=max_episode_steps, width=IMG, height=IMG, visualize_info=False)


class Camera:
    """front_pixels with two renderers: anti-aliased for 2D, without for the point cloud (depth, segmentation, the
    points' colours); geom labels for the point cloud (0 background, -1 arm, 1..5 = OBJ_NAMES)."""

    def __init__(self, model: mujoco.MjModel):
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
        self.f = 0.5 * IMG / np.tan(np.deg2rad(self.fovy) / 2)
        v, u = np.mgrid[0:IMG, 0:IMG]
        self.xn = ((u + 0.5 - IMG / 2) / self.f).reshape(-1)
        self.yn = (-(v + 0.5 - IMG / 2) / self.f).reshape(-1)
        glass = model.material('window_glass').id
        obj = np.zeros(model.ngeom, np.int8)
        for g in range(model.ngeom):
            body = model.geom_bodyid[g]
            if model.body_weldid[body] == 0:  # welded to the world
                continue
            root = model.body_rootid[body]
            if model.body_mocapid[root] >= 0:  # the cube target (mocap, transparent in the image)
                continue
            if model.geom_matid[g] == glass:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ''
            obj[g] = ROOT2OBJ.get(name, -1)
        self.obj = obj
        self.cube_geoms = np.flatnonzero(obj == 1)
        self.cap_geoms = [np.flatnonzero(obj == 2), np.flatnonzero(obj == 3)]

    def close(self):
        self.r2d.close()
        self.rflat.close()

    def in_view(self, data: mujoco.MjData, p, margin=VIEW_MARGIN) -> bool:
        R = data.cam_xmat[self.cid].reshape(3, 3)
        pc = (np.asarray(p, np.float64) - data.cam_xpos[self.cid]) @ R
        z = -pc[2]
        if z <= 0:
            return False
        u, v = pc[0] / z * self.f + IMG / 2, -pc[1] / z * self.f + IMG / 2
        return margin <= u <= IMG - margin and margin <= v <= IMG - margin

    def render(self, data: mujoco.MjData):
        """The anti-aliased 2D frame."""
        self.r2d.update_scene(data, camera=CAMERA)
        return self.r2d.render().copy()

    def render_flat(self, data: mujoco.MjData):
        """What the point cloud needs (colour without anti-aliasing, depth, segmentation) and the camera pose."""
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
        R = data.cam_xmat[self.cid].reshape(3, 3).copy()
        t = data.cam_xpos[self.cid].copy()
        return rgb, depth, seg, R, t

    def visible_px(self, seg):
        """Pixels of the cube and of each button cap visible in the image."""
        return np.array([np.isin(seg, self.cube_geoms).sum()] + [np.isin(seg, c).sum() for c in self.cap_geoms], np.int32)


def point_cloud(cam: Camera, rgb, depth, seg, R, t, rng: np.random.Generator, return_labels=False):
    """Back-project the depth -> foreground -> crop -> voxel dedup -> stratified sample (objects 25 %, split over the 5
    objects) -> shuffle."""
    n = NUM_POINTS
    lab = np.where(seg >= 0, cam.obj[np.where(seg >= 0, seg, 0)], 0)
    idx = np.flatnonzero(lab != 0)
    empty = np.zeros((n, 3), np.float32), np.zeros((n, 3), np.uint8)
    if idx.size == 0:
        return empty
    z = depth.reshape(-1)[idx]
    pts = np.stack([cam.xn[idx] * z, cam.yn[idx] * z, -z], axis=1) @ R.T + t
    cols = rgb.reshape(-1, 3)[idx]
    lab = lab[idx]
    m = np.all((pts >= CROP_BOX[0]) & (pts <= CROP_BOX[1]), axis=1)
    pts, cols, lab = pts[m], cols[m], lab[m]
    if len(pts):
        # voxel dedup: the three voxel coordinates packed into one int64 (21 bits per axis, offset to be non-negative);
        # exactly np.unique(axis=0) on the rows (the same set, the same lexicographic order, the same first-occurrence
        # indices of the stable sort), only much faster
        q = np.floor(pts / VOXEL).astype(np.int64) + (1 << 20)
        _, keep = np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2], return_index=True)
        pts, cols, lab = pts[keep], cols[keep], lab[keep]
    if len(pts) == 0:
        return empty
    groups = [np.flatnonzero(lab == k) for k in range(1, len(OBJ_NAMES) + 1)]
    rob = np.flatnonzero(lab < 0)
    # the objects' 25 %: split evenly over the 5 objects; an object short of points gives all it has, and the rest of
    # the budget keeps being split over the objects that still have points
    budget = min(int(round(OBJECT_FRACTION * n)), sum(len(g) for g in groups))
    take = np.zeros(len(groups), int)
    while budget > 0:
        open_ = [k for k, g in enumerate(groups) if take[k] < len(g)]
        share = max(budget // len(open_), 1)
        for k in open_:
            add = min(share, len(groups[k]) - take[k], budget)
            take[k] += add
            budget -= add
            if budget == 0:
                break
    sel = [rng.choice(g, k, replace=False) for g, k in zip(groups, take) if k > 0]
    n_rob = n - int(take.sum())
    if rob.size:
        sel.append(rng.choice(rob, n_rob, replace=rob.size < n_rob))
    else:
        allobj = np.concatenate(groups)
        sel.append(rng.choice(allobj, n_rob, replace=True))
    sel = rng.permutation(np.concatenate(sel))
    if return_labels:  # for checks: the object of each point (-1 arm, 1..5 = OBJ_NAMES)
        return pts[sel].astype(np.float32), cols[sel].astype(np.uint8), lab[sel]
    return pts[sel].astype(np.float32), cols[sel].astype(np.uint8)


# ------------------------------------------------------------------ state and goals
def drawer_link_y(u) -> float:
    return float(u._data.body('drawer_link').xpos[1])


def cube_in_drawer(u) -> bool:
    return bool(u._is_in_drawer(u._data.joint('object_joint_0').qpos[:3]))


def place_cube_in_drawer(u, rng):
    """Put the cube at a random spot in the drawer (between its walls, on its floor), random yaw."""
    d = u._data
    y = drawer_link_y(u) + rng.uniform(-0.05, 0.05)
    x = 0.33 + rng.uniform(-0.04, 0.04)
    d.joint('object_joint_0').qpos[:3] = (x, y, CUBE_Z_IN_DRAWER)
    d.joint('object_joint_0').qpos[3:] = lie.SO3.from_z_radians(rng.uniform(0, 2 * np.pi)).wxyz
    d.joint('object_joint_0').qvel[:] = 0.0


def randomize_others(u, task: str, rng):
    """Set every other task done / not done (1/2 each). The locks (buttons) are random from OGBench's reset already and
    stay as drawn. Returns what was set."""
    d = u._data
    out = {}
    if task != 'drawer':
        d.joint('drawer_slide').qpos[0] = DRAWER_OPEN if rng.uniform() < 0.5 else DRAWER_CLOSED
        d.joint('drawer_slide').qvel[0] = 0.0
    if task != 'window':
        d.joint('window_slide').qpos[0] = WINDOW_OPEN if rng.uniform() < 0.5 else WINDOW_CLOSED
        d.joint('window_slide').qvel[0] = 0.0
    mujoco.mj_forward(u._model, d)
    if task != 'cube':
        out['cube_in_drawer'] = rng.uniform() < 0.5
        if out['cube_in_drawer']:
            place_cube_in_drawer(u, rng)
    mujoco.mj_forward(u._model, d)
    return out


def set_task_target(u, task: str):
    """A new goal of the same task: the branch of OGBench SceneEnv.set_new_target for that task (with the env's
    np_random, as there)."""
    rng, d, m = u.np_random, u._data, u._model
    u._target_task = task
    if task == 'cube':
        u._target_block = 0
        put_in_drawer = d.joint('drawer_slide').qpos[0] < -0.12 and rng.uniform() < P_PUT_IN_DRAWER
        if put_in_drawer:
            tar_pos = u._drawer_center.copy()
            tar_pos[:2] = tar_pos[:2] + rng.uniform(-0.005, 0.005, size=2)
        else:
            tar_pos = np.array([*rng.uniform(*u._target_sampling_bounds), 0.02])
        tar_ori = lie.SO3.from_z_radians(rng.uniform(0, 2 * np.pi)).wxyz
        d.mocap_pos[u._cube_target_mocap_ids[0]] = tar_pos
        d.mocap_quat[u._cube_target_mocap_ids[0]] = tar_ori
        for gid in u._cube_target_geom_ids_list[0]:
            m.geom(gid).rgba[3] = 0.0  # the target cube is not drawn
    elif task == 'button':
        u._target_button = rng.choice(u._num_buttons)
        u._target_button_states[u._target_button] = (u._cur_button_states[u._target_button] + 1) % u._num_button_states
    elif task == 'drawer':
        u._target_drawer_pos = DRAWER_OPEN if d.joint('drawer_slide').qpos[0] >= -0.08 else DRAWER_CLOSED
        m.site('drawer_handle_center_target').pos[1] = u._target_drawer_pos
    elif task == 'window':
        u._target_window_pos = WINDOW_OPEN if d.joint('window_slide').qpos[0] <= 0.1 else WINDOW_CLOSED
        m.site('window_handle_center_target').pos[0] = u._target_window_pos
    mujoco.mj_kinematics(m, d)
    return u.compute_observation(), u.compute_ob_info()


def make_oracles(env):
    """The Markov oracles and parameters of OGBench's scene-noisy collection (generate_manipspace.py)."""
    return {
        'cube': CubeMarkovOracle(env=env, min_norm=MIN_NORM, max_step=100),
        'button': ButtonMarkovOracle(env=env, min_norm=MIN_NORM),
        'drawer': DrawerMarkovOracle(env=env, min_norm=MIN_NORM),
        'window': WindowMarkovOracle(env=env, min_norm=MIN_NORM),
    }


def healthy(cube_xyz: np.ndarray) -> bool:
    """OGBench generate_manipspace.py's scene health check (on the cube's position over the episode)."""
    if (cube_xyz[:, 1] >= 0.29).any():
        return False
    if ((cube_xyz[:, 1] <= -0.3) & ((cube_xyz[:, 2] < 0.06) | (cube_xyz[:, 2] > 0.08))).any():
        return False
    return True


# ------------------------------------------------------------------ three-part plan
def draw_plan(rng, force_type=None, episode_len=EPISODE_STEPS):
    """MetaWorld MWTaskEnv._draw_plan as is."""
    t = int(force_type) if force_type is not None else int(rng.choice(3, p=np.asarray(MIX) / np.sum(MIX)))
    noise = float(rng.uniform(0.0, NOISE_MAX)) if t < 2 else 0.0
    b0, blen = -1, 0
    if t == 1:
        b0 = int(rng.integers(15, episode_len - 40))
        if rng.uniform() < BRANCH_TAIL_PROB:
            blen = episode_len
        else:
            blen = int(rng.integers(BURST_LEN[0], BURST_LEN[1] + 1))
    return t, noise, b0, blen


def random_action(cam, data, hand, rng):
    """MetaWorld ThreePartPolicy._random_action: uniform in [-1, 1]; when 3 more steps of it would take the hand out of
    the image, the components pointing away from the image centre are flipped. OGBench's own workspace clip (z >= 0.02)
    keeps the height."""
    a = rng.uniform(-1.0, 1.0, 5)
    ahead = hand + a[:3] * ACTION_SCALE * LOOKAHEAD
    if cam.in_view(data, ahead):
        return a
    away = (VIEW_CENTRE - hand) * a[:3] < 0
    a[:3][away] = -a[:3][away]
    return a


VIEW_CENTRE = np.array([0.425, 0.0, 0.15])  # near the workspace centre (the middle of the image)


# -------------------------------------------------------------------------------------------------- env
class OGBSceneEnv(gymnasium.Env):
    """swm/OGBenchScene-v0, see the module docstring. The OGBench env is ``_env`` (gymnasium.make'd, as OGBench's own
    scripts use it; its unwrapped SceneEnv is ``_u``); the oracle (``agent``) acts on ``ob`` / ``ob_info``, OGBench's
    latest observation and info."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, task: str = 'cube', mode: str = 'data_collection', episode_len: int = EPISODE_STEPS,
                 force_type: int | None = None, render_mode: str = 'rgb_array', **kwargs):
        assert mode in ('data_collection', 'task'), mode
        self.mode, self.task, self.tid = mode, task, TASKS.index(task)
        self.episode_len, self.force_type = episode_len, force_type
        self.render_mode = render_mode
        self._env = make_env(max_episode_steps=episode_len, mode=mode)
        self._u = self._env.unwrapped
        self.action_space = Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, self._env.observation_space.shape, np.float32)
        self.cam = None
        self.rng = None
        self.agent = None
        self.plan = (0, 0.0, -1, 0)
        self.episode_seed = -1
        self._t = 0
        self.ob, self.ob_info = None, None

    # ------------------------------------------------------------------ plan
    def is_rand(self, t: int) -> bool:
        st, _, b0, blen = self.plan
        return st == 2 or (st == 1 and b0 <= t < b0 + blen)

    def rand_now(self) -> bool:
        return self.is_rand(self._t)

    def restart_now(self) -> bool:
        """The backbone takes over again from the current state (the first step after a random burst)."""
        return self._t > 0 and self.is_rand(self._t - 1) and not self.is_rand(self._t)

    def _row_info(self) -> dict:
        u, oi = self._u, self._u.compute_ob_info()
        st, noise, b0, blen = self.plan
        return {
            'task_id': np.int8(self.tid),
            'sample_type': np.int8(st),
            'rand_action': np.int8(self.rand_now()),
            'noise_std': np.float32(noise),
            'branch_t': np.int16(b0),
            'branch_len': np.int16(blen),
            'episode_seed': np.int64(self.episode_seed),
            'button_states': u._cur_button_states.astype(np.int8).copy(),
            'drawer_pos': np.float32(oi['privileged/drawer_pos'][0]),
            'window_pos': np.float32(oi['privileged/window_pos'][0]),
            'cube_pos': oi['privileged/block_0_pos'].astype(np.float32),
            'cube_in_drawer': np.int8(cube_in_drawer(u)),
            'effector_pos': oi['proprio/effector_pos'].astype(np.float32),
            'effector_yaw': oi['proprio/effector_yaw'].astype(np.float32),
            'gripper_opening': oi['proprio/gripper_opening'].astype(np.float32),
            'qpos': u._data.qpos.astype(np.float32).copy(),
            'qvel': u._data.qvel.astype(np.float32).copy(),
            'target_success': np.int8(bool(u._success)),
            'oracle_restart': np.int8(self.restart_now()),
        }

    # -------------------------------------------------------------- gym API
    def _camera(self):
        if self.cam is None or self.cam.model is not self._u._model:  # OGBench builds the model at reset
            if self.cam is not None:
                self.cam.close()
            self.cam = Camera(self._u._model)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # this env's own np_random (stable-worldmodel draws its episode ids from it); unused here
        if seed is None:  # an unseeded reset gets a seed from np_random, so the episode is still recorded and reproducible
            seed = int(self.np_random.integers(2**31))
        if self.mode == 'task':
            np.random.seed(seed % 2**32)  # OGBench's oracles and task mode draw from numpy's global RNG in places
            self.ob, self.ob_info = self._env.reset(seed=seed, options=dict(task_id=int(options['task_id'])))
            self._camera()
            self.episode_seed = seed
            return self.ob, self.ob_info
        np.random.seed(seed % 2**32)
        self.rng = np.random.default_rng(seed)
        self._env.reset(seed=seed)
        u = self._u
        self._camera()
        for _ in range(100):  # hand starts inside the image only (MetaWorld: in_view(p, 2 * margin))
            if self.cam.in_view(u._data, u._data.site_xpos[u._pinch_site_id], 2 * VIEW_MARGIN):
                break
            u.initialize_arm()
        randomize_others(u, self.task, self.rng)
        self.agent = make_oracles(self._env)[self.task]
        self.ob, self.ob_info = set_task_target(u, self.task)
        self.agent.reset(self.ob, self.ob_info)
        self.plan = draw_plan(self.rng, self.force_type, self.episode_len)
        self.episode_seed = seed
        self._t = 0
        return np.asarray(self.ob, np.float32), self._row_info()

    def step(self, action):
        if self.mode == 'task':
            return self._env.step(action)
        rand = self.rand_now()
        self.ob, reward, terminated, truncated, self.ob_info = self._env.step(action)
        self._t += 1
        if not rand and self.agent.done:  # the backbone reached its goal: a new goal of the same task (OGBench's way)
            self.ob, self.ob_info = set_task_target(self._u, self.task)
            self.agent.reset(self.ob, self.ob_info)
        return np.asarray(self.ob, np.float32), float(reward), terminated, truncated, self._row_info()

    def render(self):
        return self.cam.render(self._u._data)

    def close(self):
        if self.cam is not None:
            self.cam.close()
        self._env.close()


# ----------------------------------------------------------------------------------------------- policy
class SceneThreePartPolicy(BasePolicy):
    """Three-part collection policy for OGBSceneEnv: follows each episode's plan. A random row: random_action with the
    episode RNG; a backbone row: the env's oracle (restarted from the current state after a random burst) plus Gaussian
    noise with the episode's std on all 5 entries, from the episode RNG."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'three_part'

    def get_action(self, info_dict, **kwargs):
        envs = [e.unwrapped for e in self.env.envs]
        actions = np.zeros((len(envs), 5), dtype=np.float32)
        for i, env in enumerate(envs):
            u = env._u
            if env.rand_now():
                a = random_action(env.cam, u._data, u._data.site_xpos[u._pinch_site_id].copy(), env.rng)
            else:
                if env.restart_now():  # back to the backbone after a random burst: the oracle restarts from the current state
                    env.agent.reset(env.ob, env.ob_info)
                a = np.asarray(env.agent.select_action(env.ob, env.ob_info), np.float64) + env.rng.normal(0.0, env.plan[1], 5)
            actions[i] = np.clip(a, -1, 1).astype(np.float32)
        return actions
