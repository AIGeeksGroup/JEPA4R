"""The TriFinger env of the LeWM data and the planning evaluation: CortexBench's own TriFinger envs (eai-vc
trifinger_vc's MoveCubeEnv / CubeReachEnv with sim_nn.Task's settings, the envs its BC rollouts run in) behind the
gymnasium interface stable-worldmodel's World and eval.py use. Registered as swm/TriFinger-v0 (task='move' | 'reach';
envs/trifinger/__init__.py).

* Episodes as CortexBench's demos: the envs' fixed initial joint angles, the cube from move_cube.sample_goal(-1);
  move: the goal from move_cube.sample_goal(1), 10 steps; reach (CubeReachEnv, cube fixed): 5 steps, only finger 0
  acts (the env takes 3 entries; the last 6 of the 9-d action are set to 0 and recorded so).
* The policy's plan per episode, MetaWorld's three-part scheme (policy.py acts on it): backbone (sample_type 0: the
  task's expert, expert.py, + Gaussian noise with the episode's std, up to 3 mm), branch (1: the expert until
  branch_t, uniform random actions from there to the end of the episode), random (2: uniform random actions).
* 2D frame (render): the platform's camera60 image (its 10 fps camera; camera180 / camera300 not rendered), 270 x 270,
  resized to 224 x 224 the way bench/cortexbench/run_tf_bc.py (Img224) resizes CortexBench's frames for LeWM (bicubic,
  antialiased).
* 3D (TriFingerPCDWrapper): camera60's depth at the state of the observation (Camera60PCD, also what
  bench/cortexbench/run_tf_bc.py feeds the 3D encoder in CortexBench's rollouts): fingers' moving links + cube,
  1200 points, 5 mm voxels, 25 % on the cube, envs/pcd_wrapper.py's sampling.
* Planning evaluation: set_state(qpos, qvel, ft_des) = the official reset to the joints and the cube pose (at rest: the
  official reset takes no velocities) with the fingertip plan ft_des; set_goal_state(ft_pos, obj_pos) ends the episode
  when every fingertip is within 4 cm and the cube within 3 cm of the goal row's; max_steps raises the official
  episode limit (10 / 5 steps).
* TriFingerPlatform.reset moves the cube with pybullet.resetBasePositionAndOrientation in pybullet client 0 (no
  physicsClientId): replaced by the same reset with the platform's client, so several envs can share a process.
"""

import copy
import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pybullet
import torch
import torchvision.transforms.functional as TF
from gymnasium.spaces import Box

HERE = os.path.dirname(os.path.abspath(__file__))
TF_VC = os.path.normpath(os.path.join(HERE, '..', '..', 'third_party', 'eai-vc', 'cortexbench', 'trifinger_vc', 'src'))
if TF_VC not in sys.path:  # trifinger_vc (CortexBench's TriFinger package; not installed in lewm_tf)
    sys.path.insert(0, TF_VC)

import trifinger_simulation.camera as tf_camera  # noqa: E402
import trifinger_simulation.trifinger_platform as tf_platform
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.trifinger_platform import ObjectType
import trifinger_vc.trifinger_envs.gym_cube_env as gym_cube_env
from trifinger_vc.trifinger_envs.action import ActionType
from trifinger_vc.trifinger_envs.cube_reach import CubeReachEnv

from ..pcd_wrapper import PointCloudWrapper  # noqa: E402

SIM_SETTINGS = dict(action_type=ActionType.TORQUE, step_size=100, visualization=False, finger_type='trifinger_meta',
                    time_step=0.004, camera_delay_steps=0, object_type=ObjectType.COLORED_CUBE, enable_shadows=False,
                    camera_view='default', arena_color='default', visual_observation=True, run_rl_policy=False)
EPISODE_STEPS = {'move': 10, 'reach': 5}  # MoveCubeEnv: 1000 sim steps, CubeReachEnv: 500; one step = 100 (0.4 s)
ACTION_MAX = 0.08                         # m per step and axis: the move_cube demos' max_a (8 cm)
CUBE_HALF = move_cube._CUBE_WIDTH / 2
# the point clouds: the arena and the fingers below their upper links (the upper links sit at z ~0.28, by the camera)
SAMPLER = dict(num_points=1200, voxel_size=0.005, cube_fraction=0.25,
               crop_box=np.asarray(((-0.30, -0.30, -0.01), (0.30, 0.30, 0.25)), dtype=np.float64))


# --------------------------------------------------------------------------------------- runtime patches
def _platform_reset(self, initial_object_pose=None, initial_robot_position=None):
    """TriFingerPlatform.reset (trifinger_simulation, the fork CortexBench pins), the cube reset in the platform's own
    pybullet client."""
    self.simfinger.reset()
    self._next_camera_trigger_t = 0
    self._next_camera_observation_ready_t = None
    self.tricamera = self.make_camera()
    if initial_robot_position is None:
        initial_robot_position = self.initial_robot_position
    self._action_log = {'initial_robot_position': copy.copy(initial_robot_position),
                        'initial_object_pose': copy.copy(initial_object_pose), 'actions': []}
    self.simfinger.reset_finger_positions_and_velocities(initial_robot_position)
    pybullet.resetBasePositionAndOrientation(self.cube._object_id, initial_object_pose.position,
                                             initial_object_pose.orientation,
                                             physicsClientId=self.simfinger._pybullet_client_id)
    self._delayed_camera_observation = self._get_current_camera_observation(0)
    self._camera_observation_t = self._delayed_camera_observation


tf_platform.TriFingerPlatform.reset = _platform_reset
_get_images = tf_camera.CameraArray.get_images


def _camera60_only(self, *args, **kwargs):
    """TriFingerCameras renders camera60 only (camera180 / camera300: zeros; nothing reads them)."""
    if not isinstance(self, tf_camera.TriFingerCameras):
        return _get_images(self, *args, **kwargs)
    first = self.cameras[0].get_image(*args, **kwargs)
    return [first] + [np.zeros_like(first) for _ in self.cameras[1:]]


_get_camera_obs = tf_platform.TriFingerPlatform._get_current_camera_observation
STEP_SIM, CAM_EVERY = 100, 25  # sim steps per env step; the platform's 10 fps camera triggers every 25 sim steps


def _observed_frames_only(self, t=None):
    """Inside an env step (TriFingerEnv.step sets platform._in_step) the camera triggers at sim steps 0, 25, 50 and 75
    of the step; the observation after the step gets the frame of the trigger at 75, the three before it are replaced
    before anything reads them: their images are not rendered (the object pose still is). The frames observed do not
    change (scripts/checks/check_tf_frames.py)."""
    if t is None and getattr(self, '_in_step', False) and (self.simfinger._t + 1) % STEP_SIM != STEP_SIM - CAM_EVERY:
        self.enable_cameras = False
        try:
            return _get_camera_obs(self, t)
        finally:
            self.enable_cameras = True
    return _get_camera_obs(self, t)


def render_camera60_only(skip_unobserved: bool = True):
    tf_camera.CameraArray.get_images = _camera60_only
    if skip_unobserved:
        tf_platform.TriFingerPlatform._get_current_camera_observation = _observed_frames_only


def frame224(img) -> np.ndarray:
    """(H, W, 3) uint8 -> (224, 224, 3) uint8, resized as bench/cortexbench/run_tf_bc.py's Img224 resizes CortexBench's
    frames."""
    t = torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [224, 224], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------------------- point clouds
class Camera60PCD:
    """camera60 (TriFingerCameras, camera_view 'default') rendered at 224 x 224 in a given pybullet client: depth +
    segmentation -> the foreground (the fingers' moving links, the cube) back-projected to world points."""

    def __init__(self, cid, finger_id, cube_id, moving_links, rng):
        self.cid, self.finger_id, self.cube_id = cid, finger_id, cube_id
        self.moving = np.asarray(sorted(moving_links))
        self.cam = tf_camera.TriFingerCameras(pybullet_client_id=cid, target_positions=[None] * 3,
                                              camera_up_vectors=[None] * 3, field_of_view=52,
                                              image_size=(224, 224)).cameras[0]
        view = np.array(self.cam._view_matrix, dtype=np.float64).reshape(4, 4).T
        proj = np.array(self.cam._proj_matrix, dtype=np.float64).reshape(4, 4).T
        self.pv_inv = np.linalg.inv(proj @ view)
        self.s = SimpleNamespace(_rng=rng, **SAMPLER)

    def points(self):
        """All foreground points of the current state: xyz (N, 3) m, rgb (N, 3) uint8, is_cube (N,)."""
        _, _, rgb, depth, seg = pybullet.getCameraImage(
            224, 224, viewMatrix=self.cam._view_matrix, projectionMatrix=self.cam._proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER, shadow=False,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=self.cid)
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(224, 224, 4)[..., :3].reshape(-1, 3)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        seg = np.asarray(seg, dtype=np.int64).reshape(-1)
        hit = seg >= 0
        obj = np.where(hit, seg & 0xFFFFFF, -1)
        link = np.where(hit, (seg >> 24) - 1, -2)
        cube = obj == self.cube_id
        idx = np.flatnonzero(cube | ((obj == self.finger_id) & np.isin(link, self.moving)))
        v, u = np.divmod(idx, 224)
        ndc = np.stack([2.0 * (u + 0.5) / 224 - 1.0, 1.0 - 2.0 * (v + 0.5) / 224, 2.0 * depth[idx] - 1.0,
                        np.ones(len(idx))], axis=1)
        w = ndc @ self.pv_inv.T
        return w[:, :3] / w[:, 3:4], rgb[idx], cube[idx]

    def __call__(self):
        xyz, col = PointCloudWrapper._sample(self.s, *self.points())
        return np.concatenate([xyz, col.astype(np.float32)], axis=1)  # (1200, 6): xyz m, rgb 0..255


class TriFingerPCDWrapper(PointCloudWrapper):
    """'pcd_xyz' / 'pcd_rgb' in every reset / step info: Camera60PCD's points, envs/pcd_wrapper.py's sampling (crop,
    5 mm voxel de-duplication, a guaranteed cube share, shuffle) with the wrapper's per-episode RNG."""

    def __init__(self, env, **kwargs):
        kw = {k: v for k, v in SAMPLER.items()}
        kw.update(kwargs)
        super().__init__(env, cameras=('camera60',), **kw)

    def _point_cloud(self) -> dict:
        xyz, col = self._sample(*self.env.unwrapped.pcd_camera().points())
        return {'pcd_xyz': xyz, 'pcd_rgb': col}


# -------------------------------------------------------------------------------------------------- env
class TriFingerEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 2.5}

    def __init__(self, task: str = 'move', mix=(0.5, 0.3, 0.2), force_type: int | None = None,
                 noise_max: float = 0.003, max_steps: int | None = None, goal_tol_ft: float = 0.04,
                 goal_tol_obj: float = 0.03, render_mode: str = 'rgb_array', skip_unobserved: bool = True):
        render_camera60_only(skip_unobserved)
        if task == 'move':
            self._env = gym_cube_env.MoveCubeEnv(goal_pose=None, goal_visualization=False, no_collisions=False,
                                                 **SIM_SETTINGS)
        elif task == 'reach':
            self._env = CubeReachEnv(**SIM_SETTINGS)
        else:
            raise ValueError(task)
        self.task, self.render_mode = task, render_mode
        self.episode_steps = EPISODE_STEPS[task]
        if max_steps is not None:  # planning evaluation: more steps than an official episode
            if task == 'move':
                gym_cube_env.MOVE_CUBE_EPISODE = max(gym_cube_env.MOVE_CUBE_EPISODE, 100 * int(max_steps))
            else:
                self._env.max_episode_len = max(self._env.max_episode_len, 100 * int(max_steps))
        p = self._env.platform
        self._cid = p.simfinger._pybullet_client_id
        self.finger_id, self.cube_id = p.simfinger.finger_id, p.cube._object_id
        self.moving_links = sorted(p.simfinger.pybullet_link_indices + p.simfinger.pybullet_tip_link_indices)
        self._joints = p.simfinger.pybullet_joint_indices
        self.action_space = Box(-ACTION_MAX, ACTION_MAX, (9,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (25,), np.float64)  # fingertips, joints, cube position + quaternion
        self.mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
        self.force_type, self.noise_max = force_type, float(noise_max)
        self.goal_tol_ft, self.goal_tol_obj = float(goal_tol_ft), float(goal_tol_obj)
        self._pcd_cam = None
        self._goal, self._goal_pos = None, np.zeros(3)
        self._t, self._plan = 0, (0, 0.0, -1, 0)
        self._o, self._frame = None, None

    # ----------------------------------------------------------------------------------- state
    def pcd_camera(self) -> Camera60PCD:
        if self._pcd_cam is None:  # its sampling RNG is unused here: TriFingerPCDWrapper samples with its own
            self._pcd_cam = Camera60PCD(self._cid, self.finger_id, self.cube_id, self.moving_links,
                                        np.random.default_rng(0))
        return self._pcd_cam

    def in_view(self, p, margin: float = 12.0) -> bool:
        """p projects at least margin pixels inside camera60's 224 x 224 image."""
        cam = self.pcd_camera().cam
        view = np.array(cam._view_matrix, dtype=np.float64).reshape(4, 4).T
        proj = np.array(cam._proj_matrix, dtype=np.float64).reshape(4, 4).T
        c = proj @ view @ np.append(np.asarray(p, dtype=np.float64), 1.0)
        if c[3] <= 0:
            return False
        u, v = (c[0] / c[3] + 1.0) / 2.0 * 224, (1.0 - c[1] / c[3]) / 2.0 * 224
        return margin <= u <= 224 - margin and margin <= v <= 224 - margin

    def _set_obs(self, o):
        self._o, self._frame = o, None

    def _parts(self):
        o = self._o
        q = np.asarray(o['robot_position'], dtype=np.float64)
        ft = np.asarray(o['ft_pos_cur'], dtype=np.float64).reshape(9)
        cube_p = np.asarray(o['object_position'], dtype=np.float64)
        cube_q = np.asarray(o['object_orientation'], dtype=np.float64)
        return q, ft, cube_p, cube_q

    def _vec(self):
        q, ft, cube_p, cube_q = self._parts()
        return np.concatenate([ft, q, cube_p, cube_q])

    def _info(self):
        q, ft, cube_p, cube_q = self._parts()
        dq = np.array([s[1] for s in pybullet.getJointStates(self.finger_id, self._joints, physicsClientId=self._cid)])
        v, w = pybullet.getBaseVelocity(self.cube_id, physicsClientId=self._cid)
        t, noise, b0, blen = self._plan
        return {
            'qpos': np.concatenate([q, cube_p, cube_q]),             # the 9 joints + the cube pose: set_state
            'qvel': np.concatenate([dq, v, w]),
            'ft_pos': ft,                                             # fingertips (3 x xyz)
            'ft_des': np.asarray(self._env.x_des_plan, dtype=np.float64).reshape(9).copy(),  # the plan the actions move
            'obj_pos': cube_p,
            'obj_quat': cube_q,
            'subtask': np.int64(0 if self.task == 'move' else 1),    # 0 move, 1 reach
            'goal_pos': self._goal_pos.copy(),                        # move: where the cube goes; reach: the cube
            'sample_type': np.int64(t),
            'rand_action': np.int64(self._rand_action()),
            'noise_std': np.float64(noise),
            'branch_t': np.int64(b0),
            'branch_len': np.int64(blen),
        }

    def _success(self) -> float:  # move: the cube within 2 cm (xy) of the goal; reach: finger 0 on the cube
        _, ft, cube_p, _ = self._parts()
        if self.task == 'move':
            return float(np.linalg.norm(cube_p[:2] - self._goal_pos[:2]) < 0.02)
        return float(np.linalg.norm(ft[:3] - cube_p) < CUBE_HALF + 0.015)

    # ------------------------------------------------------------------------------------ plan
    def _draw_plan(self, rng):
        t = int(self.force_type) if self.force_type is not None else int(rng.choice(3, p=self.mix))
        noise = float(rng.uniform(0.0, self.noise_max)) if t < 2 else 0.0
        b0, blen = -1, 0
        if t == 1:  # the expert for steps 0 .. b0 - 1, random actions from b0 to the end
            b0, blen = int(rng.integers(1, self.episode_steps)), self.episode_steps
        return t, noise, b0, blen

    def _rand_action(self) -> bool:
        t, _, b0, blen = self._plan
        return t == 2 or (t == 1 and b0 <= self._t < b0 + blen)

    # ---------------------------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        move_cube.seed(int(rng.integers(0, 2**31 - 1)))  # the samplers' module RNG, from this reset's generator
        init = move_cube.sample_goal(difficulty=-1)
        init_d = {'position': np.asarray(init.position, dtype=np.float64),
                  'orientation': np.asarray(init.orientation, dtype=np.float64)}
        if self.task == 'move':
            goal = move_cube.sample_goal(difficulty=1)
            self._goal_pos = np.asarray(goal.position, dtype=np.float64)
            o = self._env.reset(goal_pose_dict={'position': self._goal_pos,
                                                'orientation': np.asarray(goal.orientation, dtype=np.float64)},
                                init_pose_dict=init_d)
        else:
            self._goal_pos = init_d['position'].copy()
            o = self._env.reset(init_pose_dict=init_d)
        self._set_obs(o)
        self._t, self._goal = 0, None
        self._plan = self._draw_plan(rng)
        info = self._info()
        info['task_success'] = np.float64(0.0)
        return self._vec(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(9), -ACTION_MAX, ACTION_MAX)
        self._env.platform._in_step = True
        try:
            o, _, _, _ = self._env.step(a if self.task == 'move' else a[:3])
        finally:
            self._env.platform._in_step = False
        self._set_obs(o)
        self._t += 1
        info = self._info()
        info['task_success'] = np.float64(self._success())
        terminated = False
        if self._goal is not None:
            _, ft, cube_p, _ = self._parts()
            ft_ok = np.linalg.norm((ft - self._goal[0]).reshape(3, 3), axis=1).max() < self.goal_tol_ft
            terminated = bool(ft_ok and np.linalg.norm(cube_p - self._goal[1]) < self.goal_tol_obj)
        return self._vec(), 0.0, terminated, False, info

    def render(self):
        if self._frame is None:
            self._frame = frame224(np.asarray(self._o['camera_observation']['camera60']['image']))
        return self._frame

    # ------------------------------------------------------------------------------ eval hooks
    def set_state(self, qpos, qvel=None, ft_des=None):
        qpos = np.asarray(qpos, dtype=np.float64)
        init = {'position': qpos[9:12], 'orientation': qpos[12:16]}
        if self.task == 'move':
            o = self._env.reset(goal_pose_dict={'position': self._goal_pos, 'orientation': np.array([0.0, 0, 0, 1])},
                                init_pose_dict=init, init_robot_position=qpos[:9])
        else:
            o = self._env.reset(init_pose_dict=init, init_robot_position=qpos[:9])
        if ft_des is not None:
            self._env.x_des_plan = torch.FloatTensor(np.asarray(ft_des, dtype=np.float64).reshape(9))
        self._set_obs(o)

    def set_goal_state(self, ft_pos, obj_pos):
        self._goal = (np.asarray(ft_pos, dtype=np.float64), np.asarray(obj_pos, dtype=np.float64))

    def close(self):
        if pybullet.isConnected(physicsClientId=self._cid):
            pybullet.disconnect(physicsClientId=self._cid)

