"""Thin fixed-task TripleCube copy of the frozen Double adapter; same plan oracle."""
from __future__ import annotations

import importlib
import hashlib
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np

from collection_protocol import action_source

OFFICIAL_CUBE_SHA = '6f8457ffd634f6620411c788c86e871f44ea5b5b5eecf76cd694fc04d12d678d'
FRAME_SHA = '695be6a7027d78cbed4c32fc5ea388c1d229203a4e0d3fe0c206bc63c2f4c13c'
TRIPLE_ENV_ID = 'swm/OGBCubeOfficialFixedTriple-v0'


def whole_success(xyzs, goals):
    return bool(np.all(np.linalg.norm(np.asarray(xyzs) - goals, axis=-1) <= .04))


def local_pitch_quaternion(quat, degrees):
    w, x, y, z = np.asarray(quat)
    c, s = np.cos(np.deg2rad(degrees) / 2), np.sin(np.deg2rad(degrees) / 2)
    return np.array([w*c - x*s, w*s + x*c, y*c + z*s, z*c - y*s])


class TripleStages:
    def __init__(self, init_xyzs, goal_xyzs, *, task_id, num_cubes=3):
        self.init_xyzs = np.array(init_xyzs, dtype=np.float64, copy=True)
        self.goal_xyzs = np.array(goal_xyzs, dtype=np.float64, copy=True)
        if num_cubes not in (3, 4) or self.init_xyzs.shape != (num_cubes, 3) or self.goal_xyzs.shape != (num_cubes, 3):
            raise ValueError(f'Cube task requires full ({num_cubes},3) initial and goal XYZ')
        if not np.isfinite(self.init_xyzs).all() or not np.isfinite(self.goal_xyzs).all():
            raise ValueError('Triple task XYZ must be finite')
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= 5:
            raise ValueError('Triple task_id must be an integer from 1 through 5')
        self.init_xyzs.setflags(write=False)
        self.goal_xyzs.setflags(write=False)
        if task_id == 4:
            # A -> buffer, C -> A, B -> C, A -> B. Buffer is a candidate pending real smoke.
            buffer = np.array([.35, -.2, .02])
            if not np.all(np.linalg.norm(np.r_[self.init_xyzs, self.goal_xyzs] - buffer, axis=1) > .04):
                raise ValueError('Triple cycle buffer overlaps a fixed initial or goal slot')
            self.stages = [
                (0, buffer),
                *[(i, self.goal_xyzs[i]) for i in range(num_cubes - 1, 0, -1)],
                (0, self.goal_xyzs[0]),
            ]
        else:
            # Dismantle the initial tower top first; build the goal tower bottom first.
            order = np.argsort(-self.init_xyzs[:, 2] if task_id == 3 else self.goal_xyzs[:, 2], kind='stable')
            self.stages = [(int(i), self.goal_xyzs[i]) for i in order
                           if not np.allclose(self.init_xyzs[i], self.goal_xyzs[i], rtol=0, atol=1e-7)]
        if not self.stages:
            raise ValueError('Triple task has no movement stage')
        self.index = self.revision = 0
        self.pending = self.finished = False

    def after_step(self, xyzs):
        if not self.pending or self.finished:
            return
        self.pending = False
        block, goal = self.stages[self.index]
        if np.linalg.norm(xyzs[block] - goal) <= .04:
            if self.index == len(self.stages) - 1:
                self.finished = True
                return
            self.index += 1
        # A timed-out unsuccessful plan retries its actual current stage.
        self.revision += 1


def make_official_triple_env(**kwargs):
    from ogbench.manipspace.envs.cube_env import CubeEnv
    source = Path(importlib.util.find_spec(CubeEnv.__module__).origin)
    if hashlib.sha256(source.read_bytes()).hexdigest() != OFFICIAL_CUBE_SHA:
        raise ValueError('Official CubeEnv source changed')

    class SeededTripleCube(CubeEnv):
        @property
        def action_space(self):
            if getattr(self, '_collection_action_space', None) is None:
                box = super().action_space
                box.seed(getattr(self, '_collection_reset_seed', None))
                self._collection_action_space = box
            return self._collection_action_space

        def reset(self, *, seed=None, options=None, **reset_kwargs):
            self._collection_action_space = None
            self._collection_reset_seed = seed
            self._collection_reset_actions = []
            self._collection_in_reset = True
            try:
                return super().reset(seed=seed, options=options, **reset_kwargs)
            finally:
                self._collection_in_reset = False

        def step(self, action):
            if getattr(self, '_collection_in_reset', False):
                self._collection_reset_actions.append(np.asarray(action).copy())
            return super().step(action)

    return SeededTripleCube(**kwargs)


def register_triple_environment():
    import gymnasium as gym
    importlib.import_module('ogbench.manipspace')   # registers the OGBench manipulation envs
    if TRIPLE_ENV_ID not in gym.registry:
        gym.register(TRIPLE_ENV_ID, entry_point=make_official_triple_env, max_episode_steps=1000)


def make_triple_wrapper(env, *, task_id, strategy, mixed_switch_step, capture, num_cubes=3, stages_type=TripleStages):
    import gymnasium as gym
    import mujoco
    frame_path = Path(capture['frame_source_path'])
    if hashlib.sha256(frame_path.read_bytes()).hexdigest() != FRAME_SHA:
        raise ValueError('Original NativePCDFrame source changed')
    spec = importlib.util.spec_from_file_location('sealed_triple_native_frame', frame_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core_factory = None
    if 'crop_box' in capture:
        from envs.pcd_wrapper import PointCloudWrapper
        core_factory = partial(PointCloudWrapper, crop_box=capture['crop_box'])

    class FixedTripleWrapper(gym.Wrapper):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.frame = None
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            if seed is None:
                raise ValueError('Fixed Triple collection requires an explicit reset seed')
            if self.frame is not None:
                self.frame.close()
            _, _ = self.env.reset(seed=int(seed), options={'task_id': task_id, 'render_goal': False})
            u = self.unwrapped
            task = u.task_infos[task_id - 1]
            self.stages = stages_type(task['init_xyzs'], task['goal_xyzs'], task_id=task_id)
            u._collection_stages = self.stages
            # Fixed nominal object XYZ, retaining the official seeded reset yaw and arm state.
            for i in range(num_cubes):
                u._data.joint(f'object_joint_{i}').qpos[:3] = self.stages.init_xyzs[i]
            self.goal_xyzs = np.stack([u._data.mocap_pos[mid].copy() for mid in u._cube_target_mocap_ids])
            if not np.allclose(self.goal_xyzs, self.stages.goal_xyzs, rtol=0, atol=1e-7):
                raise ValueError('Official fixed goal identity changed')
            self.goal_xyzs.setflags(write=False)
            mujoco.mj_forward(u._model, u._data)
            u.pre_step()
            u.post_step()
            cid = mujoco.mj_name2id(u._model, mujoco.mjtObj.mjOBJ_CAMERA, 'front_pixels')
            sid = mujoco.mj_name2id(u._model, mujoco.mjtObj.mjOBJ_CAMERA, 'front')
            if int(u._model.cam_bodyid[cid]) != 0 or int(u._model.cam_bodyid[sid]) != 0:
                raise ValueError('Camera must have fixed world parent')
            if not np.array_equal(u._model.cam_pos[sid], [1.287, 0., .509]) or u._model.cam_fovy[sid] != 45.:
                raise ValueError('Official front position/FOV changed')
            u._model.cam_pos[cid] = u._model.cam_pos[sid]
            u._model.cam_fovy[cid] = u._model.cam_fovy[sid]
            u._model.cam_quat[cid] = local_pitch_quaternion(u._model.cam_quat[sid], 5.)
            mujoco.mj_camlight(u._model, u._data)
            self.camera = (u._data.cam_xpos[cid].copy(), u._data.cam_xmat[cid].copy(), float(u._model.cam_fovy[cid]))
            self.cid, self.seed, self.steps = cid, int(seed), 0
            self.frame = module.NativePCDFrame(capture['core_path'], capture['provider_path'], self.seed,
                                               core_factory=core_factory)
            return u.compute_observation(), self._capture(u.get_reset_info())

        def _capture(self, info):
            u = self.unwrapped
            goals = np.stack([u._data.mocap_pos[mid] for mid in u._cube_target_mocap_ids])
            if not np.array_equal(goals, self.goal_xyzs):
                raise ValueError('Whole-task mocap goal changed')
            if (
                not np.array_equal(u._data.cam_xpos[self.cid], self.camera[0])
                or not np.array_equal(u._data.cam_xmat[self.cid], self.camera[1])
                or u._model.cam_fovy[self.cid] != self.camera[2]
            ):
                raise ValueError('Fixed camera changed')
            rgb = u.render(camera='front_pixels')
            cloud = self.frame.capture(u, u._data, rgb, 'current')
            blocks = np.stack([info[f'privileged/block_{i}_pos'] for i in range(num_cubes)])
            block, subgoal = self.stages.stages[self.stages.index]
            source = (
                2 if self.steps == 1000 else int(action_source(strategy, self.steps, mixed_switch_step) == 'random')
            )
            info.update(
                {
                    'pixels': rgb,
                    'pcd_xyz': cloud[:, :3].astype(np.float32),
                    'pcd_rgb': cloud[:, 3:].astype(np.uint8),
                    'privileged/target_task': 'cube',
                    'privileged/target_block': block,
                    'privileged/target_block_pos': subgoal.copy(),
                    'privileged/target_block_yaw': np.zeros(1),
                    'collection_reset_seed': self.seed,
                    'collection_task_id': task_id,
                    'collection_init_xyzs': self.stages.init_xyzs.copy(),
                    'collection_goal_xyzs': self.goal_xyzs.copy(),
                    'collection_stage': self.stages.index,
                    'collection_plan_revision': self.stages.revision,
                    'collection_action_source': source,
                    'collection_ctrl': u._data.ctrl.copy(),
                    'collection_source_time': float(u._data.time),
                    'collection_reset_actions': np.asarray(u._collection_reset_actions, dtype=np.float32),
                    'collection_active_success': bool(np.linalg.norm(blocks[block] - subgoal) <= 0.04),
                    'success': whole_success(blocks, self.goal_xyzs),
                    'collection_camera_pos': self.camera[0].copy(),
                    'collection_camera_xmat': self.camera[1].reshape(3, 3).copy(),
                    'collection_camera_fovy': self.camera[2],
                }
            )
            return info

        def step(self, action):
            ob, reward, terminated, truncated, info = self.env.step(action)
            self.steps += 1
            xyzs = np.stack([info[f'privileged/block_{i}_pos'] for i in range(num_cubes)])
            self.stages.after_step(xyzs)
            return ob, reward, terminated, truncated, self._capture(info)

        def close(self):
            if self.frame is not None:
                self.frame.close()
                self.frame = None
            u = self.unwrapped
            renderer = getattr(u, '_renderer', None)
            if renderer is not None:
                renderer.close()
                u._renderer = None
            return self.env.close()

    return FixedTripleWrapper(env)


class TripleFixedPlanPolicyMixin:
    def get_action(self, info_dict, **kwargs):
        if not hasattr(self, '_double_revisions'):
            self._double_revisions = {}
            self.oracle_subplan_seeds = {}
        for i, wrapped in enumerate(self.env.envs):
            info = {key: value[i][0] for key, value in info_dict.items() if not key.startswith('_')}
            seed, revision = int(info['collection_reset_seed']), int(info['collection_plan_revision'])
            if (int(info['step_idx']) != 0 and self._double_revisions.get(i) != (seed, revision)
                    and action_source(self.strategy, int(info['step_idx']), self.mixed_switch_step) == 'expert'):
                plan_seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))
                state = np.random.get_state()
                try:
                    np.random.seed(plan_seed)
                    self._agents[i].reset(None, info)
                finally:
                    np.random.set_state(state)
                self.oracle_subplan_seeds.setdefault(seed, []).append(plan_seed)
            self._double_revisions[i] = (seed, revision)
        actions = super().get_action(info_dict, **kwargs)
        for i, wrapped in enumerate(self.env.envs):
            info = {key: value[i][0] for key, value in info_dict.items() if not key.startswith('_')}
            seed = int(info['collection_reset_seed'])
            if int(info['step_idx']) == 0:
                self.oracle_subplan_seeds[seed] = [self.oracle_episode_seeds[seed]]
            if (
                action_source(self.strategy, int(info['step_idx']), self.mixed_switch_step) == 'expert'
                and self._agents[i].done
            ):
                wrapped.unwrapped._collection_stages.pending = True
        return actions
