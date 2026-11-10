"""Gymnasium bridge to pinned mj_envs Relocate; no normalized-action rescaling.

Native state dictionaries are flattened into numeric info columns, including
every initial value, for the existing stable-worldmodel World.collect writer.
The collection-plan formula retains the final feedback mixture. Adroit keeps
the native reset distribution, horizon and success definition.
"""

from __future__ import annotations

import hashlib
import json

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

SAMPLE_TYPES = ('backbone', 'branch', 'random')
# action, native observation, qpos/qvel, native horizon
TASKS = {'relocate-v0': (30, 39, 36, 200)}


def legacy_state_sha256(state):
    """Match the historical collector: full value lists + json sort_keys."""
    payload = json.dumps({key: np.asarray(value).tolist() for key, value in state.items()}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class AdroitEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 100}

    def __init__(self, env_name='relocate-v0', width=224, height=224, episode_len=None,
                 mix=(.5, .3, .2), force_type=None, noise_max=.2,
                 branch_tail_prob=.5, burst_len=(10, 30), render_mode='rgb_array', camera='vil_camera'):
        if env_name not in TASKS:
            raise ValueError(f'unsupported Adroit task: {env_name}')
        self.env_name = env_name
        self.task = env_name
        self.action_dim, self.observation_dim, self.qpos_dim, horizon = TASKS[env_name]
        self.episode_len = horizon if episode_len is None else int(episode_len)
        if not 0 < self.episode_len <= horizon:
            raise ValueError(f'episode_len must be in [1, {horizon}]')
        self.width, self.height, self.camera = int(width), int(height), camera
        self.render_mode = render_mode
        self.mix = np.asarray(mix, dtype=np.float64)
        if self.mix.shape != (3,) or not np.isfinite(self.mix).all() or (self.mix < 0).any() or self.mix.sum() <= 0:
            raise ValueError('mix must contain three nonnegative finite weights')
        self.mix = self.mix / self.mix.sum()
        if force_type is not None and force_type not in (0, 1, 2):
            raise ValueError('force_type must be 0, 1, 2 or None')
        if self.episode_len <= 55 and (force_type == 1 or (force_type is None and self.mix[1] > 0)):
            raise ValueError('branch sampling requires horizon >55 for the unchanged branch formula')
        self.force_type, self.noise_max = force_type, float(noise_max)
        self.branch_tail_prob, self.burst_len = float(branch_tail_prob), tuple(int(x) for x in burst_len)
        if not np.isfinite(self.noise_max) or self.noise_max < 0 or not 0 <= self.branch_tail_prob <= 1:
            raise ValueError('invalid noise_max or branch_tail_prob')
        if len(self.burst_len) != 2 or not 0 < self.burst_len[0] <= self.burst_len[1]:
            raise ValueError('invalid burst_len')
        self.native_env = self._make_native_env(env_name)
        self.sim = self.native_env.sim
        if self.native_env.action_space.shape != (self.action_dim,) or self.native_env.observation_space.shape != (self.observation_dim,):
            raise ValueError('native Adroit space dimensions do not match the pinned contract')
        self.action_space = Box(-1., 1., (self.action_dim,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.observation_dim,), np.float64)
        self._t, self._seed, self._plan = 0, -1, (0, 0., -1, 0)
        self._initial_state = None

    @staticmethod
    def _make_native_env(env_name):
        # Direct native classes avoid importing a legacy Gym TimeLimit into the
        # Gymnasium wrapper stack. This adapter supplies the native horizon.
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
        return RelocateEnvV0()

    @staticmethod
    def _vector(value, size, name):
        out = np.asarray(value, dtype=np.float64)
        if out.shape != (size,):
            raise ValueError(f'{name} must have shape ({size},), got {out.shape}')
        if not np.isfinite(out).all():
            raise ValueError(f'{name} contains NaN or Inf')
        return out.copy()

    def get_obs(self):
        return self._vector(self.native_env.get_obs(), self.observation_dim, 'native observation')

    @property
    def proprio(self):
        return self.get_obs()[:self.action_dim].copy()

    def _checked_state(self, state):
        sizes = {'qpos': self.qpos_dim, 'qvel': self.qpos_dim}
        sizes.update({'obj_pos': 3, 'target_pos': 3})
        for key, size in sizes.items():
            if key not in state:
                raise ValueError(f'native state is missing {key}')
            self._vector(state[key], size, key)
        out = {key: np.array(value, copy=True) for key, value in state.items()}
        if any(value.dtype.kind not in 'iuf' or not np.isfinite(value).all() for value in out.values()):
            raise ValueError('native state must contain finite numeric values')
        return out

    def get_env_state(self):
        return self._checked_state(self.native_env.get_env_state())

    def set_env_state(self, state):
        self.native_env.set_env_state(self._checked_state(state))

    def _draw_plan(self, rng):
        kind = int(self.force_type) if self.force_type is not None else int(rng.choice(3, p=self.mix))
        noise = float(rng.uniform(0., self.noise_max)) if kind < 2 else 0.
        branch_t, branch_len = -1, 0
        if kind == 1:
            # Retain the exclusive Relocate branch start range [15,160).
            branch_t = int(rng.integers(15, self.episode_len - 40))
            branch_len = self.episode_len if rng.uniform() < self.branch_tail_prob else int(rng.integers(self.burst_len[0], self.burst_len[1] + 1))
        return kind, noise, branch_t, branch_len

    def _info(self, obs, goal_achieved=0):
        state = self.get_env_state()
        kind, noise, branch_t, branch_len = self._plan
        info = {key: value.copy() for key, value in state.items()}
        info.update({'initial_' + key: value.copy() for key, value in self._initial_state.items()})
        digest, initial_digest = legacy_state_sha256(state), legacy_state_sha256(self._initial_state)
        info.update(native_observation=obs.copy(), initial_native_observation=self._initial_obs.copy(),
                    proprio=obs[:self.action_dim].copy(),
                    state_sha256=np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy(),
                    initial_state_sha256=np.frombuffer(bytes.fromhex(initial_digest), dtype=np.uint8).copy(),
                    legacy_initial_state_sha256=np.frombuffer(initial_digest.encode('ascii'), dtype=np.uint8).copy(),
                    episode_seed=np.int64(self._seed), native_step_idx=np.int64(self._t),
                    executed_action=np.zeros(self.action_dim, dtype=np.float64), executed_action_valid=np.int64(0),
                    sample_type=np.int64(kind), noise_std=np.float64(noise),
                    branch_t=np.int64(branch_t), branch_len=np.int64(branch_len),
                    rand_action=np.int64(kind == 2 or (kind == 1 and branch_t <= self._t < branch_t + branch_len)),
                    goal_achieved=np.int64(goal_achieved), task_success=np.float64(goal_achieved))
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = -1 if seed is None else int(seed)
        self.native_env.seed(seed)
        self.native_env.reset()
        if options and 'initial_state' in options:
            self.set_env_state(options['initial_state'])
        self._t = 0
        self._plan = self._draw_plan(self.np_random)
        obs = self.get_obs()
        self._initial_state, self._initial_obs = self.get_env_state(), obs.copy()
        return obs, self._info(obs)

    def step(self, action):
        executed = self._vector(action, self.action_dim, 'action')
        if (np.abs(executed) > 1.).any():
            raise ValueError('action must be within the normalized range [-1, 1]')
        obs, reward, done, native_info = self.native_env.step(executed)
        self._t += 1
        obs = self._vector(obs, self.observation_dim, 'native observation')
        info = self._info(obs, int(native_info.get('goal_achieved', 0)))
        info['executed_action'] = executed.copy()
        info['executed_action_valid'] = np.int64(1)
        return obs, float(reward), bool(done), bool(self._t >= self.episode_len and not done), info

    def render(self, camera=None, depth=False, segmentation=False, multisample=True):
        if not multisample and self.sim.model.vis.quality.offsamples != 0:
            raise RuntimeError('non-MSAA rendering requires a separate validated offscreen context; use the Adroit PCD backend')
        raw = self.sim.render(width=self.width, height=self.height, camera_name=camera or self.camera,
                              depth=depth, segmentation=segmentation, device_id=0)
        if depth:
            raw = np.asarray(raw[1], dtype=np.float64)[::-1]
            near = self.sim.model.vis.map.znear * self.sim.model.stat.extent
            far = self.sim.model.vis.map.zfar * self.sim.model.stat.extent
            return near / (1. - raw * (1. - near / far))
        return np.asarray(raw)[::-1].copy()

    def camera_pose(self):
        camera_id = self.sim.model.camera_name2id(self.camera)
        return (np.asarray(self.sim.data.cam_xpos[camera_id], dtype=np.float64).copy(),
                np.asarray(self.sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
                float(self.sim.model.cam_fovy[camera_id]))

    def close(self):
        self.native_env.close()
