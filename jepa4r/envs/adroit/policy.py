# Moved from OGBench-cube@4fbb0c6:lewm_pcd/envs/adroit/policy.py (verbatim)
"""Trusted MJRL state feedback and the existing World.set_policy duck interface.

This module needs only NumPy until a known checkpoint is explicitly unpickled;
the CPU expert smoke can therefore use the legacy simulator environment without
installing the modern collector stack. Random Adroit joint actions are uniform;
MetaWorld's Cartesian workspace reflection has no native joint-space analogue.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import chain
from pathlib import Path
import pickle

import numpy as np


class SavedStateExpert:
    def __init__(self, policy, *, env_name='relocate-v0', provenance=None):
        dimensions = {'relocate-v0': (39, 30)}
        if env_name not in dimensions:
            raise ValueError(f'unsupported Adroit task: {env_name}')
        self.observation_dim, self.action_dim = dimensions[env_name]
        if (getattr(policy, 'n', None), getattr(policy, 'm', None)) != dimensions[env_name]:
            raise ValueError(f'{env_name} expert must have n={self.observation_dim}, m={self.action_dim}')
        if not callable(getattr(policy, 'get_action', None)):
            raise TypeError('MJRL expert must provide get_action')
        model = getattr(policy, 'model', None)
        if any(not callable(getattr(model, method, None)) for method in ('to', 'eval', 'parameters', 'buffers')):
            raise TypeError('MJRL model must provide to/eval/parameters/buffers')
        model.to('cpu')  # pinned FCNetwork.to mutates and returns None
        model.eval()
        if getattr(model, 'training', None) is not False:
            raise ValueError('expert model must be in eval mode')
        devices = [getattr(model, 'device', 'cpu')]
        devices += [tensor.device for tensor in chain(model.parameters(), model.buffers())]
        devices += [getattr(model, name).device for name in ('in_shift', 'in_scale', 'out_shift', 'out_scale')
                    if hasattr(getattr(model, name, None), 'device')]
        if any(str(device).split(':', 1)[0] != 'cpu' for device in devices):
            raise ValueError('expert model and normalization tensors must be on CPU')
        self.policy, self.env_name = policy, env_name
        self.provenance = dict(provenance or {})

    @classmethod
    def load(cls, expert_path, expert_sha256, *, env_name='relocate-v0'):
        path = Path(expert_path).expanduser().resolve()
        expected = str(expert_sha256).lower()
        if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError('an explicit 64-character expert SHA-256 is required')
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f'expert SHA-256 mismatch: expected {expected}, got {digest}')
        policy = pickle.loads(payload)
        return cls(policy, env_name=env_name, provenance={'policy_path': str(path),
                   'policy_sha256': digest, 'policy_bytes': len(payload),
                   'policy_module': type(policy).__module__, 'policy_type': type(policy).__name__})

    @staticmethod
    def _vector(value, size, name):
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.asarray(value, dtype=np.float32)
        if out.shape != (size,):
            raise ValueError(f'{name} must have shape ({size},), got {out.shape}')
        if not np.isfinite(out).all():
            raise ValueError(f'{name} contains NaN or Inf')
        return out.copy()

    def action(self, native_env):
        observation = self._vector(native_env.get_obs(), self.observation_dim, 'expert observation')
        result = self.policy.get_action(observation)
        if not isinstance(result, (tuple, list)) or len(result) != 2 or not isinstance(result[1], Mapping) or 'evaluation' not in result[1]:
            raise ValueError('MJRL get_action must return [sampled, info] with evaluation; no sampled fallback')
        # Leave the raw mean unclipped: collection adds noise before final clip.
        return self._vector(result[1]['evaluation'], self.action_dim, 'expert evaluation action')


class AdroitFeedbackPolicy:
    """World policy: current native obs -> MJRL mean -> plan noise -> clip."""

    def __init__(self, expert_path, expert_sha256, env_name='relocate-v0', seed=None):
        self.expert = SavedStateExpert.load(expert_path, expert_sha256, env_name=env_name)
        self.type, self.seed, self.env = 'adroit_feedback', seed, None
        self.rng = np.random.default_rng(seed)  # independent from the PCD wrapper RNG

    def set_env(self, env):
        for wrapped in env.envs:
            native = wrapped.unwrapped
            if native.env_name != self.expert.env_name or native.action_space.shape != (self.expert.action_dim,):
                raise ValueError('collector task/action dimensions disagree with the expert')
        self.env = env

    def get_action(self, info_dict, **kwargs):
        if self.env is None:
            raise RuntimeError('set_env must be called before get_action')
        actions = np.empty((self.env.num_envs, self.expert.action_dim), dtype=np.float32)
        for index, wrapped in enumerate(self.env.envs):
            if bool(info_dict['rand_action'][index][0]):
                action = self.rng.uniform(-1., 1., self.expert.action_dim)
            else:
                action = self.expert.action(wrapped.unwrapped.native_env)
                action = action + self.rng.normal(0., float(info_dict['noise_std'][index][0]), self.expert.action_dim)
            actions[index] = np.clip(action, -1., 1.)
        return actions
