"""Per-environment point-cloud history for PPO with a frozen temporal encoder.

History belongs to observations, never to the neural network: PPO may replay
transitions in any order and must obtain the same features. Every returned
observation owns its arrays, including the terminal next observation. The
wrapper executes exactly one underlying control action per ``step``.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from operator import index

import gymnasium as gym
import numpy as np

from maniskill2_learn.env.wrappers import WRAPPERS

from .official import lewm_config


def _positive_integer(value, name):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


class PointCloudHistoryBuffer:
    """Store the most recent T consecutive PPO control-step frames.

    For T=3, a snapshot at step t is [o_(t-2), o_(t-1), o_t]. Missing pre-reset
    frames repeat the initial frame. History sampling is consecutive in both
    current world-model training and PPO; the original LeWM dataset frameskip
    is a separate setting that is no longer copied into these runs.
    Only xyz/rgb enter the representation; robot state is not concatenated.
    """

    def __init__(self, history_size=None, history_stride=1):
        cfg = lewm_config() if history_size is None else None
        self.history_size = _positive_integer(
            cfg.history_size if history_size is None else history_size, "history_size"
        )
        self.history_stride = _positive_integer(history_stride, "history_stride")
        if self.history_stride != 1:
            raise ValueError("PPO history_stride must be 1: use consecutive control-step frames.")
        self.raw_history_size = self.history_size
        self._frames = deque(maxlen=self.raw_history_size)

    @staticmethod
    def _copy_frame(observation):
        if not isinstance(observation, Mapping):
            raise TypeError("Expected an observation with xyz and rgb point-cloud arrays.")
        if "xyz" not in observation or "rgb" not in observation:
            raise ValueError("Point-cloud history requires both xyz and rgb.")
        frame = {key: np.array(observation[key], copy=True) for key in ("xyz", "rgb")}
        for key, value in frame.items():
            if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] == 0:
                raise ValueError(f"Expected {key} point-cloud shape [N,3], with N > 0.")
        if frame["xyz"].shape != frame["rgb"].shape:
            raise ValueError("xyz and rgb must have the same point count.")
        return frame

    def reset(self, observation):
        frame = self._copy_frame(observation)
        self._frames.clear()
        # Stored frames are immutable internally; snapshot() always copies.
        self._frames.extend(frame for _ in range(self.raw_history_size))
        return self.snapshot()

    def append(self, observation):
        if not self._frames:
            raise RuntimeError("Reset the point-cloud history before appending a frame.")
        frame = self._copy_frame(observation)
        for key in frame:
            previous = self._frames[-1][key]
            if frame[key].shape != previous.shape or frame[key].dtype != previous.dtype:
                raise ValueError(f"{key} shape and dtype must stay constant within an episode.")
        self._frames.append(frame)
        return self.snapshot()

    def snapshot(self):
        if not self._frames:
            raise RuntimeError("Reset the point-cloud history before reading an observation.")
        frames = list(self._frames)
        return {key: np.stack([frame[key] for frame in frames]) for key in ("xyz", "rgb")}


@WRAPPERS.register_module()
class PointCloudHistory(gym.Wrapper):
    """Extra wrapper after ManiSkill2_ObsWrapper, before TimeLimit/ExtendedEnv.

    ``reset`` follows Gymnasium's (obs, info) API; ``step`` returns five items.
    Termination/truncation flags pass through and the subsequent reset clears
    history. Thus a terminal next_obs keeps its own episode's valid context.
    ``env_name`` is accepted for the official ``build_wrapper`` injection.
    """

    def __init__(self, env, history_size=None, history_stride=1, env_name=None):
        super().__init__(env)
        self.history = PointCloudHistoryBuffer(history_size, history_stride)
        self.history_size = self.history.history_size
        self.history_stride = self.history.history_stride
        self.env_name = env_name
        space = env.observation_space
        # Some upstream ManiSkill2 observation wrappers expose the original
        # unprocessed space. In that case establish the actual space on reset,
        # without an additional reset, renderer call, or sampling operation.
        if isinstance(space, gym.spaces.Dict) and all(key in space.spaces for key in ("xyz", "rgb")):
            point_spaces = {}
            for key in ("xyz", "rgb"):
                original = space[key]
                if not isinstance(original, gym.spaces.Box):
                    break
                point_spaces[key] = gym.spaces.Box(
                    low=np.broadcast_to(original.low, (self.history_size, *original.shape)).copy(),
                    high=np.broadcast_to(original.high, (self.history_size, *original.shape)).copy(),
                    dtype=original.dtype,
                )
            if len(point_spaces) == 2:
                self.observation_space = gym.spaces.Dict(point_spaces)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        history = self.history.reset(observation)
        # The PPO runner infers shapes from reset observations, but also expose
        # an accurate Gymnasium space to consumers of that standard interface.
        point_spaces = {}
        for key, value in history.items():
            if np.issubdtype(value.dtype, np.integer):
                limits = np.iinfo(value.dtype)
                low, high = limits.min, limits.max
            else:
                low, high = -np.inf, np.inf
            point_spaces[key] = gym.spaces.Box(low, high, shape=value.shape, dtype=value.dtype)
        self.observation_space = gym.spaces.Dict(point_spaces)
        return history, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.history.append(observation), reward, terminated, truncated, info

    def get_obs(self):
        """Return the current history without advancing or resampling it."""
        return self.history.snapshot()
