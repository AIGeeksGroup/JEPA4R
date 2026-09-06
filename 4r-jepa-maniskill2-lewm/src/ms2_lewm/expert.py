"""Original OGB cube oracle and noise through ManiSkill interface adapters.

The official wrapper controls xyz/yaw/grip (5 DOF). ManiSkill exposes local
xyz/rotation-vector/grip (7 DOF); a coordinate/action adapter is required.
The stochastic policy method and every noise parameter come from upstream.
"""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle


# Loading this exact untouched module directly avoids executing OGB's package
# __init__, which eagerly imports its unrelated MuJoCo renderer/pygame envs.
# No source rewriting, algorithm copies, or dependency stubs are used.
_EXPERT_PATH = Path(__file__).resolve().parents[2] / "vendor/stable-worldmodel-lewm-compatible/stable_worldmodel/envs/ogbench/expert_policy.py"
_EXPERT_MODULE = "ms2_lewm._official_ogb_expert"
if _EXPERT_MODULE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_EXPERT_MODULE, _EXPERT_PATH)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_EXPERT_MODULE] = _module
    _spec.loader.exec_module(_module)
ExpertPolicy = sys.modules[_EXPERT_MODULE].ExpertPolicy


def official_noise_config():
    defaults = inspect.signature(ExpertPolicy.__init__).parameters
    return {
        name: defaults[name].default
        for name in (
            "policy_type", "action_noise", "p_random_action",
            "noise_smoothing", "min_norm",
        )
    }


def gripper_target_action(qpos, delta_closedness, lower, upper):
    """Invert the actual controller affine map, including its force endpoint."""
    closedness = np.clip((upper - qpos) / (upper - lower), 0.0, 1.0)
    target_closedness = np.clip(closedness + delta_closedness, 0.0, 1.0)
    target_qpos = upper - target_closedness * (upper - lower)
    return 2.0 * (target_qpos - lower) / (upper - lower) - 1.0


def capture_numpy_rng():
    kind, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"kind": kind, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached_gaussian": cached_gaussian}


def restore_numpy_rng(state):
    np.random.set_state((state["kind"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))


class ManiSkillNoisyExpert(ExpertPolicy):
    """Reuse original CubeMarkovOracle AND its original noisy policy method."""

    def __init__(self, seed=None):
        # No noise argument overrides: resolve every default from pinned source.
        super().__init__(seed=seed)

    def set_env(self, env):
        self.actual_env = env
        self.env = SimpleNamespace(
            envs=env.envs,
            num_envs=env.num_envs,
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(env.num_envs, 5), dtype=np.float32),
        )
        self._p_stack = np.zeros(env.num_envs)
        self._xi = np.zeros(env.num_envs)
        self._agents = [None] * env.num_envs
        self._oracle_agents = {
            "cube": [CubeMarkovOracle(env=e.unwrapped, min_norm=self.min_norm) for e in env.envs]
        }
        self.episode_noise_scales = []

    def get_action(self, info_dict, **kwargs):
        # Translate environment state keys without changing the collector's
        # stored observation dictionary or the original oracle implementation.
        oracle_info = dict(info_dict)
        if all(hasattr(env.unwrapped, "oracle_info") for env in self.actual_env.envs):
            states = [env.unwrapped.oracle_info() for env in self.actual_env.envs]
            for key in states[0]:
                oracle_info[key] = np.asarray([state[key] for state in states])[:, None, ...]
        # This executes the pinned official code unchanged: episode xi uniform,
        # random-action draw, [xi,xi,xi,3xi,10xi] Gaussian, clipping and retarget.
        canonical = super().get_action(oracle_info, **kwargs)
        self.last_canonical_actions = canonical.copy()
        for i in range(self.actual_env.num_envs):
            if np.asarray(info_dict["step_idx"])[i].reshape(-1)[0] == 0:
                self.episode_noise_scales.append(float(self._xi[i]))
        actions = np.zeros((self.actual_env.num_envs, 7), dtype=np.float32)
        for i, env in enumerate(self.actual_env.envs):
            simulator = env.unwrapped.simulator
            rotation = simulator.tcp.pose.to_transformation_matrix()[:3, :3]
            # OGBench's physical action range is [.05m,.05m,.05m,.3rad,1].
            # Panda pd_ee_delta_pose uses [.1m,.1m,.1m,.1rad,.1rad,.1rad].
            actions[i, :3] = rotation.T @ (canonical[i, :3] * 0.05 / 0.1)
            actions[i, 3:6] = rotation.T @ np.array([0.0, 0.0, canonical[i, 3] * 0.3 / 0.1])
            # OGBench uses a delta in closedness; ManiSkill a target finger
            # opening. Translate units/sign, preserving the same noisy delta.
            bounds = simulator.agent.controller.controllers["gripper"]._action_space
            actions[i, 6] = gripper_target_action(
                np.mean(simulator.agent.robot.get_qpos()[-2:]), canonical[i, 4],
                float(bounds.low[0]), float(bounds.high[0]),
            )
        return np.clip(actions, -1.0, 1.0)
