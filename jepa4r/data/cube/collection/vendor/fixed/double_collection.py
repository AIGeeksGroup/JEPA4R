"""Fixed-task collection on OGBench's DoubleCube with its plan oracle: the fixed-task wrapper, environment and plan
policy of triple_collection.py with two cubes, episodes of 500 actions and the Double task's stages."""

from __future__ import annotations

import importlib

import numpy as np

from triple_collection import TripleFixedPlanPolicyMixin, make_official_triple_env, make_triple_wrapper

DOUBLE_ENV_ID = "swm/OGBCubeOfficialFixed-v0"


class DoubleStages:
    def __init__(self, init_xyzs, goal_xyzs, *, task_id=None):  # task_id: passed by the shared wrapper, unused
        self.init_xyzs = np.array(init_xyzs, dtype=np.float64, copy=True)
        self.goal_xyzs = np.array(goal_xyzs, dtype=np.float64, copy=True)
        if self.init_xyzs.shape != (2, 3) or self.goal_xyzs.shape != (2, 3):
            raise ValueError("Double task requires full (2,3) initial and goal XYZ")
        if not np.isfinite(self.init_xyzs).all() or not np.isfinite(self.goal_xyzs).all():
            raise ValueError("Double task XYZ must be finite")
        self.init_xyzs.setflags(write=False)
        self.goal_xyzs.setflags(write=False)
        if np.allclose(self.goal_xyzs, self.init_xyzs[::-1], rtol=0, atol=1e-7):
            self.stages = [(0, np.array([0.35, 0.0, 0.02])), (1, self.goal_xyzs[1]), (0, self.goal_xyzs[0])]
        else:
            self.stages = [
                (int(i), self.goal_xyzs[i])
                for i in np.argsort(self.goal_xyzs[:, 2], kind="stable")
                if not np.allclose(self.init_xyzs[i], self.goal_xyzs[i], rtol=0, atol=1e-7)
            ]
        if not self.stages:
            raise ValueError("Double task has no movement stage")
        self.index = self.revision = 0
        self.pending = self.finished = False

    def after_step(self, xyzs):
        if not self.pending or self.finished:
            return
        self.pending = False
        block, goal = self.stages[self.index]
        if np.linalg.norm(xyzs[block] - goal) <= 0.04:
            if self.index == len(self.stages) - 1:
                self.finished = True
                return
            self.index += 1
        # A timed-out unsuccessful plan retries its actual current stage.
        self.revision += 1


DoubleFixedPlanPolicyMixin = TripleFixedPlanPolicyMixin


def register_double_environment():
    import gymnasium as gym

    importlib.import_module("ogbench.manipspace")  # registers the OGBench manipulation envs
    if DOUBLE_ENV_ID not in gym.registry:
        gym.register(DOUBLE_ENV_ID, entry_point=make_official_triple_env, max_episode_steps=500)


def make_double_wrapper(env, *, task_id, strategy, mixed_switch_step, capture):
    return make_triple_wrapper(
        env,
        task_id=task_id,
        strategy=strategy,
        mixed_switch_step=mixed_switch_step,
        capture=capture,
        num_cubes=2,
        stages_type=DoubleStages,
        episode_steps=500,
    )
