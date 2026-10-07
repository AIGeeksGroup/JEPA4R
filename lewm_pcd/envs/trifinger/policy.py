"""Three-part collection policy for env.TriFingerEnv, as envs/metaworld/policy.py: it follows each episode's plan
(sample type, noise level, random steps); the env marks the random rows (``rand_action``).

* backbone (type 0): the task's CortexBench expert (expert.MoveExpert / ReachExpert, reproduced from CortexBench's
  demos), Gaussian noise with the episode's std (up to 3 mm) on every action entry the env executes.
* branch (type 1): the expert until branch_t, uniform random actions from there to the end of the episode.
* random (type 2): uniform random actions.
Random actions: uniform fingertip displacements in [-2, 2] cm per axis, reflected (the components pointing away from
the arena centre at 6 cm height) when three such steps would take a fingertip target out of camera60's view, below
1 cm or above 15 cm. Reach: the env executes finger 0's three entries only; the other six are 0.
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.policy import BasePolicy

from .env import ACTION_MAX
from .expert import MoveExpert, ReachExpert

RAND_STEP = 0.02
LOW_Z, HIGH_Z = 0.01, 0.15
VIEW_CENTRE = np.array([0.0, 0.0, 0.06])


class TriFingerThreePartPolicy(BasePolicy):
    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'three_part'
        self.rng = np.random.default_rng(seed)
        self._experts = []

    def set_env(self, env):
        self.env = env
        self._experts = [None] * env.num_envs

    def get_action(self, info_dict, **kwargs):
        envs = [e.unwrapped for e in self.env.envs]
        actions = np.zeros((len(envs), 9), dtype=np.float32)
        for i, env in enumerate(envs):
            info = {k: v[i][0] for k, v in info_dict.items() if not k.startswith('_')}
            k = int(info['step_idx'])
            move = int(info['subtask']) == 0
            cube_p = np.asarray(info['obj_pos'], dtype=np.float64)
            cube_q = np.asarray(info['obj_quat'], dtype=np.float64)
            if k == 0 or self._experts[i] is None:
                ft = np.asarray(info['ft_pos'], dtype=np.float64)
                if move:
                    self._experts[i] = MoveExpert()
                    self._experts[i].reset(ft, cube_p, cube_q, np.asarray(info['goal_pos'], dtype=np.float64))
                else:
                    self._experts[i] = ReachExpert()
                    self._experts[i].reset(ft, cube_p, cube_q)
            if bool(info['rand_action']):
                a = self._random_action(env, np.asarray(info['ft_des'], dtype=np.float64))
            else:
                a = self._experts[i].act(k, cube_p, cube_q) + self.rng.normal(0.0, float(info['noise_std']), 9)
            if not move:
                a[3:] = 0.0
            actions[i] = np.clip(a, -ACTION_MAX, ACTION_MAX)
        return actions

    def _random_action(self, env, des):
        a = self.rng.uniform(-RAND_STEP, RAND_STEP, (3, 3))
        d = des.reshape(3, 3)
        for f in range(3):
            ahead = d[f] + 3 * a[f]  # where three such steps would take this fingertip's target
            if env.in_view(ahead) and LOW_Z <= ahead[2] <= HIGH_Z:
                continue
            away = (VIEW_CENTRE - d[f]) * a[f] < 0  # reflect every component that points away from the centre
            a[f][away] = -a[f][away]
        return a.reshape(9)
