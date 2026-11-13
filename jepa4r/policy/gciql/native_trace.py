# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/native_trace.py (verbatim)
"""Passive Native state/action trace; no physics, random draws or encoder calls."""
import numpy as np


class NativeTrace:
    def __init__(self):
        self.states = []
        self.observations = []
        self.goal = None
        self.inputs = []
        self.events = []

    def _state(self, base):
        data = base._data
        self.states.append(dict(qpos=np.array(data.qpos, copy=True),
                                qvel=np.array(data.qvel, copy=True),
                                ctrl=np.array(data.ctrl, copy=True), time=float(data.time)))

    def reset(self, base, observation, goal):
        if self.states:
            raise ValueError('one NativeTrace belongs to exactly one real reset')
        self._state(base)
        self.observations.append(np.array(observation, copy=True))
        self.goal = np.array(goal, copy=True)

    def policy(self, current_latent, goal_latent, current_input, goal_input, raw_action):
        self.inputs.append(dict(current_latent=np.array(current_latent, copy=True),
                                goal_latent=np.array(goal_latent, copy=True),
                                policy_current=np.array(current_input, copy=True),
                                policy_goal=np.array(goal_input, copy=True),
                                policy_actions=np.array(raw_action, copy=True)))

    def step(self, base, action, observation, reward, terminated, truncated, info):
        self.events.append(dict(executed_actions=np.array(action, copy=True), rewards=float(reward),
                                terminated=bool(terminated), truncated=bool(truncated),
                                successes=bool(info.get('success', False))))
        self._state(base)
        self.observations.append(np.array(observation, copy=True))

    def arrays(self):
        count = len(self.events)
        if count == 0 or len(self.inputs) != count or len(self.states) != count + 1:
            raise ValueError('Native policy/action/state trace alignment differs')
        arrays = {name: np.asarray([row[name] for row in self.states]) for name in self.states[0]}
        arrays.update({name: np.asarray([row[name] for row in self.inputs]) for name in self.inputs[0]})
        arrays.update({name: np.asarray([row[name] for row in self.events]) for name in self.events[0]})
        arrays.update(observations=np.asarray(self.observations), goal=self.goal,
                      observation_step=np.arange(count + 1, dtype=np.int64),
                      action_step=np.arange(count, dtype=np.int64))
        if arrays['executed_actions'].shape != (count, 5) or arrays['policy_actions'].shape != (count, 5):
            raise ValueError('Native trace action dimensions differ')
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('nonfinite Native trace')
        return arrays
