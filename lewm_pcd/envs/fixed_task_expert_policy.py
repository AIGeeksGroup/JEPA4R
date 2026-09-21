"""Plan-oracle policy for the fixed OGBench UR5e cube tasks."""

from __future__ import annotations

import numpy as np
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from stable_worldmodel.policy import BasePolicy

from .ur5e_cube_tasks import build_subtask_plan, get_task


class FixedTaskCubePlanPolicy(BasePolicy):
    """Sequence OGBench's single-cube plan oracle over one fixed multi-cube task."""

    def __init__(
        self,
        family: str,
        task_id: int,
        action_noise: float = 0.1,
        noise_smoothing: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = "fixed_task_plan_oracle"
        self.task = get_task(family, task_id)
        self.action_noise = float(action_noise)
        self.noise_smoothing = float(noise_smoothing)

    def set_env(self, env):
        super().set_env(env)
        if not env.envs:
            raise ValueError("fixed-task policy requires at least one environment")
        for wrapped in env.envs:
            single = wrapped.unwrapped
            if getattr(single, "_env_type", None) != self.task.env_type:
                raise ValueError(
                    f"env_type must be {self.task.env_type}, got "
                    f"{getattr(single, '_env_type', None)}"
                )
            if getattr(single, "_mode", None) != "task":
                raise ValueError("fixed-task policy requires OGBench task mode")
            configured_task = getattr(
                single, "_reward_task_id", getattr(single, "cur_task_id", None)
            )
            if configured_task != self.task.task_id:
                raise ValueError(
                    f"task_id must be {self.task.task_id}, got {configured_task}"
                )
            if getattr(single, "_permute_blocks", False):
                raise ValueError("fixed-task policy requires permute_blocks=False")

        self._plans = [build_subtask_plan(self.task) for _ in env.envs]
        self._plan_indices = np.zeros(env.num_envs, dtype=np.int64)
        self.finished = np.zeros(env.num_envs, dtype=bool)
        self._agents = [
            CubePlanOracle(
                env=wrapped.unwrapped,
                noise=self.action_noise,
                noise_smoothing=self.noise_smoothing,
            )
            for wrapped in env.envs
        ]

    @staticmethod
    def _extract_info(info_dict, env_index):
        return {
            key: value[env_index][0]
            for key, value in info_dict.items()
            if not key.startswith("_")
        }

    def _begin_step(self, env_index, info):
        step = self._plans[env_index][self._plan_indices[env_index]]
        single = self.env.envs[env_index].unwrapped
        single.set_target_pos(step.block_id, step.target_xyz)
        oracle_info = dict(info)
        oracle_info["privileged/target_task"] = "cube"
        oracle_info["privileged/target_block"] = step.block_id
        oracle_info["privileged/target_block_pos"] = step.target_xyz.copy()
        oracle_info["privileged/target_block_yaw"] = np.zeros(1, dtype=np.float64)
        self._agents[env_index].reset(None, oracle_info)

    def get_action(self, info_dict, **_kwargs):
        if self.env is None:
            raise RuntimeError("environment is not attached")
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for env_index in range(self.env.num_envs):
            info = self._extract_info(info_dict, env_index)
            if int(np.asarray(info["step_idx"]).item()) == 0:
                self._plan_indices[env_index] = 0
                self.finished[env_index] = False
                self._begin_step(env_index, info)
            if self.finished[env_index]:
                continue

            agent = self._agents[env_index]
            actions[env_index] = np.clip(agent.select_action(None, info), -1, 1)
            if agent.done:
                self._plan_indices[env_index] += 1
                if self._plan_indices[env_index] >= len(self._plans[env_index]):
                    self.finished[env_index] = True
                else:
                    self._begin_step(env_index, info)
        return actions
