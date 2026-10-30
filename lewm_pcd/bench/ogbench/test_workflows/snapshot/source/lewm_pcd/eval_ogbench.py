"""Native OGBench task-level evaluation helpers.

Unlike ``eval.py``, this module never samples starts or goals from an offline
dataset. Each episode begins with ``env.reset(task_id=...)`` and runs until the
environment terminates or reaches its time limit.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


class _SingleEnvBatchAdapter:
    """Expose one Gymnasium env with the action-space surface planners expect."""

    def __init__(self, env):
        from gymnasium.vector.utils import batch_space

        self.env = env
        self.envs = [env]
        self.num_envs = 1
        self.single_action_space = env.action_space
        self.action_space = batch_space(env.action_space, 1)


class NativeBatchedPolicy:
    """Use a stable-worldmodel batched policy in the native single-env loop."""

    def __init__(self, policy, env, static_info=None):
        self.policy = policy
        self.env = env
        self._batch_env = _SingleEnvBatchAdapter(env)
        self.static_info = dict(static_info or {})
        self._step_idx = 0
        self._last_action = None

    def reset(self):
        # WorldModelPolicy.set_env rebuilds action/history/warm-start buffers.
        self.policy.set_env(self._batch_env)
        reset_solver = getattr(getattr(self.policy, "solver", None), "reset_episode", None)
        if reset_solver is not None:
            reset_solver()
        self._step_idx = 0
        self._last_action = None

    def act(self, obs, info):
        info = {**info, **self.static_info}
        batched = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                batched[key] = value[None, None, ...]
            elif isinstance(value, (bool, int, float, np.number)):
                batched[key] = np.asarray([[value]])
            elif isinstance(value, (str, bytes)):
                batched[key] = [[value]]
        action_info = (
            np.full(self.env.action_space.shape, np.nan, dtype=np.float32)
            if self._last_action is None
            else self._last_action
        )
        batched["action"] = np.asarray(action_info, dtype=np.float32)[None, None]
        batched["step_idx"] = np.asarray([[self._step_idx]], dtype=np.int64)
        action = np.asarray(self.policy.get_action(batched))
        if action.shape != (1, *self.env.action_space.shape):
            raise ValueError(
                f"expected batched action shape {(1, *self.env.action_space.shape)}, "
                f"got {action.shape}"
            )
        self._last_action = action[0].astype(np.float32, copy=True)
        self._step_idx += 1
        return action[0]


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        pass

    def act(self, obs, info):
        return self.action_space.sample()


def _episode_seed(seed_manifest, task_id, episode_idx):
    if seed_manifest is None:
        return None
    seeds = seed_manifest[task_id]
    if len(seeds) <= episode_idx:
        raise ValueError(
            f"task {task_id} has {len(seeds)} seeds, "
            f"but episode {episode_idx} was requested"
        )
    return int(seeds[episode_idx])


def _task_name(env, task_id):
    task_info = getattr(getattr(env, "unwrapped", env), "cur_task_info", None)
    if isinstance(task_info, Mapping):
        return task_info.get("task_name", f"task{task_id}")
    return f"task{task_id}"


def evaluate_native(
    env,
    policy,
    *,
    task_ids: Sequence[int],
    episodes_per_task: int,
    seed_manifest: Mapping[int, Sequence[int]] | None = None,
    on_reset=None,
    on_step=None,
    on_episode_end=None,
):
    """Run complete native task episodes and return one record per episode."""
    records = []
    for task_id in task_ids:
        for episode_idx in range(episodes_per_task):
            seed = _episode_seed(seed_manifest, task_id, episode_idx)
            obs, info = env.reset(
                seed=seed,
                options={"task_id": int(task_id), "render_goal": True},
            )
            goal_info = {
                key: value for key, value in info.items() if key.startswith("goal")
            }
            policy.reset()
            if on_reset is not None:
                on_reset(
                    task_id=int(task_id),
                    episode_idx=episode_idx,
                    seed=seed,
                    obs=obs,
                    info=info,
                )
            terminated = False
            truncated = False
            steps = 0
            started_at = time.perf_counter()

            while not (terminated or truncated):
                policy_info = dict(info)
                policy_info.update(goal_info)
                action = policy.act(obs, policy_info)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if on_step is not None:
                    on_step(
                        task_id=int(task_id),
                        episode_idx=episode_idx,
                        seed=seed,
                        steps=steps,
                        obs=obs,
                        info=info,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    )

            record = {
                "task_id": int(task_id),
                "task_name": _task_name(env, task_id),
                "episode_idx": episode_idx,
                "seed": seed,
                "success": bool(info["success"]),
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "wall_time": time.perf_counter() - started_at,
            }
            records.append(record)
            if on_episode_end is not None:
                on_episode_end(record=record, obs=obs, info=info)
    return records


def summarize_results(records):
    """Return per-task SR and the unweighted mean across tasks."""
    successes = defaultdict(list)
    for record in records:
        successes[int(record["task_id"])].append(float(record["success"]))
    if not successes:
        raise ValueError("cannot summarize an empty result set")

    summary = {
        f"task{task_id}_sr": float(np.mean(values))
        for task_id, values in sorted(successes.items())
    }
    summary["mean_sr"] = float(
        np.mean([summary[f"task{task_id}_sr"] for task_id in sorted(successes)])
    )
    return summary
