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
            terminated = False
            truncated = False
            steps = 0
            started_at = time.perf_counter()

            while not (terminated or truncated):
                policy_info = dict(info)
                policy_info.update(goal_info)
                action = policy.act(obs, policy_info)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1

            records.append(
                {
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
            )
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
