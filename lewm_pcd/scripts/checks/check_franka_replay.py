"""Deterministic MuJoCo replay checker for the OGBench Franka adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import mujoco
import numpy as np

import envs  # noqa: F401
from scripts.checks.verify_franka import assert_rollout_equal

ENV_ID = "swm/OGBCubeFranka-v0"


def _variation_values(space, prefix=""):
    values = {}
    children = getattr(space, "spaces", None)
    if children is not None:
        for name, child in children.items():
            values.update(_variation_values(child, f"{prefix}{name}/"))
    elif hasattr(space, "value"):
        values[prefix.rstrip("/")] = np.asarray(space.value).copy()
    return values


def snapshot_env(env) -> dict[str, object]:
    """Capture full MuJoCo physics plus target, variation and previous-state data."""
    robot = env.unwrapped
    state_size = mujoco.mj_stateSize(
        robot._model, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(
        robot._model,
        robot._data,
        state,
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    return {
        "mujoco_state": state,
        "ctrl": robot._data.ctrl.copy(),
        "mocap_pos": robot._data.mocap_pos.copy(),
        "mocap_quat": robot._data.mocap_quat.copy(),
        "prev_qpos": robot._prev_qpos.copy(),
        "prev_qvel": robot._prev_qvel.copy(),
        "target_block": int(robot._target_block),
        "success": bool(robot._success),
        "variation": _variation_values(robot.variation_space),
    }


def restore_env(env, snapshot: dict[str, object]) -> None:
    """Restore all fields captured by :func:`snapshot_env`."""
    robot = env.unwrapped
    mujoco.mj_setState(
        robot._model,
        robot._data,
        np.asarray(snapshot["mujoco_state"]),
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    robot._data.ctrl[:] = snapshot["ctrl"]
    robot._data.mocap_pos[:] = snapshot["mocap_pos"]
    robot._data.mocap_quat[:] = snapshot["mocap_quat"]
    robot._prev_qpos = np.asarray(snapshot["prev_qpos"]).copy()
    robot._prev_qvel = np.asarray(snapshot["prev_qvel"]).copy()
    robot._target_block = int(snapshot["target_block"])
    robot._success = bool(snapshot["success"])
    for path, value in snapshot["variation"].items():
        node = robot.variation_space
        for name in path.split("/"):
            node = node.spaces[name]
        node._value = np.asarray(value).copy()
    robot._reset_next_step = False
    mujoco.mj_forward(robot._model, robot._data)


def rollout_states(env, actions: np.ndarray) -> list[np.ndarray]:
    robot = env.unwrapped
    states = []
    for action in actions:
        _, _, _, _, info = env.step(action)
        cube_pos = np.concatenate(
            [info[f"privileged/block_{i}_pos"] for i in range(robot._num_cubes)]
        )
        states.append(
            np.concatenate([robot._data.qpos.copy(), robot._data.qvel.copy(), cube_pos])
        )
    return states


def check_replay(episodes: int = 10, steps: int = 20, seed: int = 7000) -> dict:
    env = gym.make(
        ENV_ID,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
    )
    max_error = 0.0
    checked_steps = 0
    try:
        for episode in range(episodes):
            np.random.seed(seed + episode)
            env.reset(seed=seed + episode)
            initial = snapshot_env(env)
            actions = np.random.default_rng(seed + episode).uniform(
                -0.4, 0.4, size=(steps, 5)
            )
            first = rollout_states(env, actions)
            restore_env(env, initial)
            second = rollout_states(env, actions)
            assert_rollout_equal(first, second, atol=1e-10)
            for expected, observed in zip(first, second):
                max_error = max(
                    max_error,
                    float(np.max(np.abs(expected - observed))),
                )
            checked_steps += steps
    finally:
        env.close()
    return {
        "episodes": episodes,
        "steps_per_episode": steps,
        "checked_steps": checked_steps,
        "max_abs_error": max_error,
        "atol": 1e-10,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()
    result = check_replay(args.episodes, args.steps, args.seed)
    home = Path(os.environ["STABLEWM_HOME"])
    report = home / "logs" / "franka" / "check_franka_replay.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
