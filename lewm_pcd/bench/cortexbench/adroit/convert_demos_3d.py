"""Convert official Adroit demonstrations to the independent PCD extension format."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from .pcd import AdroitPCDConfig
from .smoke import capture_adroit_frame


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"replay shape mismatch: actual={actual.shape}, expected={expected.shape}")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def convert_pen_demo(
    demo: dict,
    env,
    *,
    capture_fn: Callable = capture_adroit_frame,
    rng: np.random.Generator | None = None,
    config: AdroitPCDConfig = AdroitPCDConfig(),
    replay_atol: float = 1e-6,
) -> tuple[dict, dict]:
    """Replay one official Pen demo and replace only ``images`` with XYZRGB frames.

    The official action, observation, rewards, initial state and environment metadata
    remain unchanged.  ``state_obs`` is used as the replay contract because the
    official ``observations`` field is the 1536-dimensional CortexBench feature
    vector, while the native Pen environment returns a 45-dimensional state.
    """
    required = {"actions", "state_obs", "images", "env_infos", "init_state_dict"}
    missing = sorted(required.difference(demo))
    if missing:
        raise ValueError(f"official Pen demo is missing keys: {missing}")

    actions = np.asarray(demo["actions"])
    state_obs = np.asarray(demo["state_obs"])
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, D), got {actions.shape}")
    if state_obs.ndim != 2 or len(state_obs) != len(actions):
        raise ValueError(f"state_obs must have shape (T, S) aligned with actions, got {state_obs.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(state_obs).all():
        raise ValueError("official demo action/state contains NaN or Inf")

    env_infos = demo["env_infos"]
    proprio = np.asarray(env_infos.get("proprio")) if "proprio" in env_infos else None
    if proprio is not None and (proprio.ndim != 2 or len(proprio) != len(actions)):
        raise ValueError(f"env_infos['proprio'] is not aligned with actions: {proprio.shape}")

    if rng is None:
        rng = np.random.default_rng(0)
    env.set_env_state(deepcopy(demo["init_state_dict"]))

    pcd_frames = []
    max_state_error = 0.0
    max_proprio_error = 0.0
    stats = []
    for timestep in range(len(actions)):
        actual_state = np.asarray(env.get_obs())
        state_error = _max_abs_error(actual_state, state_obs[timestep])
        max_state_error = max(max_state_error, state_error)
        if state_error > replay_atol:
            raise ValueError(f"state replay mismatch at frame {timestep}: max_abs_error={state_error}")

        if proprio is not None:
            actual_proprio = actual_state[: proprio.shape[1]]
            proprio_error = _max_abs_error(actual_proprio, proprio[timestep])
            max_proprio_error = max(max_proprio_error, proprio_error)
            if proprio_error > replay_atol:
                raise ValueError(f"proprio replay mismatch at frame {timestep}: max_abs_error={proprio_error}")

        _, xyz, rgb, frame_stats = capture_fn(env, rng, config)
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.uint8)
        if xyz.shape != (config.num_points, 3) or rgb.shape != (config.num_points, 3):
            raise ValueError(f"PCD frame {timestep} has invalid shapes: xyz={xyz.shape}, rgb={rgb.shape}")
        if not np.isfinite(xyz).all():
            raise ValueError(f"PCD frame {timestep} contains NaN or Inf")
        pcd_frames.append(np.concatenate([xyz, rgb.astype(np.float32)], axis=1))
        stats.append(dict(frame=timestep, **{str(k): float(v) for k, v in frame_stats.items()}))

        if timestep + 1 < len(actions):
            env.step(np.asarray(actions[timestep]))

    converted = dict(demo)
    converted["images"] = np.asarray(pcd_frames, dtype=np.float32)
    converted["env_infos"] = dict(env_infos)
    report = {
        "frames": int(len(actions)),
        "action_dim": int(actions.shape[1]),
        "state_dim": int(state_obs.shape[1]),
        "proprio_dim": int(proprio.shape[1]) if proprio is not None else 0,
        "max_state_replay_error": max_state_error,
        "max_proprio_replay_error": max_proprio_error,
        "pcd_shape": list(converted["images"].shape),
        "pcd_finite": True,
        "frame_stats": stats,
    }
    return converted, report
