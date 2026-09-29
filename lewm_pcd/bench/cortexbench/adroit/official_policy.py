"""Official CortexBench Adroit Pen demonstration-backed Backbone policy.

The official Adroit package contains 100 expert trajectories.  This adapter stores
only their target axes and actions in a compact NPZ and selects the trajectory
whose target direction is closest to the current Pen target direction.  Actions
are clipped to the normalized [-1, 1] contract before being returned; the
official Pen environment performs the native actuator scaling internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np


DEFAULT_POLICY_PATH = Path(
    os.environ.get(
        "CORTEXBENCH_ADROIT_PEN_POLICY",
        "/root/autodl-tmp/workspace/users/lewm/cortexbench-official-data/adroit/pen_policy_actions.npz",
    )
)


def normalize_official_action(raw) -> np.ndarray:
    """Convert an official policy action to the normalized environment contract."""
    action = np.asarray(raw, dtype=np.float32)
    if action.ndim != 1:
        raise ValueError(f"official action must be 1D, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("official action contains NaN or Inf")
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def _unit_axis(axis: np.ndarray) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"target axis must be finite and nonzero, got {value}")
    return value / norm


def select_demo_for_target_axis(target_axis, demo_axes) -> int:
    """Return the official demo with maximum target-axis cosine similarity."""
    target = _unit_axis(np.asarray(target_axis))
    axes = np.asarray(demo_axes, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1:] != (3,):
        raise ValueError(f"demo_axes must have shape (N, 3), got {axes.shape}")
    norms = np.linalg.norm(axes, axis=1)
    if not np.isfinite(axes).all() or np.any(norms <= 1e-12):
        raise ValueError("demo_axes contains invalid axes")
    scores = (axes / norms[:, None]) @ target
    return int(np.argmax(scores))


def target_axis_from_env(env) -> np.ndarray:
    """Read the task target direction from the official Pen environment."""
    top = np.asarray(env.data.site_xpos[env.tar_t_sid], dtype=np.float64)
    bottom = np.asarray(env.data.site_xpos[env.tar_b_sid], dtype=np.float64)
    return _unit_axis(top - bottom)


@dataclass(frozen=True)
class OfficialPenPolicy:
    target_axes: np.ndarray
    actions: np.ndarray

    def __post_init__(self):
        axes = np.asarray(self.target_axes, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        if axes.ndim != 2 or axes.shape[1:] != (3,):
            raise ValueError(f"target_axes must have shape (N, 3), got {axes.shape}")
        if actions.ndim != 3 or actions.shape[0] != axes.shape[0]:
            raise ValueError(f"actions must have shape (N, T, D), got {actions.shape}")
        if not np.isfinite(axes).all() or not np.isfinite(actions).all():
            raise ValueError("official policy data contains NaN or Inf")
        object.__setattr__(self, "target_axes", axes)
        object.__setattr__(self, "actions", actions)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "OfficialPenPolicy":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"official Pen policy archive not found: {path}; "
                "create it from the official pen-v0.pickle first"
            )
        with np.load(path, allow_pickle=False) as data:
            return cls(data["target_axes"], data["actions"])

    @property
    def episode_len(self) -> int:
        return int(self.actions.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[2])

    def action(self, env, step: int) -> np.ndarray:
        """Return the clipped official action for the current target and timestep."""
        demo = select_demo_for_target_axis(target_axis_from_env(env), self.target_axes)
        timestep = min(max(int(step), 0), self.episode_len - 1)
        return normalize_official_action(self.actions[demo, timestep])