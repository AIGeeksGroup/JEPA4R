"""Audit model ranking against open-loop LiftCube simulator branches.

The audit deliberately separates frozen world-model scores from simulator truth:
one planning boundary is snapshotted, the complete final CEM population is
scored once, and every candidate is then replayed open-loop from the same
snapshot.  The command-line runner writes JSONL/JSON and selected RGB videos.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# The simulator dependency is intentionally isolated from the world-model env.
# Make a direct invocation as robust as the existing lzh24 shell runners.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    _ROOT.parent / "envs" / "mpc_packages",
    _ROOT.parent / "envs" / "worldmodel" / "lib" / "python3.11" / "site-packages",
    _ROOT,
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from scipy.stats import kendalltau, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(_ROOT))

from data.maniskill import action_stats
from planning.liftcube_goal import GoalSpec, LiftCubeCostWeights, liftcube_trajectory_cost
from planning.latent_cem import CEMConfig, LatentCEMPlanner, rollout_with_aux
from scripts.eval_liftcube_latent_cem import (
    DEFAULT_DATA_ROOT,
    NEUTRAL_OPEN_ACTION,
    _aggregate_clouds,
    build_goal_cloud,
    _get_observation,
    _normalize_proprio_history,
    _preprocess_observation,
    _proprio_enabled,
    _resolve_lift_threshold,
    _resolve_sim_frames_per_control_step,
    extract_proprio,
    make_env,
    summarize_height_trace,
)
from scripts.eval_maniskill_group1_rollout import (
    atomic_write_json,
    load_checkpoint_model,
    sha256_file,
)
from scripts.check_liftcube_mpc_env import _patch_robot_link_ids, configure_vulkan_icd


def _digest_update(digest: "hashlib._Hash", value: Any) -> None:
    """Canonical digest for ndarray/dict/list simulator states and scalars."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_update(digest, repr(key))
            _digest_update(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _digest_update(digest, item)
    elif isinstance(value, (np.generic, int, float, bool, str, bytes, type(None))):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(repr(value).encode("utf-8"))
    else:
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(repr(value).encode("utf-8"))


def _state_digest(state: Any) -> str:
    digest = hashlib.sha256()
    _digest_update(digest, state)
    return digest.hexdigest()


def _states_equivalent(expected: Any, actual: Any, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Compare restored simulator state, allowing SAPIEN float reconstruction."""
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape or expected.dtype.kind != actual.dtype.kind:
            return False
        if expected.dtype.kind in "fc":
            return bool(np.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True))
        return bool(np.array_equal(expected, actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _states_equivalent(expected[key], actual[key], atol=atol, rtol=rtol)
            for key in expected
        )
    if isinstance(expected, (tuple, list)) and isinstance(actual, type(expected)):
        return len(expected) == len(actual) and all(
            _states_equivalent(left, right, atol=atol, rtol=rtol)
            for left, right in zip(expected, actual)
        )
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
        return bool(np.isclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True))
    return expected == actual


def _iter_env_chain(env) -> Iterable[Any]:
    cursor = env
    seen: set[int] = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        yield cursor
        cursor = getattr(cursor, "env", None)


def _copy_rng_state(obj: Any, attr: str) -> Any | None:
    rng = getattr(obj, attr, None)
    if rng is None:
        return None
    try:
        if hasattr(rng, "bit_generator"):
            return copy.deepcopy(rng.bit_generator.state)
        if hasattr(rng, "get_state"):
            return copy.deepcopy(rng.get_state())
    except Exception:
        return None
    return None


def _restore_rng_state(obj: Any, attr: str, state: Any) -> None:
    if state is None:
        return
    rng = getattr(obj, attr, None)
    if rng is None:
        return
    if hasattr(rng, "bit_generator"):
        rng.bit_generator.state = copy.deepcopy(state)
    elif hasattr(rng, "set_state"):
        rng.set_state(copy.deepcopy(state))


@dataclass
class SimulatorSnapshot:
    """Restorable simulator boundary with a digest for branch verification."""

    state: Any
    digest: str
    rng_states: list[tuple[Any, str, Any]] = field(default_factory=list)
    wrapper_counters: list[tuple[Any, str, Any]] = field(default_factory=list)

    @classmethod
    def capture(cls, env) -> "SimulatorSnapshot":
        base = getattr(env, "unwrapped", env)
        if not hasattr(base, "get_state") or not hasattr(base, "set_state"):
            raise TypeError("environment must expose unwrapped get_state/set_state")
        state = copy.deepcopy(base.get_state())
        rng_states: list[tuple[Any, str, Any]] = []
        wrapper_counters: list[tuple[Any, str, Any]] = []
        for obj in _iter_env_chain(env):
            for attr in ("np_random", "_np_random"):
                saved = _copy_rng_state(obj, attr)
                if saved is not None:
                    rng_states.append((obj, attr, saved))
            for attr in ("_elapsed_steps", "elapsed_steps"):
                if hasattr(obj, attr):
                    try:
                        value = getattr(obj, attr)
                        if isinstance(value, (int, np.integer)):
                            wrapper_counters.append((obj, attr, int(value)))
                    except Exception:
                        pass
        return cls(state=state, digest=_state_digest(state), rng_states=rng_states,
                   wrapper_counters=wrapper_counters)

    def restore(self, env) -> None:
        base = getattr(env, "unwrapped", env)
        base.set_state(copy.deepcopy(self.state))
        for obj, attr, value in self.wrapper_counters:
            try:
                setattr(obj, attr, value)
            except Exception:
                pass
        for obj, attr, value in self.rng_states:
            try:
                _restore_rng_state(obj, attr, value)
            except Exception:
                pass
        current_state = base.get_state()
        current = _state_digest(current_state)
        if current != self.digest and not _states_equivalent(self.state, current_state):
            raise RuntimeError(
                f"simulator restore digest mismatch: expected {self.digest}, got {current}"
            )


@dataclass
class CandidateBatch:
    actions: torch.Tensor
    model_cost: torch.Tensor
    model_components: list[dict[str, float]]


@dataclass
class RealTrajectory:
    actions: np.ndarray
    object_pos: np.ndarray
    tcp_xyz: np.ndarray
    heights: np.ndarray
    tcp_object_distances: np.ndarray
    gripper_gap: np.ndarray
    contact_steps: np.ndarray
    penetrations: np.ndarray
    command_gripper: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    command_closed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    actual_gripper_closed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    cube_robot_contact: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    contact_separation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    contact_pairs: list[list[str]] = field(default_factory=list)
    rewards: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    terminated: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    restore_digest: str | None = None


def _tcp_xyz_from_observation(env, observation: Any) -> np.ndarray:
    try:
        value = np.asarray(observation["extra"]["tcp_pose"], dtype=np.float32)[:3]
        if value.shape == (3,) and np.isfinite(value).all():
            return value.copy()
    except (KeyError, TypeError, IndexError):
        pass
    tcp = getattr(getattr(env, "unwrapped", env), "tcp", None)
    if tcp is not None:
        value = np.asarray(tcp.pose.p, dtype=np.float32)
        if value.shape == (3,) and np.isfinite(value).all():
            return value.copy()
    return np.full(3, np.nan, dtype=np.float32)


def _gripper_gap_from_observation(observation: Any) -> float:
    try:
        qpos = np.asarray(observation["agent"]["qpos"], dtype=np.float32)
        if qpos.size >= 2 and np.isfinite(qpos[-2:]).all():
            return float(np.abs(qpos[-2:]).sum())
    except (KeyError, TypeError, IndexError):
        pass
    return float("nan")


def _contact_diagnostics(env) -> tuple[bool, bool, float, float, list[str]]:
    """Return cube contact, cube-robot contact, max penetration, min separation."""
    base = getattr(env, "unwrapped", env)
    scene = getattr(base, "_scene", None)
    if scene is None or not hasattr(scene, "get_contacts"):
        return False, False, float("nan"), float("nan"), []
    robot_names: set[str] = set()
    agent = getattr(base, "agent", None)
    robot = getattr(agent, "robot", None)
    for item in list(getattr(robot, "get_links", lambda: [])() or []):
        robot_names.add(str(getattr(item, "name", "")))
    for item in (getattr(agent, "finger1_link", None), getattr(agent, "finger2_link", None),
                 getattr(base, "tcp", None)):
        if item is not None:
            robot_names.add(str(getattr(item, "name", "")))
    cube_contact = False
    cube_robot_contact = False
    max_penetration = 0.0
    min_separation = float("nan")
    pairs: list[str] = []
    try:
        contacts = scene.get_contacts()
    except Exception:
        return False, False, float("nan"), float("nan"), []
    for contact in contacts:
        actor0 = str(getattr(getattr(contact, "actor0", None), "name", ""))
        actor1 = str(getattr(getattr(contact, "actor1", None), "name", ""))
        if "cube" not in (actor0, actor1):
            continue
        cube_contact = True
        pairs.append(f"{actor0}:{actor1}")
        other = actor1 if actor0 == "cube" else actor0
        cube_robot_contact = cube_robot_contact or other in robot_names or other.startswith("panda_")
        for point in getattr(contact, "points", []) or []:
            try:
                separation = float(point.separation)
            except (AttributeError, TypeError, ValueError):
                continue
            if not np.isfinite(separation):
                continue
            min_separation = separation if not np.isfinite(min_separation) else min(min_separation, separation)
            # SAPIEN ContactPoint.separation was runtime-validated on this host:
            # negative values indicate overlap, positive values indicate a gap.
            max_penetration = max(max_penetration, max(0.0, -separation))
    return cube_contact, cube_robot_contact, max_penetration, min_separation, pairs


def execute_candidate(
    env,
    snapshot: SimulatorSnapshot,
    actions: np.ndarray,
    *,
    closed_gap_threshold: float = 0.02,
) -> RealTrajectory:
    """Restore one boundary and execute exactly H*chunk atomic actions."""
    candidate = np.asarray(actions, dtype=np.float32)
    if candidate.ndim != 3 or candidate.shape[-1] < 2:
        raise ValueError(f"actions must be [H,chunk,A], got {candidate.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate actions must be finite")
    snapshot.restore(env)
    base = getattr(env, "unwrapped", env)
    # Keep the canonical target digest; SAPIEN may reconstruct equivalent float
    # state with sub-ulp differences after set_state().
    restore_digest = snapshot.digest
    object_endpoints: list[np.ndarray] = []
    tcp_xyz: list[np.ndarray] = []
    heights: list[float] = []
    distances: list[float] = []
    gaps: list[float] = []
    contacts: list[bool] = []
    robot_contacts: list[bool] = []
    penetrations: list[float] = []
    separations: list[float] = []
    command_gripper: list[float] = []
    actual_closed: list[bool] = []
    contact_pairs: list[list[str]] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    for chunk in candidate:
        for action in chunk:
            result = env.step(np.asarray(action, dtype=np.float32))
            observation = result[0]
            reward = float(result[1]) if len(result) > 1 else 0.0
            done = bool(result[2]) if len(result) > 2 else False
            if len(result) > 3:
                done = done or bool(result[3])
            tcp = _tcp_xyz_from_observation(env, observation)
            gap = _gripper_gap_from_observation(observation)
            obj = np.asarray(base.obj.pose.p, dtype=np.float32).copy()
            contact, robot_contact, penetration, separation, pairs = _contact_diagnostics(env)
            tcp_xyz.append(tcp)
            heights.append(float(obj[2]))
            distances.append(float(np.linalg.norm(tcp - obj)) if np.isfinite(tcp).all() else float("nan"))
            gaps.append(gap)
            contacts.append(contact)
            robot_contacts.append(robot_contact)
            penetrations.append(penetration)
            separations.append(separation)
            command_gripper.append(float(action[-1]))
            actual_closed.append(bool(np.isfinite(gap) and gap <= closed_gap_threshold))
            contact_pairs.append(pairs)
            rewards.append(reward)
            terminated.append(done)
        object_endpoints.append(np.asarray(base.obj.pose.p, dtype=np.float32).copy())
    return RealTrajectory(
        actions=candidate.copy(),
        object_pos=np.stack(object_endpoints) if object_endpoints else np.empty((0, 3), dtype=np.float32),
        tcp_xyz=np.stack(tcp_xyz) if tcp_xyz else np.empty((0, 3), dtype=np.float32),
        heights=np.asarray(heights, dtype=np.float32),
        tcp_object_distances=np.asarray(distances, dtype=np.float32),
        gripper_gap=np.asarray(gaps, dtype=np.float32),
        contact_steps=np.asarray(contacts, dtype=bool),
        penetrations=np.asarray(penetrations, dtype=np.float32),
        command_gripper=np.asarray(command_gripper, dtype=np.float32),
        command_closed=np.asarray(command_gripper, dtype=np.float32) < 0.0,
        actual_gripper_closed=np.asarray(actual_closed, dtype=bool),
        cube_robot_contact=np.asarray(robot_contacts, dtype=bool),
        contact_separation=np.asarray(separations, dtype=np.float32),
        contact_pairs=contact_pairs,
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=bool),
        restore_digest=restore_digest,
    )


def compute_real_liftcube_cost(
    trajectory: RealTrajectory,
    *,
    initial_cube_height: float,
    lift_threshold: float,
    hold_steps: int = 10,
    height_tolerance: float = 0.01,
    approach_radius: float = 0.04,
    gripper_closed_threshold: float = 0.02,
    weights: LiftCubeCostWeights | None = None,
) -> dict[str, Any]:
    """Compute a simulator-truth physical objective at the model's endpoints."""
    if hold_steps < 1 or height_tolerance < 0 or approach_radius < 0:
        raise ValueError("hold_steps must be positive and tolerances non-negative")
    heights = np.asarray(trajectory.heights, dtype=np.float64).reshape(-1)
    endpoints = np.asarray(trajectory.object_pos, dtype=np.float64)
    endpoint_heights = endpoints[:, 2] if endpoints.ndim == 2 and endpoints.shape[1] == 3 else np.empty(0)
    endpoint_count = int(len(endpoint_heights))
    final_height = float(endpoint_heights[-1]) if endpoint_count else float(heights[-1]) if heights.size else float("nan")
    terminal_deficit = max(0.0, float(lift_threshold) - final_height) if np.isfinite(final_height) else float("inf")
    height_cost = terminal_deficit * terminal_deficit
    chunk_size = max(1, int(trajectory.actions.shape[1]))
    required_nodes = int(math.ceil(hold_steps / chunk_size) + 1)
    endpoint_trace = np.concatenate([[float(initial_cube_height)], endpoint_heights])
    tail = endpoint_trace[-required_nodes:] if endpoint_trace.size else np.empty(0)
    hold_deficits = np.maximum(0.0, float(lift_threshold) - tail)
    hold_cost = float(np.mean(hold_deficits * hold_deficits)) if hold_deficits.size else float("inf")
    if tail.size >= 2:
        delta = np.diff(tail)
        stability_cost = float(np.mean(np.maximum(0.0, np.abs(delta) - height_tolerance) ** 2)
                               + np.mean(np.maximum(0.0, -delta) ** 2))
    else:
        stability_cost = 0.0
    distance = np.asarray(trajectory.tcp_object_distances, dtype=np.float64)
    finite_distance = distance[np.isfinite(distance)]
    endpoint_indices = np.arange(chunk_size - 1, len(distance), chunk_size)
    endpoint_distance = distance[endpoint_indices]
    endpoint_pre_lift = endpoint_heights < float(lift_threshold)
    valid_distance = np.isfinite(endpoint_distance) & endpoint_pre_lift
    beyond = np.maximum(0.0, endpoint_distance - float(approach_radius)) ** 2
    approach_cost = float(np.mean(beyond[valid_distance])) if np.any(valid_distance) else 0.0
    close_amount = np.clip(
        (1.0 - np.asarray(trajectory.actions[..., -1], dtype=np.float64)) / 2.0,
        0.0,
        1.0,
    ).mean(axis=1)
    early_close_cost = float(np.mean(close_amount[endpoint_pre_lift] * beyond[endpoint_pre_lift])) if np.any(endpoint_pre_lift) else 0.0
    endpoint_gap = np.asarray(trajectory.gripper_gap, dtype=np.float64)[endpoint_indices]
    lifted = endpoint_heights > float(initial_cube_height) + float(height_tolerance)
    near = endpoint_distance <= float(approach_radius)
    gripper_required = near | lifted
    gripper_error = np.maximum(0.0, endpoint_gap - float(gripper_closed_threshold)) ** 2
    gripper_cost = float(np.mean(gripper_error[gripper_required])) if np.any(gripper_required) else 0.0
    endpoint_closed = endpoint_gap <= float(gripper_closed_threshold)
    contact_trace = np.asarray(trajectory.cube_robot_contact, dtype=bool)
    if contact_trace.shape != distance.shape:
        contact_trace = np.zeros(distance.shape, dtype=bool)
    endpoint_contact = contact_trace[endpoint_indices]
    contact_required = (near & endpoint_closed) | lifted
    contact_cost = float(np.mean((~endpoint_contact[contact_required]).astype(np.float64))) if np.any(contact_required) else 0.0
    resolved_weights = weights or LiftCubeCostWeights()
    total = (
        resolved_weights.height * height_cost
        + resolved_weights.hold * hold_cost
        + resolved_weights.stability * stability_cost
        + resolved_weights.approach * approach_cost
        + resolved_weights.early_close * early_close_cost
        + resolved_weights.gripper * gripper_cost
        + resolved_weights.contact * contact_cost
    )
    summary = summarize_height_trace(
        heights,
        cube_z0=float(initial_cube_height),
        lift_threshold=float(lift_threshold),
        hold_steps=int(hold_steps),
        height_tolerance=float(height_tolerance),
    )
    return {
        "total": float(total),
        "height_cost": float(height_cost),
        "hold_cost": float(hold_cost),
        "stability_cost": float(stability_cost),
        "approach_cost": float(approach_cost),
        "early_close_cost": float(early_close_cost),
        "gripper_cost": float(gripper_cost),
        "contact_cost": float(contact_cost),
        "min_tcp_object_distance": float(np.min(finite_distance)) if finite_distance.size else float("nan"),
        "median_tcp_object_distance": float(np.median(finite_distance)) if finite_distance.size else float("nan"),
        "max_tcp_object_distance": float(np.max(finite_distance)) if finite_distance.size else float("nan"),
        "object_pos_endpoint_count": endpoint_count,
        "final_cube_height": final_height,
        "max_cube_height": float(np.nanmax(heights)) if heights.size and np.isfinite(heights).any() else float("nan"),
        "contact_steps": int(np.count_nonzero(trajectory.contact_steps)),
        "cube_robot_contact_steps": int(np.count_nonzero(trajectory.cube_robot_contact)),
        "max_penetration": float(np.nanmax(trajectory.penetrations)) if trajectory.penetrations.size and np.isfinite(trajectory.penetrations).any() else float("nan"),
        "min_contact_separation": float(np.nanmin(trajectory.contact_separation)) if trajectory.contact_separation.size and np.isfinite(trajectory.contact_separation).any() else float("nan"),
        "command_closed_first_step": int(np.flatnonzero(trajectory.command_closed)[0] + 1) if np.any(trajectory.command_closed) else None,
        "actual_closed_first_step": int(np.flatnonzero(trajectory.actual_gripper_closed)[0] + 1) if np.any(trajectory.actual_gripper_closed) else None,
        "physical_success": bool(summary["success"]),
        "height_summary": summary,
    }


def summarize_candidate_ranking(
    model_cost: np.ndarray | list[float],
    real_cost: np.ndarray | list[float],
    top_k: tuple[int, ...] = (1, 8, 16),
) -> dict[str, Any]:
    """Compute per-state lower-is-better rank metrics with tie-safe statistics."""
    model = np.asarray(model_cost, dtype=np.float64).reshape(-1)
    real = np.asarray(real_cost, dtype=np.float64).reshape(-1)
    if model.shape != real.shape or model.size < 2:
        raise ValueError("model_cost and real_cost must have the same length >= 2")
    finite = np.isfinite(model) & np.isfinite(real)
    result: dict[str, Any] = {
        "valid": False,
        "reason": None,
        "spearman": None,
        "kendall_tau_b": None,
        "top_k": {},
        "regret_at_1": None,
        "real_cost_variance": float(np.var(real[finite])) if np.any(finite) else float("nan"),
    }
    if not finite.all():
        result["reason"] = "non_finite_cost"
    elif np.ptp(real) <= max(1e-8, 1e-6 * max(1.0, float(np.max(np.abs(real))))):
        result["reason"] = "low_real_cost_variance"
    else:
        spear = spearmanr(model, real)
        tau = kendalltau(model, real, variant="b")
        result["spearman"] = float(spear.statistic) if np.isfinite(spear.statistic) else None
        result["kendall_tau_b"] = float(tau.statistic) if np.isfinite(tau.statistic) else None
        result["valid"] = result["spearman"] is not None and result["kendall_tau_b"] is not None
    model_order = np.argsort(model, kind="stable")
    real_order = np.argsort(real, kind="stable")
    oracle = float(np.min(real[finite])) if np.any(finite) else float("nan")
    for requested in top_k:
        k = max(1, min(int(requested), model.size))
        mset = set(int(x) for x in model_order[:k])
        rset = set(int(x) for x in real_order[:k])
        overlap = len(mset & rset)
        result["top_k"][str(requested)] = {
            "k": k,
            "overlap": int(overlap),
            "overlap_rate": float(overlap / k),
            "jaccard": float(overlap / len(mset | rset)) if mset | rset else 1.0,
            "model_indices": [int(x) for x in model_order[:k]],
            "real_indices": [int(x) for x in real_order[:k]],
            "regret": float(np.min(real[model_order[:k]]) - oracle),
        }
    result["regret_at_1"] = float(real[model_order[0]] - oracle)
    result["model_top1"] = int(model_order[0])
    result["real_top1"] = int(real_order[0])
    result["oracle_cost"] = oracle
    return result


def _context_at_boundary(env, model, *, env_seed: int, device: torch.device):
    reset_result = env.reset(seed=int(env_seed))
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    chunk_size = int(model.cfg.skip_frame)
    point_count = int(model.cfg.K)
    sample_key = int(env_seed)
    enabled = _proprio_enabled(model)
    clouds = [_preprocess_observation(observation, sample_key=sample_key, point_count=point_count)]
    proprio_values = [extract_proprio(observation)] if enabled else []
    warmup_actions: list[np.ndarray] = []
    for _ in range(int(model.cfg.T) - 1):
        for _ in range(chunk_size):
            observation, *_ = env.step(NEUTRAL_OPEN_ACTION.copy())
            warmup_actions.append(NEUTRAL_OPEN_ACTION.copy())
        clouds.append(_preprocess_observation(observation, sample_key=sample_key, point_count=point_count))
        if enabled:
            proprio_values.append(extract_proprio(observation))
    latent_history = _aggregate_clouds(model, clouds)
    context_actions = torch.as_tensor(np.stack(warmup_actions), device=device, dtype=torch.float32).unsqueeze(0)
    proprio_history = _normalize_proprio_history(model, proprio_values) if enabled else None
    cube_z0 = float(np.asarray(env.unwrapped.obj.pose.p)[2])
    threshold = _resolve_lift_threshold(env)
    return latent_history, context_actions, proprio_history, cube_z0, threshold


@torch.no_grad()
def capture_cem_candidates(
    model,
    planner: LatentCEMPlanner,
    env,
    snapshot: SimulatorSnapshot,
    *,
    latent_history: torch.Tensor,
    context_actions: torch.Tensor,
    proprio_history: torch.Tensor,
    goal_spec: GoalSpec,
    seed: int,
) -> CandidateBatch:
    """Run CEM once and return the complete final population plus components."""
    del env, snapshot
    _, candidates, scores = planner.plan_with_goal_population(
        model,
        latent_history,
        proprio_history,
        context_actions,
        goal_spec,
        env="maniskill2",
        seed=int(seed),
    )
    components: list[dict[str, float]] = []
    for action_batch in candidates.split(planner.config.model_batch_size):
        rollout = rollout_with_aux(
            model, latent_history, proprio_history, context_actions,
            action_batch, "maniskill2"
        )
        cost = liftcube_trajectory_cost(
            rollout, action_batch, goal_spec, chunk_size=planner.config.chunk_size
        )
        for index in range(action_batch.shape[0]):
            components.append({name: float(value[index].detach().cpu()) for name, value in cost.components.items()})
    return CandidateBatch(actions=candidates.detach(), model_cost=scores.detach(), model_components=components)


def _render_candidate(env, snapshot: SimulatorSnapshot, actions: np.ndarray, path: Path, *, fps: int = 20) -> dict[str, Any]:
    """Write an RGB mp4 when ffmpeg works, otherwise preserve an npz frame artifact."""
    snapshot.restore(env)
    frames: list[np.ndarray] = []
    try:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8).copy())
    except Exception:
        frame = None
    for chunk in actions:
        for action in chunk:
            env.step(np.asarray(action, dtype=np.float32))
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8).copy())
            except Exception:
                pass
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return {"path": None, "format": None, "frame_count": 0}
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        return {"path": str(path), "format": "mp4", "frame_count": len(frames)}
    except Exception as error:
        fallback = path.with_suffix(".npz")
        np.savez_compressed(fallback, frames=np.stack(frames))
        return {"path": str(fallback), "format": "npz", "frame_count": len(frames), "video_error": repr(error)}


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def _bootstrap_mean(values: list[float], *, seed: int, samples: int = 2000) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "low": None, "high": None, "count": 0}
    if finite.size == 1:
        value = float(finite[0])
        return {"mean": value, "low": value, "high": value, "count": 1}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, finite.size, size=(samples, finite.size))
    means = finite[draws].mean(axis=1)
    return {"mean": float(finite.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)), "count": int(finite.size)}


def _make_audit_env(render: bool):
    if not render:
        return make_env()
    configure_vulkan_icd()
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401
    from mani_skill2.utils.wrappers.observation import BaseGymObservationWrapper
    _patch_robot_link_ids(BaseGymObservationWrapper)
    return gym.make("LiftCube-v0", obs_mode="pointcloud_robot_seg",
                    control_mode="pd_ee_delta_pose", render_mode="rgb_array")


def run_audit(
    *,
    checkpoint: Path,
    output: Path,
    data_root: Path,
    state_seeds: list[int],
    candidates: int,
    model_batch_size: int,
    iterations: int,
    elites: int,
    horizon: int,
    chunk_size: int,
    device: str,
    render: bool,
    hold_steps: int = 10,
    height_tolerance: float = 0.01,
    approach_radius: float = 0.04,
    gripper_closed_threshold: float = 0.02,
    cost_weights: LiftCubeCostWeights | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    model, state = load_checkpoint_model(checkpoint, device_obj)
    mean, std = action_stats(data_root, split="train")
    config = CEMConfig(horizon=horizon, chunk_size=chunk_size, action_dim=len(mean),
                       candidates=candidates, model_batch_size=model_batch_size,
                       iterations=iterations, elites=elites)
    planner = LatentCEMPlanner(config, mean, std)
    env = _make_audit_env(render)
    weight = cost_weights or LiftCubeCostWeights(gripper=0.1, contact=0.05)
    rows: list[dict[str, Any]] = []
    video_manifest: list[dict[str, Any]] = []
    start_time = time.time()
    try:
        with (output / "states.jsonl").open("x", encoding="utf-8") as handle:
            for state_index, env_seed in enumerate(state_seeds):
                latent, context_actions, proprio, cube_z0, threshold = _context_at_boundary(
                    env, model, env_seed=env_seed, device=device_obj
                )
                if proprio is None:
                    raise RuntimeError("ranking audit requires the proprio-enabled checkpoint")
                goal_cloud = build_goal_cloud(
                    env, sample_key=int(env_seed), point_count=int(model.cfg.K)
                )
                goal_latent = _aggregate_clouds(model, [goal_cloud])[:, 0]
                goal = GoalSpec(initial_cube_height=cube_z0, lift_threshold=threshold,
                                hold_control_steps=hold_steps, height_tolerance=height_tolerance,
                                approach_radius=approach_radius,
                                gripper_closed_threshold=gripper_closed_threshold,
                                demonstrated_goal_latent=goal_latent, weights=weight)
                snapshot = SimulatorSnapshot.capture(env)
                batch = capture_cem_candidates(model, planner, env, snapshot,
                                               latent_history=latent, context_actions=context_actions,
                                               proprio_history=proprio, goal_spec=goal,
                                               seed=31000 + state_index)
                trajectories: list[RealTrajectory] = []
                real_costs: list[float] = []
                real_details: list[dict[str, Any]] = []
                for candidate_index in range(batch.actions.shape[0]):
                    trajectory = execute_candidate(env, snapshot,
                                                   batch.actions[candidate_index].detach().cpu().numpy())
                    detail = compute_real_liftcube_cost(
                        trajectory, initial_cube_height=cube_z0, lift_threshold=threshold,
                        hold_steps=hold_steps, height_tolerance=height_tolerance,
                        approach_radius=approach_radius,
                        gripper_closed_threshold=gripper_closed_threshold,
                        weights=weight,
                    )
                    trajectories.append(trajectory)
                    real_costs.append(float(detail["total"]))
                    real_details.append(detail)
                ranking = summarize_candidate_ranking(batch.model_cost.cpu().numpy(), real_costs)
                selected = {
                    "model_top1": int(ranking["model_top1"]),
                    "sim_top1": int(ranking["real_top1"]),
                    "median_model_rank": int(np.argsort(batch.model_cost.cpu().numpy(), kind="stable")[len(real_costs) // 2]),
                }
                if render:
                    for label, candidate_index in selected.items():
                        artifact = _render_candidate(
                            env, snapshot, batch.actions[candidate_index].detach().cpu().numpy(),
                            output / "videos" / f"state_{state_index:03d}_seed_{env_seed}_{label}_candidate_{candidate_index:03d}.mp4",
                        )
                        video_manifest.append({"state_index": state_index, "env_seed": env_seed,
                                               "label": label, "candidate_index": candidate_index,
                                               "restore_digest": snapshot.digest, **artifact})
                row = {
                    "state_index": state_index,
                    "env_seed": int(env_seed),
                    "restore_digest": snapshot.digest,
                    "initial_cube_height": cube_z0,
                    "lift_threshold": threshold,
                    "model_cost": _serializable(batch.model_cost),
                    "real_cost": real_costs,
                    "model_components": _serializable(batch.model_components),
                    "real_details": real_details,
                    "ranking": ranking,
                    "candidate_actions": _serializable(batch.actions),
                    "restore_digests": [trajectory.restore_digest for trajectory in trajectories],
                    "selected": selected,
                }
                rows.append(row)
                handle.write(json.dumps(_serializable(row), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        env.close()
    valid = [row["ranking"] for row in rows if row["ranking"]["valid"]]
    summary = {
        "schema_version": 1,
        "task": "LiftCube-v0",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), "step": int(state["step"])},
        "data_root": str(data_root.resolve()),
        "state_count": len(rows),
        "candidate_count": candidates,
        "cem": _serializable(vars(config)),
        "goal_cost": {
            "hold_steps": int(hold_steps),
            "height_tolerance": float(height_tolerance),
            "approach_radius": float(approach_radius),
            "gripper_closed_threshold": float(gripper_closed_threshold),
            "weights": weight.to_dict(),
            "goal_latent_source": "simulator goal cloud encoded by frozen spatial aggregator",
        },
        "valid_rank_state_count": len(valid),
        "low_variance_state_count": sum(row["ranking"]["reason"] == "low_real_cost_variance" for row in rows),
        "rank_metrics": {
            "spearman": _bootstrap_mean([float(row["spearman"]) for row in valid], seed=4101),
            "kendall_tau_b": _bootstrap_mean([float(row["kendall_tau_b"]) for row in valid], seed=4102),
            "regret_at_1": _bootstrap_mean([float(row["ranking"]["regret_at_1"]) for row in rows], seed=4103),
            "top1_overlap_rate": _bootstrap_mean([float(row["ranking"]["top_k"]["1"]["overlap_rate"]) for row in rows], seed=4104),
        },
        "physical_success_count": int(sum(bool(detail["physical_success"]) for row in rows for detail in row["real_details"])),
        "physical_success_candidate_total": int(sum(len(row["real_details"]) for row in rows)),
        "video_manifest": video_manifest,
        "elapsed_sec": float(time.time() - start_time),
        "states": rows,
    }
    atomic_write_json(output / "summary.json", _serializable(summary))
    atomic_write_json(output / "video_manifest.json", _serializable({"videos": video_manifest}))
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--state-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gripper-closed-threshold", type=float, default=0.02)
    parser.add_argument("--weight-height", type=float, default=1.0)
    parser.add_argument("--weight-hold", type=float, default=1.0)
    parser.add_argument("--weight-stability", type=float, default=0.5)
    parser.add_argument("--weight-latent", type=float, default=0.05)
    parser.add_argument("--weight-approach", type=float, default=0.1)
    parser.add_argument("--weight-early-close", type=float, default=0.1)
    parser.add_argument("--weight-gripper", type=float, default=0.1)
    parser.add_argument("--weight-contact", type=float, default=0.05)
    args = parser.parse_args(argv)
    cost_weights = LiftCubeCostWeights(
        height=args.weight_height,
        hold=args.weight_hold,
        stability=args.weight_stability,
        latent=args.weight_latent,
        approach=args.weight_approach,
        early_close=args.weight_early_close,
        gripper=args.weight_gripper,
        contact=args.weight_contact,
    )
    summary = run_audit(
        checkpoint=args.checkpoint, output=args.output, data_root=args.data_root,
        state_seeds=args.state_seeds, candidates=args.candidates,
        model_batch_size=args.model_batch_size, iterations=args.iterations,
        elites=args.elites, horizon=args.horizon, chunk_size=args.chunk_size,
        device=args.device, render=args.render,
        gripper_closed_threshold=args.gripper_closed_threshold,
        cost_weights=cost_weights,
    )
    print(json.dumps(_serializable(summary), sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
