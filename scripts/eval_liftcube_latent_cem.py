"""Evaluate a frozen LiftCube world model with pure latent-cost CEM."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.maniskill import action_stats
from data.maniskill_online import preprocess_online_cloud
from planning.liftcube_goal import (
    GoalSpec,
    LiftCubeCostWeights,
    liftcube_trajectory_cost,
)
from planning.latent_cem import (
    CEMConfig,
    LatentCEMPlanner,
    latent_trajectory_cost,
    normalized_latent_mse,
    rollout_from_latent,
    rollout_with_aux,
)
from scripts.check_liftcube_mpc_env import (
    _patch_robot_link_ids,
    configure_vulkan_icd,
)
from scripts.eval_maniskill_group1_rollout import (
    LiftCubeRolloutDataset,
    atomic_write_json,
    load_checkpoint_model,
    sha256_file,
)


DEFAULT_DATA_ROOT = Path(
    "/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0"
)
NEUTRAL_OPEN_ACTION = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
DEFAULT_HOLD_STEPS = 10
DEFAULT_HEIGHT_TOLERANCE = 0.01


def _proprio_enabled(model) -> bool:
    return bool(
        getattr(getattr(model.cfg, "proprio", None), "enabled", False)
    )


def extract_proprio(observation: dict[str, Any]) -> np.ndarray:
    """Extract qpos, qvel and TCP pose in the manifest's fixed 25-D layout."""
    try:
        qpos = np.asarray(observation["agent"]["qpos"], dtype=np.float32)
        qvel = np.asarray(observation["agent"]["qvel"], dtype=np.float32)
        tcp_pose = np.asarray(observation["extra"]["tcp_pose"], dtype=np.float32)
    except (KeyError, TypeError) as error:
        raise ValueError("observation is missing qpos, qvel, or tcp_pose") from error
    expected = {"qpos": (9,), "qvel": (9,), "tcp_pose": (7,)}
    for name, value in (
        ("qpos", qpos),
        ("qvel", qvel),
        ("tcp_pose", tcp_pose),
    ):
        if value.shape != expected[name] or not np.isfinite(value).all():
            raise ValueError(
                f"online {name} must be finite with shape {expected[name]}, "
                f"got {value.shape}"
            )
    return np.ascontiguousarray(
        np.concatenate([qpos, qvel, tcp_pose]), dtype=np.float32
    )


def make_env(task: str = "LiftCube-v0"):
    """Create the renderer-backed point-cloud environment used by MPC."""
    configure_vulkan_icd()
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401
    from mani_skill2.utils.wrappers.observation import BaseGymObservationWrapper

    _patch_robot_link_ids(BaseGymObservationWrapper)
    return gym.make(
        task,
        obs_mode="pointcloud_robot_seg",
        control_mode="pd_ee_delta_pose",
    )


def _pointcloud(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cloud = observation.get("pointcloud", observation)
    rgb = None if "rgb" not in cloud else np.asarray(cloud["rgb"])
    return np.asarray(cloud["xyzw"]), np.asarray(cloud["robot_seg"]), rgb


def _get_observation(env) -> dict[str, Any]:
    observation = env.unwrapped.get_obs()
    transforms = []
    cursor = env
    while cursor is not env.unwrapped and hasattr(cursor, "env"):
        transform = getattr(cursor, "observation", None)
        if callable(transform):
            transforms.append(transform)
        cursor = cursor.env
    for transform in reversed(transforms):
        observation = transform(observation)
    return observation


def _preprocess_observation(
    observation: dict[str, Any], *, sample_key: int, point_count: int
) -> np.ndarray:
    xyzw, robot_seg, rgb = _pointcloud(observation)
    return preprocess_online_cloud(
        xyzw,
        robot_seg,
        sample_key=sample_key,
        point_count=point_count,
        rgb=rgb,
    )


def build_goal_cloud(
    env, *, sample_key: int, point_count: int = 512
) -> np.ndarray:
    """Render the cube at the task goal, then restore the complete simulator state."""
    base = env.unwrapped
    state = copy.deepcopy(base.get_state())
    current_pose = base.obj.pose
    try:
        base.obj.set_pose(type(current_pose)(base.goal_pos, current_pose.q))
        observation = _get_observation(env)
        return _preprocess_observation(
            observation,
            sample_key=sample_key,
            point_count=point_count,
        )
    finally:
        base.set_state(state)


def _aggregate_clouds(model, clouds: list[np.ndarray]) -> torch.Tensor:
    device = next(model.parameters()).device
    points = torch.as_tensor(
        np.stack(clouds), device=device, dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        return model._aggregate(points)


def _zero_action_cost(
    model,
    latent_history: torch.Tensor,
    context_actions: torch.Tensor,
    goal_latent: torch.Tensor,
    planner,
) -> float:
    cfg = planner.config
    zero = latent_history.new_zeros(
        (1, cfg.horizon, cfg.chunk_size, cfg.action_dim)
    )
    rollout = rollout_from_latent(
        model, latent_history, context_actions, zero, "maniskill2"
    )
    cost = latent_trajectory_cost(
        rollout,
        goal_latent,
        trajectory_weight=cfg.trajectory_weight,
    )
    return float(cost.item())


def _normalize_proprio_history(model, values: list[np.ndarray]) -> torch.Tensor:
    if model.proprio_encoder is None:
        raise RuntimeError("enabled model is missing proprio_encoder")
    device = next(model.parameters()).device
    raw = torch.as_tensor(
        np.stack(values), device=device, dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        return model.proprio_encoder.normalize(raw)


def _zero_action_cost_with_goal(
    model,
    latent_history: torch.Tensor,
    proprio_history_normalized: torch.Tensor,
    context_actions: torch.Tensor,
    goal_spec: GoalSpec,
    planner,
) -> float:
    cfg = planner.config
    zero = latent_history.new_zeros(
        (1, cfg.horizon, cfg.chunk_size, cfg.action_dim)
    )
    zero[..., -1] = 1.0
    rollout = rollout_with_aux(
        model,
        latent_history,
        proprio_history_normalized,
        context_actions,
        zero,
        "maniskill2",
    )
    cost = liftcube_trajectory_cost(
        rollout, zero, goal_spec, chunk_size=cfg.chunk_size
    )
    return float(cost.total.item())


def _predicted_success(
    heights: list[float], goal: GoalSpec, *, chunk_size: int
) -> bool:
    required = math.ceil(goal.hold_control_steps / chunk_size) + 1
    values = np.asarray(heights, dtype=np.float64)
    if values.size < required or not np.isfinite(values).all():
        return False
    tail = values[-required:]
    return bool(
        np.all(tail >= goal.lift_threshold)
        and np.all(np.abs(np.diff(tail)) <= goal.height_tolerance)
    )


def _cube_height(env) -> float:
    return float(np.asarray(env.unwrapped.obj.pose.p)[2])


def _resolve_lift_threshold(env) -> float:
    base = env.unwrapped
    explicit = getattr(base, "lift_threshold", None)
    if explicit is not None:
        threshold = float(explicit)
    else:
        goal_height = getattr(base, "goal_height", None)
        cube_half_size = getattr(base, "cube_half_size", None)
        if goal_height is None or cube_half_size is None:
            raise AttributeError(
                "LiftCube environment must expose goal_height and cube_half_size "
                "or an explicit lift_threshold"
            )
        threshold = float(goal_height) + float(np.asarray(cube_half_size)[2])
    if not np.isfinite(threshold):
        raise ValueError(f"lift threshold must be finite, got {threshold}")
    return threshold


def _resolve_sim_frames_per_control_step(env) -> int:
    base = env.unwrapped
    explicit = getattr(base, "_sim_steps_per_control", None)
    if explicit is not None:
        value = int(explicit)
    else:
        sim_freq = getattr(base, "sim_freq", None)
        control_freq = getattr(base, "control_freq", None)
        if sim_freq is not None and control_freq is not None:
            value = int(round(float(sim_freq) / float(control_freq)))
        else:
            value = 1
    if value < 1:
        raise ValueError(f"sim frames per control step must be positive, got {value}")
    return value


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.astype(bool, copy=False):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def summarize_height_trace(
    height_trace: list[float] | np.ndarray,
    *,
    cube_z0: float,
    lift_threshold: float,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    height_tolerance: float = DEFAULT_HEIGHT_TOLERANCE,
) -> dict[str, object]:
    """Summarize peak-and-hold LiftCube height success conservatively."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    if height_tolerance < 0 or not np.isfinite(height_tolerance):
        raise ValueError("height_tolerance must be finite and non-negative")
    if not np.isfinite(lift_threshold):
        raise ValueError("lift_threshold must be finite")

    values = np.asarray(height_trace, dtype=np.float64).reshape(-1)
    initial = float(cube_z0)
    has_nan = bool(not np.isfinite(values).all() or not np.isfinite(initial))
    final = float(values[-1]) if values.size else float("nan")
    max_height = initial if np.isfinite(initial) else float("nan")
    max_height_step = 0
    for index, value in enumerate(values):
        if np.isfinite(value) and (not np.isfinite(max_height) or value > max_height):
            max_height = float(value)
            max_height_step = index + 1

    max_height_delta = (
        float(max_height - initial)
        if np.isfinite(max_height) and np.isfinite(initial)
        else float("nan")
    )
    above_threshold = np.isfinite(values) & (values >= lift_threshold)
    max_consecutive_above = _longest_true_run(above_threshold)
    peak_reached = bool(np.isfinite(max_height) and max_height >= lift_threshold)
    stable_peak = False
    if peak_reached and max_height_step > 0:
        peak_index = max_height_step - 1
        start = max(0, peak_index - 2)
        stop = min(values.size, peak_index + 3)
        peak_region = values[start:stop]
        stable_peak = bool(
            peak_region.size > 0
            and np.isfinite(peak_region).all()
            and float(peak_region.max() - peak_region.min()) < height_tolerance
        )
    success_by_peak_and_hold = bool(
        peak_reached and max_consecutive_above >= int(hold_steps)
    )
    success = bool(
        success_by_peak_and_hold and stable_peak and not has_nan
    )
    return {
        "initial_cube_height": initial,
        "max_cube_height": float(max_height),
        "max_height_delta": max_height_delta,
        "max_height_step": int(max_height_step),
        "max_height_step_unit": "control_step",
        "final_cube_height": final,
        "lift_threshold": float(lift_threshold),
        "hold_steps": int(hold_steps),
        "height_tolerance": float(height_tolerance),
        "max_consecutive_above": int(max_consecutive_above),
        "peak_reached": peak_reached,
        "success_by_peak": peak_reached,
        "success_by_peak_and_hold": success_by_peak_and_hold,
        "stable_peak": stable_peak,
        "has_nan": has_nan,
        "success": success,
    }


def _step(env, action: np.ndarray):
    observation, reward, terminated, truncated, info = env.step(action)
    return (
        observation,
        float(reward),
        bool(terminated),
        bool(truncated),
        dict(info),
    )


def run_episode(
    env,
    model,
    planner,
    *,
    env_seed: int,
    cem_seed: int,
    max_steps: int,
    lift_threshold: float | None = None,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    height_tolerance: float = DEFAULT_HEIGHT_TOLERANCE,
    approach_radius: float = 0.04,
    gripper_closed_threshold: float = 0.02,
    cost_weights: LiftCubeCostWeights | None = None,
) -> dict[str, object]:
    """Run skip-10 receding-horizon control for one reset seed."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    if height_tolerance < 0 or not np.isfinite(height_tolerance):
        raise ValueError("height_tolerance must be finite and non-negative")
    chunk_size = int(planner.config.chunk_size)
    if chunk_size != int(model.cfg.skip_frame):
        raise ValueError("planner chunk_size must match model skip_frame")
    if int(planner.config.action_dim) != int(env.action_space.shape[0]):
        raise ValueError("planner action_dim must match the environment")

    reset_result = env.reset(seed=int(env_seed))
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    cube_z0 = _cube_height(env)
    resolved_lift_threshold = (
        _resolve_lift_threshold(env)
        if lift_threshold is None
        else float(lift_threshold)
    )
    if not np.isfinite(resolved_lift_threshold):
        raise ValueError("lift_threshold must be finite")
    sim_frames_per_control_step = _resolve_sim_frames_per_control_step(env)
    point_count = int(model.cfg.K)
    sample_key = int(env_seed)
    enabled = _proprio_enabled(model)
    goal_cloud = build_goal_cloud(
        env, sample_key=sample_key, point_count=point_count
    )
    goal_latent = _aggregate_clouds(model, [goal_cloud])[:, 0]
    goal_spec = None
    if enabled:
        goal_spec = GoalSpec(
            initial_cube_height=cube_z0,
            lift_threshold=resolved_lift_threshold,
            hold_control_steps=hold_steps,
            height_tolerance=height_tolerance,
            approach_radius=approach_radius,
            gripper_closed_threshold=gripper_closed_threshold,
            demonstrated_goal_latent=goal_latent,
            weights=cost_weights or LiftCubeCostWeights(),
        )

    clouds = [
        _preprocess_observation(
            observation, sample_key=sample_key, point_count=point_count
        )
    ]
    proprio_values = [extract_proprio(observation)] if enabled else []
    executed_actions: list[np.ndarray] = []
    height_trace: list[float] = []
    height_trace_control_steps: list[int] = []
    height_trace_sim_frames: list[int] = []
    steps = 0
    reward = 0.0
    environment_success = False
    stopped = False

    def record_height() -> None:
        height_trace.append(_cube_height(env))
        height_trace_control_steps.append(int(steps))
        height_trace_sim_frames.append(int(steps * sim_frames_per_control_step))

    while len(clouds) < int(model.cfg.T) and steps < max_steps:
        actions_to_boundary = min(chunk_size, max_steps - steps)
        for _ in range(actions_to_boundary):
            observation, reward, terminated, truncated, info = _step(
                env, NEUTRAL_OPEN_ACTION
            )
            executed_actions.append(NEUTRAL_OPEN_ACTION.copy())
            steps += 1
            record_height()
            environment_success = bool(info.get("success", False)) or environment_success
            if terminated or truncated:
                stopped = True
                break
        if stopped or actions_to_boundary < chunk_size:
            break
        clouds.append(
            _preprocess_observation(
                observation, sample_key=sample_key, point_count=point_count
            )
        )
        if enabled:
            proprio_values.append(extract_proprio(observation))

    replans: list[dict[str, object]] = []
    if len(clouds) == int(model.cfg.T):
        latent_history = _aggregate_clouds(model, clouds)
        history_length = (int(model.cfg.T) - 1) * chunk_size
        context_actions = torch.as_tensor(
            np.stack(executed_actions[-history_length:]),
            device=latent_history.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        proprio_history_normalized = (
            _normalize_proprio_history(model, proprio_values)
            if enabled
            else None
        )
        if enabled:
            assert goal_spec is not None
            initial_cost = _zero_action_cost_with_goal(
                model,
                latent_history,
                proprio_history_normalized,
                context_actions,
                goal_spec,
                planner,
            )
        else:
            initial_cost = float(
                normalized_latent_mse(latent_history[:, -1], goal_latent).item()
            )
        while not stopped and steps < max_steps:
            cost_components: dict[str, float] = {}
            predicted_object_height: list[float] = []
            predicted_object_pos: list[list[float]] = []
            predicted_tcp_xyz: list[list[float]] = []
            planner_predicted_success = False
            if enabled:
                assert goal_spec is not None
                assert proprio_history_normalized is not None
                zero_cost = _zero_action_cost_with_goal(
                    model,
                    latent_history,
                    proprio_history_normalized,
                    context_actions,
                    goal_spec,
                    planner,
                )
                plan = planner.plan_with_goal(
                    model,
                    latent_history,
                    proprio_history_normalized,
                    context_actions,
                    goal_spec,
                    env="maniskill2",
                    seed=int(cem_seed) + len(replans),
                )
                predicted_rollout = planner.last_rollout
                if predicted_rollout is None:
                    raise RuntimeError("goal-aware planner did not retain its final rollout")
                predicted_object_pos = (
                    predicted_rollout.object_pos[0].detach().cpu().tolist()
                )
                predicted_object_height = [
                    float(value[2]) for value in predicted_object_pos
                ]
                predicted_tcp_xyz = (
                    predicted_rollout.proprio[0, :, 18:21]
                    .detach()
                    .cpu()
                    .tolist()
                )
                cost_components = dict(planner.last_cost_components)
                planner_predicted_success = _predicted_success(
                    predicted_object_height, goal_spec, chunk_size=chunk_size
                )
            else:
                zero_cost = _zero_action_cost(
                    model, latent_history, context_actions, goal_latent, planner
                )
                plan = planner.plan(
                    model,
                    latent_history,
                    context_actions,
                    goal_latent,
                    env="maniskill2",
                    seed=int(cem_seed) + len(replans),
                )
            first_chunk = plan[0].detach().cpu().numpy().astype(np.float32)
            predicted_cost = float(planner.last_cost)
            reward = 0.0
            executed = 0
            for action in first_chunk:
                if steps >= max_steps:
                    break
                observation, reward, terminated, truncated, info = _step(env, action)
                executed_actions.append(action.copy())
                steps += 1
                executed += 1
                record_height()
                environment_success = bool(info.get("success", False)) or environment_success
                if terminated or truncated:
                    stopped = True
                    break

            arm = first_chunk[:executed, :-1]
            replans.append(
                {
                    "replan": len(replans),
                    "step": steps,
                    "control_step": steps,
                    "sim_frame": steps * sim_frames_per_control_step,
                    "predicted_cost": predicted_cost,
                    "predicted_cost_components": cost_components,
                    "predicted_object_pos": predicted_object_pos,
                    "predicted_object_height": predicted_object_height,
                    "predicted_tcp_xyz": predicted_tcp_xyz,
                    "planner_predicted_success": planner_predicted_success,
                    "zero_action_cost": zero_cost,
                    "cube_height": _cube_height(env),
                    "reward": reward,
                    "success": environment_success,
                    "arm_action_mean": float(arm.mean()) if arm.size else 0.0,
                    "arm_action_std": float(arm.std()) if arm.size else 0.0,
                    "gripper": float(first_chunk[0, -1]),
                }
            )
            if stopped or executed < chunk_size:
                break

            boundary_cloud = _preprocess_observation(
                observation, sample_key=sample_key, point_count=point_count
            )
            boundary_latent = _aggregate_clouds(model, [boundary_cloud])
            latent_history = torch.cat(
                [latent_history[:, 1:], boundary_latent], dim=1
            )
            if enabled:
                boundary_proprio = extract_proprio(observation)
                boundary_normalized = _normalize_proprio_history(
                    model, [boundary_proprio]
                )
                proprio_history_normalized = torch.cat(
                    [proprio_history_normalized[:, 1:], boundary_normalized],
                    dim=1,
                )
            chunk_tensor = torch.as_tensor(
                first_chunk,
                device=context_actions.device,
                dtype=context_actions.dtype,
            ).unsqueeze(0)
            context_actions = torch.cat(
                [context_actions, chunk_tensor], dim=1
            )[:, chunk_size:]
    else:
        initial_cost = float("nan")

    height_summary = summarize_height_trace(
        height_trace,
        cube_z0=cube_z0,
        lift_threshold=resolved_lift_threshold,
        hold_steps=hold_steps,
        height_tolerance=height_tolerance,
    )
    return {
        "env_seed": int(env_seed),
        "cem_seed": int(cem_seed),
        "success": bool(height_summary["success"]),
        "environment_success": environment_success,
        "steps": steps,
        "control_steps": steps,
        "sim_frames": steps * sim_frames_per_control_step,
        "sim_frames_per_control_step": sim_frames_per_control_step,
        "planning_calls": len(replans),
        "planner_predicted_success": any(
            bool(row["planner_predicted_success"]) for row in replans
        ),
        "goal_spec": None if goal_spec is None else goal_spec.to_dict(),
        "initial_cost": initial_cost,
        "final_reward": reward,
        "height_trace": height_trace,
        "height_trace_control_steps": height_trace_control_steps,
        "height_trace_sim_frames": height_trace_sim_frames,
        "replans": replans,
        **height_summary,
        "max_height_sim_frame": int(
            height_summary["max_height_step"] * sim_frames_per_control_step
        ),
    }


@torch.no_grad()
def score_validation_reference(
    model,
    state: dict,
    device: torch.device,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, float]:
    """Score one fixed H2 demonstrated-action window for cost calibration."""
    del state  # Kept in the public interface for checkpoint provenance symmetry.
    dataset = LiftCubeRolloutDataset(
        data_root,
        T=int(model.cfg.T),
        point_count=int(model.cfg.K),
        horizon=2,
        skip_frame=int(model.cfg.skip_frame),
        include_rgb=int(model.cfg.C) == 6,
    )
    _, context, future, actions = dataset[0]
    latent_history = model._aggregate(context.unsqueeze(0).to(device))
    future_latents = model._aggregate(future.unsqueeze(0).to(device))
    goal = future_latents[:, -1]
    context_actions = dataset.context_actions_for_index(0).unsqueeze(0).to(device)
    action_candidates = actions.reshape(
        1, 2, int(model.cfg.skip_frame), -1
    ).to(device)
    zero = torch.zeros_like(action_candidates)
    if _proprio_enabled(model):
        ref = dataset.index[0]
        episode = dataset._load_episode(ref.episode_id)
        frame_ids = ref.start + np.arange(int(model.cfg.T)) * int(
            model.cfg.skip_frame
        )
        raw_proprio = np.ascontiguousarray(
            np.concatenate(
                [
                    episode["qpos"][frame_ids],
                    episode["qvel"][frame_ids],
                    episode["tcp_pose"][frame_ids],
                ],
                axis=-1,
            ),
            dtype=np.float32,
        )
        proprio_history = model.proprio_encoder.normalize(
            torch.as_tensor(raw_proprio, device=device).unsqueeze(0)
        )
        demonstrated = rollout_with_aux(
            model,
            latent_history,
            proprio_history,
            context_actions,
            action_candidates,
            "maniskill2",
        ).scene_latent
        zero_rollout = rollout_with_aux(
            model,
            latent_history,
            proprio_history,
            context_actions,
            zero,
            "maniskill2",
        ).scene_latent
    else:
        demonstrated = rollout_from_latent(
            model, latent_history, context_actions, action_candidates, "maniskill2"
        )
        zero_rollout = rollout_from_latent(
            model, latent_history, context_actions, zero, "maniskill2"
        )
    persistence = latent_history[:, -1:].expand(
        -1, 2, -1, -1
    )
    weight = 0.25
    return {
        "demonstrated_action_cost": float(
            latent_trajectory_cost(demonstrated, goal, trajectory_weight=weight).item()
        ),
        "zero_action_cost": float(
            latent_trajectory_cost(
                zero_rollout,
                goal,
                trajectory_weight=weight,
            ).item()
        ),
        "persistence_cost": float(
            latent_trajectory_cost(persistence, goal, trajectory_weight=weight).item()
        ),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_median(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen LiftCube world model with latent or goal-aware CEM."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cem-seed", type=int, default=31_000)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=None,
        help="Cube height threshold in meters; defaults to the LiftCube task criterion.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=DEFAULT_HOLD_STEPS,
        help="Required consecutive control steps above the lift threshold.",
    )
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=DEFAULT_HEIGHT_TOLERANCE,
        help="Maximum peak-window height range in meters.",
    )
    parser.add_argument("--approach-radius", type=float, default=0.04)
    parser.add_argument("--gripper-closed-threshold", type=float, default=0.02)
    parser.add_argument("--weight-height", type=float, default=1.0)
    parser.add_argument("--weight-hold", type=float, default=1.0)
    parser.add_argument("--weight-stability", type=float, default=0.5)
    parser.add_argument("--weight-latent", type=float, default=0.05)
    parser.add_argument("--weight-approach", type=float, default=0.1)
    parser.add_argument("--weight-early-close", type=float, default=0.1)
    parser.add_argument("--weight-gripper", type=float, default=0.1)
    parser.add_argument("--weight-contact", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    mean, std = action_stats(args.data_root, split="train")
    config = CEMConfig(
        horizon=args.horizon,
        chunk_size=args.chunk_size,
        action_dim=len(mean),
        candidates=args.candidates,
        model_batch_size=args.model_batch_size,
        iterations=args.iterations,
        elites=args.elites,
    )
    planner = LatentCEMPlanner(config, mean, std)
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
    provenance = {
        "schema_version": 1,
        "task": "LiftCube-v0",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "step": int(state["step"]),
        },
        "data_root": str(args.data_root.resolve()),
        "seeds": list(args.seeds),
        "cem_seed": int(args.cem_seed),
        "cem": vars(config),
        "goal_cost": {
            "approach_radius": float(args.approach_radius),
            "gripper_closed_threshold": float(args.gripper_closed_threshold),
            "weights": cost_weights.to_dict(),
        },
        "max_steps": int(args.max_steps),
        "height_evaluation": {
            "lift_threshold": args.lift_threshold,
            "hold_steps": int(args.hold_steps),
            "hold_step_unit": "control_step",
            "height_tolerance": float(args.height_tolerance),
            "sim_frames_per_control_step": None,
            "sim_frame_source": (
                "env.unwrapped._sim_steps_per_control or sim_freq/control_freq"
            ),
        },
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
    }
    atomic_write_json(args.output / "config.json", provenance)
    validation = score_validation_reference(model, state, device, args.data_root)

    results = []
    env = make_env()
    try:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as handle:
            for index, seed in enumerate(args.seeds):
                result = run_episode(
                    env,
                    model,
                    planner,
                    env_seed=seed,
                    cem_seed=args.cem_seed + index * 10_000,
                    max_steps=args.max_steps,
                    lift_threshold=args.lift_threshold,
                    hold_steps=args.hold_steps,
                    height_tolerance=args.height_tolerance,
                    approach_radius=args.approach_radius,
                    gripper_closed_threshold=args.gripper_closed_threshold,
                    cost_weights=cost_weights,
                )
                results.append(result)
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        env.close()

    replan_rows = [row for result in results for row in result["replans"]]
    sim_frame_values = sorted(
        {
            int(row["sim_frames_per_control_step"])
            for row in results
            if "sim_frames_per_control_step" in row
        }
    )
    threshold_values = sorted(
        {
            float(row["lift_threshold"])
            for row in results
            if "lift_threshold" in row
        }
    )
    success_count = sum(bool(result["success"]) for result in results)
    summary = {
        "schema_version": 1,
        "success_count": success_count,
        "episode_count": len(results),
        "success_rate": success_count / len(results),
        "mean_max_cube_height": _finite_mean(
            [float(row["max_cube_height"]) for row in results if "max_cube_height" in row]
        ),
        "median_max_cube_height": _finite_median(
            [float(row["max_cube_height"]) for row in results if "max_cube_height" in row]
        ),
        "mean_final_cube_height": _finite_mean(
            [float(row["final_cube_height"]) for row in results if "final_cube_height" in row]
        ),
        "success_by_peak": sum(
            bool(row.get("success_by_peak", False)) for row in results
        ),
        "success_by_peak_rate": sum(
            bool(row.get("success_by_peak", False)) for row in results
        ) / len(results),
        "success_by_peak_and_hold": sum(
            bool(row.get("success_by_peak_and_hold", False)) for row in results
        ),
        "success_by_peak_and_hold_rate": sum(
            bool(row.get("success_by_peak_and_hold", False)) for row in results
        ) / len(results),
        "height_evaluation": {
            "lift_threshold": (
                threshold_values[0]
                if len(threshold_values) == 1
                else threshold_values
            ),
            "hold_steps": int(args.hold_steps),
            "hold_step_unit": "control_step",
            "height_tolerance": float(args.height_tolerance),
            "sim_frames_per_control_step": (
                sim_frame_values[0]
                if len(sim_frame_values) == 1
                else sim_frame_values
            ),
            "sim_frame_source": (
                "env.unwrapped._sim_steps_per_control or sim_freq/control_freq"
            ),
        },
        "mean_initial_cost": _mean([float(row["initial_cost"]) for row in results]),
        "mean_cem_cost": _mean(
            [float(row["predicted_cost"]) for row in replan_rows]
        ),
        "mean_zero_action_cost": _mean(
            [float(row["zero_action_cost"]) for row in replan_rows]
        ),
        "validation_reference": validation,
        "episodes": results,
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
