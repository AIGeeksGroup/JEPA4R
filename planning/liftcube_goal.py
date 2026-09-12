"""Validated LiftCube lift-and-hold goals and trajectory costs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LiftCubeCostWeights:
    height: float = 1.0
    hold: float = 1.0
    stability: float = 0.5
    latent: float = 0.05
    approach: float = 0.1
    early_close: float = 0.1
    gripper: float = 0.0
    contact: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"cost weight {name} must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class GoalSpec:
    initial_cube_height: float
    lift_threshold: float
    hold_control_steps: int = 10
    height_tolerance: float = 0.01
    approach_radius: float = 0.04
    gripper_closed_threshold: float = 0.02
    demonstrated_goal_latent: torch.Tensor | None = None
    weights: LiftCubeCostWeights = field(default_factory=LiftCubeCostWeights)

    def __post_init__(self) -> None:
        for name in (
            "initial_cube_height",
            "lift_threshold",
            "height_tolerance",
            "approach_radius",
            "gripper_closed_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.hold_control_steps < 1:
            raise ValueError("hold_control_steps must be positive")
        if (
            self.height_tolerance < 0
            or self.approach_radius < 0
            or self.gripper_closed_threshold < 0
        ):
            raise ValueError(
                "height_tolerance, approach_radius, and gripper_closed_threshold "
                "must be non-negative"
            )
        if self.demonstrated_goal_latent is not None:
            latent = self.demonstrated_goal_latent
            if latent.ndim != 3 or latent.shape[0] != 1:
                raise ValueError("demonstrated_goal_latent must be [1,K,D]")
            if not torch.isfinite(latent).all():
                raise ValueError("demonstrated_goal_latent must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_cube_height": float(self.initial_cube_height),
            "lift_threshold": float(self.lift_threshold),
            "hold_control_steps": int(self.hold_control_steps),
            "height_tolerance": float(self.height_tolerance),
            "approach_radius": float(self.approach_radius),
            "gripper_closed_threshold": float(self.gripper_closed_threshold),
            "demonstrated_goal_latent": None
            if self.demonstrated_goal_latent is None
            else {"shape": list(self.demonstrated_goal_latent.shape)},
            "weights": self.weights.to_dict(),
        }


@dataclass(frozen=True)
class LiftCubeCost:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def serializable(self, index: int = 0) -> dict[str, object]:
        return {
            "total": float(self.total[index].detach().cpu()),
            "components": {
                name: float(value[index].detach().cpu())
                for name, value in self.components.items()
            },
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1)
    total = (value * mask.to(value.dtype)).sum(dim=1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))


def liftcube_trajectory_cost(
    rollout,
    action_candidates: torch.Tensor,
    goal: GoalSpec,
    *,
    chunk_size: int,
) -> LiftCubeCost:
    """Score denormalized predicted geometry; normalized proprio is never used."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    scene = rollout.scene_latent
    normalized = rollout.proprio_normalized
    proprio = rollout.proprio
    object_pos = rollout.object_pos
    contact_probability = getattr(rollout, "contact_probability", None)
    if scene.ndim != 4:
        raise ValueError("scene_latent must be [N,H,K,D]")
    candidates, horizon = scene.shape[:2]
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    required_nodes = math.ceil(goal.hold_control_steps / chunk_size) + 1
    if horizon < max(2, required_nodes):
        raise ValueError(
            f"rollout horizon {horizon} is too short for hold requirement "
            f"of {goal.hold_control_steps} control steps"
        )
    if normalized.ndim != 3 or normalized.shape[:2] != (candidates, horizon):
        raise ValueError("proprio_normalized must be [N,H,P]")
    if proprio.shape != normalized.shape or proprio.shape[-1] < 21:
        raise ValueError("raw proprio must match normalized [N,H,P] with TCP xyz")
    if object_pos.shape != (candidates, horizon, 3):
        raise ValueError("object_pos must be [N,H,3]")
    if contact_probability is not None:
        if contact_probability.shape != (candidates, horizon):
            raise ValueError("contact_probability must be [N,H]")
        if not torch.isfinite(contact_probability).all() or not torch.all(
            (contact_probability >= 0) & (contact_probability <= 1)
        ):
            raise ValueError("contact_probability must be finite and in [0,1]")
    elif goal.weights.contact > 0:
        raise ValueError("contact_probability is required by nonzero contact weight")
    if action_candidates.ndim != 4 or action_candidates.shape[:3] != (
        candidates,
        horizon,
        chunk_size,
    ):
        raise ValueError("action_candidates must be [N,H,chunk_size,A]")
    if action_candidates.shape[-1] < 2:
        raise ValueError("action candidates must include arm and gripper dimensions")
    if not torch.isfinite(action_candidates).all():
        raise ValueError("action candidates must be finite")

    height = object_pos[..., 2]
    tail = height[:, -required_nodes:]
    height_cost = torch.relu(goal.lift_threshold - height[:, -1]).square()
    hold_cost = torch.relu(goal.lift_threshold - tail).square().mean(dim=1)
    delta = tail[:, 1:] - tail[:, :-1]
    stability_cost = (
        torch.relu(delta.abs() - goal.height_tolerance).square().mean(dim=1)
        + torch.relu(-delta).square().mean(dim=1)
    )

    latent_cost = height_cost.new_zeros(candidates)
    if goal.demonstrated_goal_latent is not None:
        demonstrated = goal.demonstrated_goal_latent.to(
            device=scene.device, dtype=scene.dtype
        )
        if demonstrated.shape[1:] != scene.shape[2:]:
            raise ValueError("demonstrated goal latent shape does not match rollout")
        per_step = (
            F.normalize(scene, dim=-1, eps=1e-8)
            - F.normalize(demonstrated, dim=-1, eps=1e-8).unsqueeze(1)
        ).square().mean(dim=(-1, -2))
        latent_cost = per_step[:, -1] + 0.25 * per_step.mean(dim=1)

    tcp_xyz = proprio[..., 18:21]
    distance = torch.linalg.vector_norm(tcp_xyz - object_pos, dim=-1)
    pre_lift = height < goal.lift_threshold
    beyond_radius = torch.relu(distance - goal.approach_radius).square()
    approach_cost = _masked_mean(beyond_radius, pre_lift)
    close_amount = ((1.0 - action_candidates[..., -1]) / 2.0).clamp(0.0, 1.0)
    close_amount = close_amount.mean(dim=2)
    early_close_cost = _masked_mean(close_amount * beyond_radius, pre_lift)

    gripper_gap = proprio[..., 7].abs() + proprio[..., 8].abs()
    lifted = height > goal.initial_cube_height + goal.height_tolerance
    near_object = distance <= goal.approach_radius
    gripper_required = near_object | lifted
    gripper_cost = _masked_mean(
        torch.relu(gripper_gap - goal.gripper_closed_threshold).square(),
        gripper_required,
    )
    contact_cost = height_cost.new_zeros(candidates)
    if contact_probability is not None:
        predicted_closed = gripper_gap <= goal.gripper_closed_threshold
        contact_required = (near_object & predicted_closed) | lifted
        contact_cost = _masked_mean(
            (1.0 - contact_probability).square(), contact_required
        )

    components = {
        "height": height_cost,
        "hold": hold_cost,
        "stability": stability_cost,
        "latent": latent_cost,
        "approach": approach_cost,
        "early_close": early_close_cost,
        "gripper": gripper_cost,
        "contact": contact_cost,
    }
    valid = (
        torch.isfinite(scene).flatten(1).all(dim=1)
        & torch.isfinite(normalized).flatten(1).all(dim=1)
        & torch.isfinite(proprio).flatten(1).all(dim=1)
        & torch.isfinite(object_pos).flatten(1).all(dim=1)
    )
    if contact_probability is not None:
        valid = valid & torch.isfinite(contact_probability).flatten(1).all(dim=1)
    infinity = height_cost.new_full((candidates,), float("inf"))
    components = {
        name: torch.where(valid & torch.isfinite(value), value, infinity)
        for name, value in components.items()
    }
    weights = goal.weights
    total = (
        weights.height * components["height"]
        + weights.hold * components["hold"]
        + weights.stability * components["stability"]
        + weights.latent * components["latent"]
        + weights.approach * components["approach"]
        + weights.early_close * components["early_close"]
        + weights.gripper * components["gripper"]
        + weights.contact * components["contact"]
    )
    total = torch.where(valid & torch.isfinite(total), total, infinity)
    return LiftCubeCost(total=total, components=components)
