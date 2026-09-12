"""Torch-only CEM over ordered action chunks and cached spatial latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from planning.liftcube_goal import GoalSpec, liftcube_trajectory_cost


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 2
    chunk_size: int = 10
    action_dim: int = 7
    candidates: int = 128
    iterations: int = 5
    elites: int = 16
    model_batch_size: int = 32
    min_std: float = 0.03
    trajectory_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.chunk_size < 1 or self.action_dim < 2:
            raise ValueError("horizon and chunk_size must be positive; action_dim must be >= 2")
        if self.candidates < 2 or not 1 <= self.elites < self.candidates:
            raise ValueError("elites must be positive and smaller than candidates")
        if self.model_batch_size < 1:
            raise ValueError("model_batch_size must be positive")
        if self.iterations < 1 or self.min_std <= 0 or self.trajectory_weight < 0:
            raise ValueError("iterations/min_std must be positive and trajectory_weight non-negative")


@dataclass
class AuxiliaryRollout:
    scene_latent: torch.Tensor
    proprio_normalized: torch.Tensor
    proprio: torch.Tensor
    object_pos: torch.Tensor
    contact_probability: torch.Tensor | None = None


def normalized_latent_mse(prediction: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return mean squared distance after per-token feature normalization."""
    prediction = F.normalize(prediction, dim=-1, eps=1e-8)
    goal = F.normalize(goal, dim=-1, eps=1e-8)
    return (prediction - goal).square().mean(dim=(-1, -2))


def latent_trajectory_cost(
    rollout: torch.Tensor,
    goal: torch.Tensor,
    *,
    trajectory_weight: float = 0.25,
) -> torch.Tensor:
    """Score `[N,H,K,D]` rollouts against one `[1,K,D]` spatial goal."""
    if rollout.ndim != 4 or goal.ndim != 3:
        raise ValueError("rollout must be [N,H,K,D] and goal must be [1,K,D]")
    per_step = normalized_latent_mse(rollout, goal)
    return per_step[:, -1] + float(trajectory_weight) * per_step.mean(dim=1)


def _expand_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} batch must be 1 or {batch}, got {tensor.shape[0]}")


def rollout_from_latent(
    model,
    latent_history: torch.Tensor,
    context_actions: torch.Tensor,
    action_candidates: torch.Tensor,
    env: str,
) -> torch.Tensor:
    """Roll `[N,H,S,A]` candidates while shifting both model histories."""
    if latent_history.ndim != 4 or action_candidates.ndim != 4:
        raise ValueError("latent_history must be [B,T,K,D] and candidates [N,H,S,A]")
    candidates, horizon, chunk_size, action_dim = action_candidates.shape
    if chunk_size != int(model.cfg.skip_frame):
        raise ValueError(
            f"candidate chunk size {chunk_size} != model skip_frame {model.cfg.skip_frame}"
        )
    if latent_history.shape[1:] != (model.cfg.T, model.cfg.K, model.cfg.D):
        raise ValueError("latent history shape does not match model config")
    expected_history = (model.cfg.T - 1) * chunk_size
    if context_actions.ndim != 3 or context_actions.shape[1:] != (
        expected_history,
        action_dim,
    ):
        raise ValueError(
            f"context_actions must be [B,{expected_history},{action_dim}], "
            f"got {tuple(context_actions.shape)}"
        )

    history = _expand_batch(latent_history, candidates, "latent_history")
    action_history = _expand_batch(context_actions, candidates, "context_actions")
    predictions = []
    for step in range(horizon):
        current_actions = action_candidates[:, step]
        model_actions = torch.cat([action_history, current_actions], dim=1)
        prediction_sequence, _ = model._predict_sequence(history, model_actions, env)
        prediction = prediction_sequence[:, -1]
        predictions.append(prediction)
        history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)
        action_history = model_actions[:, chunk_size:]
    return torch.stack(predictions, dim=1)


def rollout_with_aux(
    model,
    latent_history: torch.Tensor,
    proprio_history_normalized: torch.Tensor,
    context_actions: torch.Tensor,
    action_candidates: torch.Tensor,
    env: str,
) -> AuxiliaryRollout:
    """Roll candidates with predicted normalized proprio as recurrent state."""
    if latent_history.ndim != 4 or action_candidates.ndim != 4:
        raise ValueError("latent_history must be [B,T,K,D] and candidates [N,H,S,A]")
    candidates, horizon, chunk_size, action_dim = action_candidates.shape
    if horizon < 1:
        raise ValueError("candidate horizon must be positive")
    if chunk_size != int(model.cfg.skip_frame):
        raise ValueError(
            f"candidate chunk size {chunk_size} != model skip_frame {model.cfg.skip_frame}"
        )
    if latent_history.shape[1:] != (model.cfg.T, model.cfg.K, model.cfg.D):
        raise ValueError("latent history shape does not match model config")
    proprio_dim = int(model.cfg.proprio.dim)
    if proprio_history_normalized.ndim != 3 or proprio_history_normalized.shape[1:] != (
        model.cfg.T,
        proprio_dim,
    ):
        raise ValueError(
            f"proprio history must be [B,{model.cfg.T},{proprio_dim}]"
        )
    expected_history = (model.cfg.T - 1) * chunk_size
    if context_actions.ndim != 3 or context_actions.shape[1:] != (
        expected_history,
        action_dim,
    ):
        raise ValueError(
            f"context_actions must be [B,{expected_history},{action_dim}], "
            f"got {tuple(context_actions.shape)}"
        )

    history = _expand_batch(latent_history, candidates, "latent_history")
    proprio_history = _expand_batch(
        proprio_history_normalized, candidates, "proprio_history_normalized"
    )
    action_history = _expand_batch(context_actions, candidates, "context_actions")
    scene_predictions = []
    normalized_predictions = []
    raw_predictions = []
    object_predictions = []
    contact_predictions = []
    for step in range(horizon):
        current_actions = action_candidates[:, step]
        model_actions = torch.cat([action_history, current_actions], dim=1)
        transition = model.predict_transition(
            history,
            proprio_history,
            model_actions,
            env=env,
        )
        next_scene = transition.scene_latent
        next_normalized = transition.next_proprio_normalized
        next_raw = transition.next_proprio
        next_object = transition.object_pos
        next_contact = getattr(transition, "contact_probability", None)
        if next_scene.shape != (candidates, model.cfg.K, model.cfg.D):
            raise ValueError("transition scene_latent has an unexpected shape")
        if next_normalized.shape != (candidates, proprio_dim):
            raise ValueError("transition next_proprio_normalized has an unexpected shape")
        if next_raw.shape != next_normalized.shape:
            raise ValueError("transition raw and normalized proprio shapes differ")
        if next_object.shape != (candidates, 3):
            raise ValueError("transition object_pos has an unexpected shape")
        if next_contact is not None:
            if next_contact.shape != (candidates,):
                raise ValueError(
                    "transition contact_probability has an unexpected shape"
                )
            contact_predictions.append(next_contact)
        elif contact_predictions:
            raise ValueError("transition contact_probability disappeared during rollout")
        scene_predictions.append(next_scene)
        normalized_predictions.append(next_normalized)
        raw_predictions.append(next_raw)
        object_predictions.append(next_object)
        history = torch.cat([history[:, 1:], next_scene.unsqueeze(1)], dim=1)
        proprio_history = torch.cat(
            [proprio_history[:, 1:], next_normalized.unsqueeze(1)], dim=1
        )
        action_history = model_actions[:, chunk_size:]
    if contact_predictions and len(contact_predictions) != horizon:
        raise ValueError("transition contact_probability availability changed during rollout")
    return AuxiliaryRollout(
        scene_latent=torch.stack(scene_predictions, dim=1),
        proprio_normalized=torch.stack(normalized_predictions, dim=1),
        proprio=torch.stack(raw_predictions, dim=1),
        object_pos=torch.stack(object_predictions, dim=1),
        contact_probability=None
        if not contact_predictions
        else torch.stack(contact_predictions, dim=1),
    )


class LatentCEMPlanner:
    def __init__(
        self,
        config: CEMConfig,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.config = config
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32).flatten()
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32).flatten()
        if self.action_mean.shape != (config.action_dim,) or self.action_std.shape != (
            config.action_dim,
        ):
            raise ValueError(f"action statistics must have shape [{config.action_dim}]")
        if not torch.isfinite(self.action_mean).all() or not torch.isfinite(self.action_std).all():
            raise ValueError("action statistics must be finite")
        self.last_cost = float("nan")
        self.last_cost_components: dict[str, float] = {}
        self.last_rollout: AuxiliaryRollout | None = None

    def _optimize(
        self,
        latent_history,
        *,
        seed: int,
        score_actions,
        return_population: bool = False,
    ):
        cfg = self.config
        device = latent_history.device
        dtype = latent_history.dtype
        statistics_mean = self.action_mean.to(device=device, dtype=dtype)
        statistics_std = self.action_std.to(device=device, dtype=dtype)
        arm_mean = statistics_mean[:-1].view(1, 1, -1).expand(
            cfg.horizon, cfg.chunk_size, -1
        ).clone()
        arm_std = statistics_std[:-1].clamp_min(cfg.min_std).view(
            1, 1, -1
        ).expand_as(arm_mean).clone()
        plus_probability = ((statistics_mean[-1] + 1.0) / 2.0).clamp(0.05, 0.95)
        plus_probability = plus_probability.expand(cfg.horizon).clone()
        generator = torch.Generator(device=device).manual_seed(int(seed))

        final_actions = None
        final_costs = None
        for _ in range(cfg.iterations):
            noise = torch.randn(
                cfg.candidates,
                cfg.horizon,
                cfg.chunk_size,
                cfg.action_dim - 1,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            arms = (
                arm_mean.unsqueeze(0) + arm_std.unsqueeze(0) * noise
            ).clamp(-1.0, 1.0)
            draws = torch.rand(
                cfg.candidates,
                cfg.horizon,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            gripper = torch.where(
                draws < plus_probability.unsqueeze(0),
                torch.ones_like(draws),
                -torch.ones_like(draws),
            )
            gripper = gripper[:, :, None, None].expand(
                -1, -1, cfg.chunk_size, 1
            )
            actions = torch.cat([arms, gripper], dim=-1)
            costs = torch.cat(
                [
                    score_actions(action_batch)
                    for action_batch in actions.split(cfg.model_batch_size)
                ],
                dim=0,
            )
            final_actions = actions
            final_costs = costs
            finite = torch.nonzero(torch.isfinite(costs), as_tuple=False).flatten()
            if len(finite) < cfg.elites:
                raise RuntimeError(
                    f"only {len(finite)} finite CEM candidates; {cfg.elites} elites required"
                )
            elite_order = torch.topk(
                costs[finite], cfg.elites, largest=False
            ).indices
            elite_indices = finite[elite_order]
            elite_arms = arms[elite_indices]
            elite_gripper = gripper[elite_indices, :, 0, 0]
            arm_mean = elite_arms.mean(dim=0)
            arm_std = elite_arms.std(dim=0, unbiased=False).clamp_min(cfg.min_std)
            plus_probability = (
                (elite_gripper == 1)
                .to(dtype)
                .mean(dim=0)
                .clamp(0.05, 0.95)
            )

        final_gripper = torch.where(
            plus_probability >= 0.5,
            torch.ones_like(plus_probability),
            -torch.ones_like(plus_probability),
        )
        final_gripper = final_gripper[:, None, None].expand(
            -1, cfg.chunk_size, 1
        )
        plan = torch.cat(
            [arm_mean.clamp(-1.0, 1.0), final_gripper], dim=-1
        )
        if return_population:
            assert final_actions is not None and final_costs is not None
            return plan, final_actions, final_costs
        return plan

    @torch.no_grad()
    def plan(
        self,
        model,
        latent_history: torch.Tensor,
        context_actions: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        env: str,
        seed: int,
    ) -> torch.Tensor:
        cfg = self.config

        def score_actions(actions):
            rollout = rollout_from_latent(
                model, latent_history, context_actions, actions, env
            )
            return latent_trajectory_cost(
                rollout,
                goal_latent,
                trajectory_weight=cfg.trajectory_weight,
            )

        plan = self._optimize(
            latent_history, seed=seed, score_actions=score_actions
        )
        final_rollout = rollout_from_latent(
            model, latent_history, context_actions, plan.unsqueeze(0), env
        )
        final_cost = latent_trajectory_cost(
            final_rollout, goal_latent, trajectory_weight=cfg.trajectory_weight
        )
        self.last_cost = float(final_cost.item())
        self.last_cost_components = {}
        self.last_rollout = None
        return plan

    @torch.no_grad()
    def plan_with_goal(
        self,
        model,
        latent_history: torch.Tensor,
        proprio_history_normalized: torch.Tensor,
        context_actions: torch.Tensor,
        goal_spec: GoalSpec,
        *,
        env: str,
        seed: int,
    ) -> torch.Tensor:
        cfg = self.config

        def score_actions(actions):
            rollout = rollout_with_aux(
                model,
                latent_history,
                proprio_history_normalized,
                context_actions,
                actions,
                env,
            )
            return liftcube_trajectory_cost(
                rollout, actions, goal_spec, chunk_size=cfg.chunk_size
            ).total

        plan = self._optimize(
            latent_history, seed=seed, score_actions=score_actions
        )
        final_actions = plan.unsqueeze(0)
        final_rollout = rollout_with_aux(
            model,
            latent_history,
            proprio_history_normalized,
            context_actions,
            final_actions,
            env,
        )
        final_cost = liftcube_trajectory_cost(
            final_rollout,
            final_actions,
            goal_spec,
            chunk_size=cfg.chunk_size,
        )
        self.last_cost = float(final_cost.total.item())
        self.last_cost_components = {
            name: float(value.item())
            for name, value in final_cost.components.items()
        }
        self.last_rollout = final_rollout
        return plan

    @torch.no_grad()
    def plan_with_goal_population(
        self,
        model,
        latent_history: torch.Tensor,
        proprio_history_normalized: torch.Tensor,
        context_actions: torch.Tensor,
        goal_spec: GoalSpec,
        *,
        env: str,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan and retain the final CEM population for ranking audits."""
        cfg = self.config

        def score_actions(actions):
            rollout = rollout_with_aux(
                model,
                latent_history,
                proprio_history_normalized,
                context_actions,
                actions,
                env,
            )
            return liftcube_trajectory_cost(
                rollout, actions, goal_spec, chunk_size=cfg.chunk_size
            ).total

        plan, candidates, costs = self._optimize(
            latent_history,
            seed=seed,
            score_actions=score_actions,
            return_population=True,
        )
        final_actions = plan.unsqueeze(0)
        final_rollout = rollout_with_aux(
            model,
            latent_history,
            proprio_history_normalized,
            context_actions,
            final_actions,
            env,
        )
        final_cost = liftcube_trajectory_cost(
            final_rollout,
            final_actions,
            goal_spec,
            chunk_size=cfg.chunk_size,
        )
        self.last_cost = float(final_cost.total.item())
        self.last_cost_components = {
            name: float(value.item())
            for name, value in final_cost.components.items()
        }
        self.last_rollout = final_rollout
        return plan, candidates, costs
