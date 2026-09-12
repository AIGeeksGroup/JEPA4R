"""Deterministic representation, geometric-match, and action-use diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), q).detach().cpu())


def representation_health(
    tokens: torch.Tensor, *, pair_samples: int = 65536, seed: int = 3072
) -> dict[str, float | int | bool]:
    """Summarize collapse, anisotropy, rank, norm, and pairwise cosine."""
    if tokens.shape[-1] < 1 or pair_samples < 1:
        raise ValueError("tokens need features and pair_samples must be positive")
    flat = tokens.detach().float().reshape(-1, tokens.shape[-1])
    if flat.shape[0] < 2:
        raise ValueError("representation health requires at least two tokens")
    finite = bool(torch.isfinite(flat).all())
    if not finite:
        return {"finite": False, "sample_count": int(flat.shape[0])}
    channel_std = flat.std(dim=0, unbiased=False)
    norms = flat.norm(dim=-1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min_(0)
    total = eigenvalues.sum()
    if float(total) <= 1e-12:
        effective_rank = 0.0
        participation_ratio = 0.0
    else:
        probability = eigenvalues / total
        nonzero = probability > 0
        entropy = -(probability[nonzero] * probability[nonzero].log()).sum()
        effective_rank = float(entropy.exp().cpu())
        participation_ratio = float(
            (total.square() / eigenvalues.square().sum().clamp_min(1e-12)).cpu()
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(pair_samples, max(flat.shape[0] * (flat.shape[0] - 1), 1))
    left = torch.randint(flat.shape[0], (count,), generator=generator)
    offset = torch.randint(1, flat.shape[0], (count,), generator=generator)
    right = (left + offset) % flat.shape[0]
    cosine = F.cosine_similarity(flat[left], flat[right], dim=-1, eps=1e-8)
    return {
        "finite": True,
        "sample_count": int(flat.shape[0]),
        "feature_dim": int(flat.shape[1]),
        "near_constant_channel_fraction": float((channel_std < 1e-6).float().mean()),
        "channel_std_min": float(channel_std.min().cpu()),
        "channel_std_mean": float(channel_std.mean().cpu()),
        "channel_std_max": float(channel_std.max().cpu()),
        "token_norm_mean": float(norms.mean().cpu()),
        "token_norm_std": float(norms.std(unbiased=False).cpu()),
        "token_norm_q05": _quantile(norms, 0.05),
        "token_norm_q50": _quantile(norms, 0.50),
        "token_norm_q95": _quantile(norms, 0.95),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "off_diagonal_cosine_mean": float(cosine.mean().cpu()),
        "off_diagonal_cosine_q05": _quantile(cosine, 0.05),
        "off_diagonal_cosine_q50": _quantile(cosine, 0.50),
        "off_diagonal_cosine_q95": _quantile(cosine, 0.95),
    }


def mutual_nearest_matches(
    left_xyz: torch.Tensor,
    right_xyz: torch.Tensor,
    *,
    max_distance: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Return mutual nearest point pairs for independently sampled batches."""
    if (
        left_xyz.dim() != 3
        or right_xyz.dim() != 3
        or left_xyz.shape[0] != right_xyz.shape[0]
        or left_xyz.shape[-1] != 3
        or right_xyz.shape[-1] != 3
    ):
        raise ValueError("positions must be batch-aligned [B,K,3] tensors")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    batch_out: list[torch.Tensor] = []
    left_out: list[torch.Tensor] = []
    right_out: list[torch.Tensor] = []
    distance_out: list[torch.Tensor] = []
    for batch_index in range(left_xyz.shape[0]):
        distances = torch.cdist(
            left_xyz[batch_index].float(), right_xyz[batch_index].float()
        )
        left_to_right = distances.argmin(dim=1)
        right_to_left = distances.argmin(dim=0)
        left_index = torch.arange(left_xyz.shape[1], device=left_xyz.device)
        right_index = left_to_right
        mutual = right_to_left[right_index] == left_index
        selected_distance = distances[left_index, right_index]
        keep = mutual & (selected_distance <= max_distance)
        batch_out.append(torch.full_like(left_index[keep], batch_index))
        left_out.append(left_index[keep])
        right_out.append(right_index[keep])
        distance_out.append(selected_distance[keep])
    return {
        "batch": torch.cat(batch_out),
        "left": torch.cat(left_out),
        "right": torch.cat(right_out),
        "distance": torch.cat(distance_out),
    }


def matched_temporal_signal(
    left_xyz: torch.Tensor,
    right_xyz: torch.Tensor,
    left_latent: torch.Tensor,
    right_latent: torch.Tensor,
    *,
    max_distance: float = 0.03,
) -> dict[str, float | int]:
    """Compare geometric and latent displacement on mutual nearest matches."""
    if left_latent.shape[:2] != left_xyz.shape[:2]:
        raise ValueError("left latent and position counts differ")
    if right_latent.shape[:2] != right_xyz.shape[:2]:
        raise ValueError("right latent and position counts differ")
    matches = mutual_nearest_matches(
        left_xyz, right_xyz, max_distance=max_distance
    )
    if matches["batch"].numel() == 0:
        raise ValueError("no mutual nearest matches within max_distance")
    batch = matches["batch"]
    left = matches["left"]
    right = matches["right"]
    geometry = (
        right_xyz[batch, right].float() - left_xyz[batch, left].float()
    ).norm(dim=-1)
    latent = (
        right_latent[batch, right].float() - left_latent[batch, left].float()
    ).norm(dim=-1)
    if float(geometry.std(unbiased=False)) < 1e-12 or float(
        latent.std(unbiased=False)
    ) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(torch.corrcoef(torch.stack([geometry, latent]))[0, 1])
    low = torch.quantile(geometry, 0.25)
    high = torch.quantile(geometry, 0.75)
    static = latent[geometry <= low]
    moving = latent[geometry >= high]
    return {
        "match_count": int(geometry.numel()),
        "geometry_displacement_mean": float(geometry.mean().cpu()),
        "latent_displacement_mean": float(latent.mean().cpu()),
        "geometry_latent_correlation": correlation,
        "static_latent_displacement_mean": float(static.mean().cpu()),
        "moving_latent_displacement_mean": float(moving.mean().cpu()),
    }


def action_gradient_norm(
    model: torch.nn.Module,
    points: torch.Tensor,
    actions: torch.Tensor,
    *,
    env: str | None = None,
    context_actions: torch.Tensor | None = None,
) -> float:
    """Measure local prediction sensitivity to atomic actions."""
    action_input = actions.detach().clone().requires_grad_(True)
    rollout_kwargs = {"env": env}
    if context_actions is not None:
        rollout_kwargs["context_actions"] = context_actions
    prediction = model.rollout(points, action_input, **rollout_kwargs)
    scalar = prediction.float().square().mean()
    gradient = torch.autograd.grad(scalar, action_input, retain_graph=False)[0]
    value = float(gradient.float().norm().detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("action gradient norm is not finite")
    return value


def judge_action_control(
    correct: float,
    shuffled: float,
    persistence: float,
    *,
    threshold: float = 0.10,
) -> dict[str, float | bool]:
    """Require action separation and genuine improvement over persistence."""
    if min(correct, shuffled, persistence) < 0 or threshold < 0:
        raise ValueError("errors and threshold must be non-negative")
    degradation = (shuffled - correct) / max(correct, 1e-12)
    beats_persistence = correct < persistence
    return {
        "shuffle_degradation": degradation,
        "threshold": threshold,
        "beats_persistence": beats_persistence,
        "passed": bool(degradation >= threshold and beats_persistence),
    }
