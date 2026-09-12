"""Checkpoint-level LiftCube latent and action-use diagnostics."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.eval_maniskill_group1_rollout import (
    DEFAULT_HORIZONS,
    LiftCubeRolloutDataset,
    atomic_write_json,
    evaluate_rollouts,
    load_checkpoint_model,
    rollout_predictions,
    sha256_file,
)
from scripts.latent_diagnostics import (
    judge_action_control,
    matched_temporal_signal,
    representation_health,
)


def _spatial_stages(
    model: torch.nn.Module, points: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    batch_size, frame_count, point_count, channel_count = points.shape
    aggregator = model.aggregator
    if hasattr(aggregator, "diagnostic_features"):
        stages = aggregator.diagnostic_features(
            points.reshape(batch_size * frame_count, point_count, channel_count)
        )
        raw = stages["raw_spatial"].view(
            batch_size, frame_count, point_count, -1
        )
        projected = stages["projected_spatial"].view(
            batch_size, frame_count, point_count, -1
        )
        positions = stages["positions"].view(
            batch_size, frame_count, point_count, 3
        )
        return raw, projected, positions
    projected, positions = model._aggregate(points, return_positions=True)
    return None, projected, positions


def _conditioned_action_gradient_norm(
    model: torch.nn.Module,
    dataset: LiftCubeRolloutDataset,
    indices: torch.Tensor,
    context: torch.Tensor,
    actions: torch.Tensor,
    context_actions: torch.Tensor,
    latent_history: torch.Tensor,
) -> float:
    action_input = actions.detach().clone().requires_grad_(True)
    prediction = rollout_predictions(
        model,
        dataset,
        indices,
        context,
        action_input,
        context_actions,
        latent_history=latent_history,
    )
    scalar = prediction.float().square().mean()
    gradient = torch.autograd.grad(scalar, action_input, retain_graph=False)[0]
    value = float(gradient.float().norm().detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("action gradient norm is not finite")
    return value


def evaluate_representation_stages(
    model: torch.nn.Module,
    dataset: LiftCubeRolloutDataset,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Measure each representation boundary on deterministic validation windows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_batches: list[torch.Tensor] = []
    projected_batches: list[torch.Tensor] = []
    temporal_batches: list[torch.Tensor] = []
    predicted_batches: list[torch.Tensor] = []
    match_inputs: list[tuple[torch.Tensor, ...]] = []
    window_indices: list[int] = []
    gradient_norms: list[float] = []

    model.eval()
    for batch_index, (indices, context, future, actions) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = context.to(device)
        future = future.to(device)
        actions = actions.to(device)
        context_actions = torch.stack(
            [dataset.context_actions_for_index(int(index)) for index in indices]
        ).to(device)
        all_points = torch.cat([context, future], dim=1)

        with torch.no_grad():
            raw, projected, positions = _spatial_stages(model, all_points)
            temporal = model.temporal(
                projected[:, : model.cfg.T], causal=True
            )
            predicted = rollout_predictions(
                model,
                dataset,
                indices,
                context,
                actions,
                context_actions,
                latent_history=projected[:, : model.cfg.T],
            )

        if raw is not None:
            raw_batches.append(raw.cpu())
        projected_batches.append(projected.cpu())
        temporal_batches.append(temporal.cpu())
        predicted_batches.append(predicted.cpu())
        match_inputs.append(
            (
                positions[:, model.cfg.T - 1].cpu(),
                positions[:, model.cfg.T].cpu(),
                projected[:, model.cfg.T - 1].cpu(),
                projected[:, model.cfg.T].cpu(),
            )
        )
        window_indices.extend(int(index) for index in indices)
        gradient_norms.append(
            _conditioned_action_gradient_norm(
                model,
                dataset,
                indices,
                context,
                actions,
                context_actions,
                projected[:, : model.cfg.T],
            )
        )

    if not projected_batches:
        raise RuntimeError("no validation batches were evaluated")

    left_xyz, right_xyz, left_latent, right_latent = (
        torch.cat([items[position] for items in match_inputs], dim=0)
        for position in range(4)
    )
    return {
        "raw_spatial": (
            representation_health(torch.cat(raw_batches), seed=seed)
            if raw_batches
            else None
        ),
        "projected_spatial": representation_health(
            torch.cat(projected_batches), seed=seed
        ),
        "temporal_context": representation_health(
            torch.cat(temporal_batches), seed=seed
        ),
        "predicted": representation_health(
            torch.cat(predicted_batches), seed=seed
        ),
        "matched_temporal_signal": matched_temporal_signal(
            left_xyz, right_xyz, left_latent, right_latent
        ),
        "action_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "window_indices": window_indices,
    }


def annotate_action_judgments(controls: dict[str, object]) -> dict[str, object]:
    """Attach the two-part action-control gate to every reported horizon."""
    result = copy.deepcopy(controls)
    for row in result["per_horizon"].values():
        judgment = judge_action_control(
            correct=row["model_correct_action"]["mse"],
            shuffled=row["model_shuffled_action"]["mse"],
            persistence=row["persistence"]["mse"],
        )
        row["action_judgment"] = judgment
        row["action_valid"] = judgment["passed"]
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LiftCube latent health and action use."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    device = torch.device(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    include_rgb = int(model.cfg.C) == 6
    dataset = LiftCubeRolloutDataset(
        args.data_root,
        T=model.cfg.T,
        point_count=model.cfg.K,
        horizon=max(horizons),
        skip_frame=getattr(model.cfg, "skip_frame", 1),
        include_rgb=include_rgb,
    )

    started = time.perf_counter()
    controls = annotate_action_judgments(
        evaluate_rollouts(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
            horizons=horizons,
            seed=args.seed,
        )
    )
    representations = evaluate_representation_stages(
        model,
        dataset,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    manifest = args.data_root / "manifest.json"
    result = {
        "schema_version": 1,
        "task": "LiftCube-v0",
        "split": "val",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "step": int(state["step"]),
            "model_config": state["model_config"],
            "train_config": state["train_config"],
        },
        "data": {
            "root": str(args.data_root.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "episode_ids": list(dataset.episode_ids),
            "include_rgb": include_rgb,
        },
        "evaluation": {
            "horizons": list(horizons),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "window_count": len(dataset),
            "elapsed_sec": time.perf_counter() - started,
        },
        "action_controls": controls,
        "representations": representations,
    }
    atomic_write_json(args.output, result)
    return result


if __name__ == "__main__":
    main()
