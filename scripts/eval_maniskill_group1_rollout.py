"""Validation-only multi-step latent rollout evaluation for LiftCube Group 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.maniskill import _sample_features, _validate_episode
from models import WorldModel, WorldModelConfig
from planning.latent_cem import rollout_with_aux
from scripts.train_maniskill_group1 import validate_group1_manifest


ACTION_DIM = 7
PROPRIO_DIM = 25
DEFAULT_HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class WindowRef:
    episode_id: int
    start: int


class LiftCubeRolloutDataset(Dataset):
    """Common H_max-eligible windows from Group 1 validation episodes."""

    def __init__(
        self,
        root: str | Path,
        *,
        T: int,
        point_count: int,
        horizon: int,
        skip_frame: int = 1,
        cache_size: int = 2,
        include_rgb: bool = False,
    ):
        if T < 1 or point_count < 1 or horizon < 1 or skip_frame < 1:
            raise ValueError("T, point_count, horizon, and skip_frame must be positive")
        self.root = Path(root)
        self.T = int(T)
        self.point_count = int(point_count)
        self.horizon = int(horizon)
        self.skip_frame = int(skip_frame)
        self.include_rgb = bool(include_rgb)
        self.cache_size = max(1, int(cache_size))
        manifest = validate_group1_manifest(self.root)
        self.episode_ids = [int(x) for x in manifest["val_episode_ids"]]
        self.index: list[WindowRef] = []
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        for episode_id in self.episode_ids:
            path = self.root / f"episode_{episode_id:04d}.npz"
            episode = self._read_episode(path)
            frames = int(episode["xyz"].shape[0])
            if episode["xyz"].shape[1] < self.point_count:
                raise ValueError(
                    f"{path}: has {episode['xyz'].shape[1]} points; "
                    f"{self.point_count} required"
                )
            window_count = frames - (self.T + self.horizon - 1) * self.skip_frame
            self.index.extend(
                WindowRef(episode_id, start)
                for start in range(max(0, window_count))
            )
        if not self.index:
            raise ValueError(
                f"no complete T={self.T}, H={self.horizon} validation windows "
                f"under {self.root}"
            )

    def _read_episode(self, path: Path) -> dict[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"missing validation episode: {path}")
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: np.asarray(archive[key]) for key in archive.files}
        _validate_episode(episode, path, require_rgb=self.include_rgb)
        return episode

    def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        if episode_id in self._cache:
            self._cache.move_to_end(episode_id)
            return self._cache[episode_id]
        episode = self._read_episode(self.root / f"episode_{episode_id:04d}.npz")
        self._cache[episode_id] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def __len__(self) -> int:
        return len(self.index)

    def actions_for_index(self, index: int) -> torch.Tensor:
        ref = self.index[index]
        episode = self._load_episode(ref.episode_id)
        first_action = ref.start + (self.T - 1) * self.skip_frame
        actions = np.ascontiguousarray(
            episode["action"][first_action : first_action + self.horizon * self.skip_frame],
            dtype=np.float32,
        )
        expected = (self.horizon * self.skip_frame, ACTION_DIM)
        if actions.shape != expected:
            raise ValueError(
                f"episode {ref.episode_id} start {ref.start}: expected actions "
                f"{expected}, got {actions.shape}"
            )
        return torch.from_numpy(actions)
    def context_actions_for_index(self, index: int) -> torch.Tensor:
        ref = self.index[index]
        episode = self._load_episode(ref.episode_id)
        actions = np.ascontiguousarray(
            episode["action"][ref.start : ref.start + (self.T - 1) * self.skip_frame],
            dtype=np.float32,
        )
        expected = ((self.T - 1) * self.skip_frame, ACTION_DIM)
        if actions.shape != expected:
            raise ValueError(
                f"episode {ref.episode_id} start {ref.start}: expected context actions {expected}, got {actions.shape}"
            )
        return torch.from_numpy(actions)

    def proprio_for_index(self, index: int) -> torch.Tensor:
        """Return context proprio aligned to the sampled context frames."""
        ref = self.index[index]
        episode = self._load_episode(ref.episode_id)
        frame_ids = ref.start + np.arange(self.T) * self.skip_frame
        parts: list[np.ndarray] = []
        for name, width in (("qpos", 9), ("qvel", 9), ("tcp_pose", 7)):
            if name not in episode:
                raise ValueError(
                    f"episode {ref.episode_id} is missing proprio field {name!r}"
                )
            values = np.asarray(episode[name])
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(
                    f"episode {ref.episode_id} {name} must be [F,{width}], "
                    f"got {values.shape}"
                )
            parts.append(values[frame_ids])
        proprio = np.ascontiguousarray(np.concatenate(parts, axis=-1), dtype=np.float32)
        if proprio.shape != (self.T, PROPRIO_DIM) or not np.isfinite(proprio).all():
            raise ValueError(
                f"episode {ref.episode_id} context proprio must be finite "
                f"[{self.T},{PROPRIO_DIM}], got {proprio.shape}"
            )
        return torch.from_numpy(proprio)


    def __getitem__(
        self, index: int
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.index[index]
        episode = self._load_episode(ref.episode_id)
        frame_ids = [
            ref.start + offset * self.skip_frame
            for offset in range(self.T + self.horizon)
        ]
        sampled = np.stack(
            [
                _sample_features(
                    episode,
                    frame_id,
                    self.point_count,
                    ref.episode_id,
                    self.include_rgb,
                )
                for frame_id in frame_ids
            ]
        )
        context = torch.from_numpy(sampled[: self.T])
        future = torch.from_numpy(sampled[self.T :])
        return index, context, future, self.actions_for_index(index)


def build_action_shuffle_indices(count: int, seed: int) -> np.ndarray:
    """Return a deterministic cyclic derangement of range(count)."""
    if count < 2:
        raise ValueError("action shuffling requires at least two windows")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(1, count))
    return (np.arange(count, dtype=np.int64) + offset) % count


def _action_chunks(actions: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if actions.dim() != 3 or chunk_size < 1 or actions.shape[1] % chunk_size:
        raise ValueError(
            "actions must be [B,M,A] with M divisible by a positive chunk_size"
        )
    return actions.reshape(
        actions.shape[0], actions.shape[1] // chunk_size, chunk_size, actions.shape[2]
    )


def reverse_action_chunks(actions: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Reverse atomic actions inside each transition chunk."""
    return _action_chunks(actions, chunk_size).flip(2).reshape_as(actions)


def repeat_last_action_chunks(actions: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Replace each transition chunk with repetitions of its final action."""
    chunks = _action_chunks(actions, chunk_size)
    return chunks[:, :, -1:].expand_as(chunks).reshape_as(actions)


def rollout_predictions(
    model: torch.nn.Module,
    dataset: LiftCubeRolloutDataset,
    indices: torch.Tensor,
    context: torch.Tensor,
    actions: torch.Tensor,
    context_actions: torch.Tensor,
    *,
    latent_history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll scene latents through the checkpoint's actual conditioning path."""
    proprio_enabled = bool(
        getattr(getattr(model.cfg, "proprio", None), "enabled", False)
    )
    if not proprio_enabled:
        return model.rollout(
            context,
            actions,
            env="maniskill2",
            context_actions=context_actions,
        )
    if getattr(model, "proprio_encoder", None) is None:
        raise RuntimeError("proprio-enabled model is missing its proprio encoder")
    if latent_history is None:
        latent_history = model._aggregate(context)
    raw_proprio = torch.stack(
        [dataset.proprio_for_index(int(index)) for index in indices]
    ).to(device=context.device, dtype=context.dtype)
    normalized_proprio = model.proprio_encoder.normalize(raw_proprio)
    action_candidates = _action_chunks(actions, dataset.skip_frame)
    return rollout_with_aux(
        model,
        latent_history,
        normalized_proprio,
        context_actions,
        action_candidates,
        "maniskill2",
    ).scene_latent


@dataclass
class ErrorTotals:
    squared_error: float = 0.0
    element_count: int = 0
    cosine_distance: float = 0.0
    vector_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.shape[-1] < 1:
            raise ValueError("latent feature dimension must be positive")
        delta = prediction - target
        self.squared_error += float(delta.square().sum().detach().cpu())
        self.element_count += int(delta.numel())
        cosine = 1.0 - F.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=1e-8
        )
        self.cosine_distance += float(cosine.sum().detach().cpu())
        self.vector_count += int(cosine.numel())

    def result(self) -> dict[str, float]:
        if self.element_count == 0 or self.vector_count == 0:
            raise RuntimeError("cannot finalize empty error totals")
        return {
            "mse": self.squared_error / self.element_count,
            "cosine_distance": self.cosine_distance / self.vector_count,
        }


@dataclass
class SquaredErrorTotals:
    squared_error: float = 0.0
    element_count: int = 0

    def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.shape != right.shape:
            raise ValueError(
                f"delta shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
            )
        delta = left - right
        self.squared_error += float(delta.square().sum().detach().cpu())
        self.element_count += int(delta.numel())

    def result(self) -> float:
        if self.element_count == 0:
            raise RuntimeError("cannot finalize empty squared-error totals")
        return self.squared_error / self.element_count


def evaluate_rollouts(
    model: torch.nn.Module,
    dataset: LiftCubeRolloutDataset,
    *,
    batch_size: int,
    device: torch.device,
    horizons: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    ordered_horizons = tuple(sorted(set(int(x) for x in horizons)))
    if not ordered_horizons or ordered_horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    if ordered_horizons[-1] != dataset.horizon:
        raise ValueError(
            f"dataset horizon {dataset.horizon} must equal max horizon "
            f"{ordered_horizons[-1]}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    shuffle_indices = build_action_shuffle_indices(len(dataset), seed)
    action_bank = torch.stack(
        [dataset.actions_for_index(index) for index in range(len(dataset))]
    )
    methods = (
        "model_correct_action",
        "persistence",
        "model_zero_action",
        "model_shuffled_action",
        "model_reversed_action",
        "model_repeat_last_action",
    )
    totals = {
        horizon: {method: ErrorTotals() for method in methods}
        for horizon in ordered_horizons
    }
    deltas = {
        horizon: {
            "prediction_delta_zero_mse": SquaredErrorTotals(),
            "prediction_delta_shuffled_mse": SquaredErrorTotals(),
            "prediction_delta_reversed_mse": SquaredErrorTotals(),
            "prediction_delta_repeat_last_mse": SquaredErrorTotals(),
        }
        for horizon in ordered_horizons
    }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for indices, context, future, correct_actions in loader:
            context = context.to(device)
            future = future.to(device)
            correct_actions = correct_actions.to(device)
            context_actions = torch.stack(
                [dataset.context_actions_for_index(int(index)) for index in indices]
            ).to(device)

            shuffled_actions = action_bank[
                torch.from_numpy(shuffle_indices[indices.numpy()])
            ].to(device)
            zero_actions = torch.zeros_like(correct_actions)
            reversed_actions = reverse_action_chunks(
                correct_actions, dataset.skip_frame
            )
            repeat_last_actions = repeat_last_action_chunks(
                correct_actions, dataset.skip_frame
            )

            truth = model._aggregate(torch.cat([context, future], dim=1))
            target = truth[:, model.cfg.T :]
            persistence = truth[:, model.cfg.T - 1 : model.cfg.T].expand_as(target)
            latent_history = truth[:, : model.cfg.T]

            def predict(candidate_actions: torch.Tensor) -> torch.Tensor:
                return rollout_predictions(
                    model,
                    dataset,
                    indices,
                    context,
                    candidate_actions,
                    context_actions,
                    latent_history=latent_history,
                )

            correct = predict(correct_actions)
            zero = predict(zero_actions)
            shuffled = predict(shuffled_actions)
            reversed_prediction = predict(reversed_actions)
            repeat_last = predict(repeat_last_actions)

            expected = (
                context.shape[0],
                dataset.horizon,
                model.cfg.K,
                model.cfg.D,
            )
            for name, value in (
                ("target", target),
                ("correct", correct),
                ("zero", zero),
                ("shuffled", shuffled),
                ("reversed", reversed_prediction),
                ("repeat_last", repeat_last),
            ):
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"{name} has shape {tuple(value.shape)}, expected {expected}"
                    )
                if not torch.isfinite(value).all().item():
                    raise RuntimeError(f"{name} contains NaN or Inf")

            predictions = {
                "model_correct_action": correct,
                "persistence": persistence,
                "model_zero_action": zero,
                "model_shuffled_action": shuffled,
                "model_reversed_action": reversed_prediction,
                "model_repeat_last_action": repeat_last,
            }
            for horizon in ordered_horizons:
                step = horizon - 1
                for method, prediction in predictions.items():
                    totals[horizon][method].update(
                        prediction[:, step], target[:, step]
                    )
                deltas[horizon]["prediction_delta_zero_mse"].update(
                    correct[:, step], zero[:, step]
                )
                deltas[horizon]["prediction_delta_shuffled_mse"].update(
                    correct[:, step], shuffled[:, step]
                )
                deltas[horizon]["prediction_delta_reversed_mse"].update(
                    correct[:, step], reversed_prediction[:, step]
                )
                deltas[horizon]["prediction_delta_repeat_last_mse"].update(
                    correct[:, step], repeat_last[:, step]
                )

    per_horizon: dict[str, object] = {}
    for horizon in ordered_horizons:
        row = {
            method: accumulator.result()
            for method, accumulator in totals[horizon].items()
        }
        correct_mse = row["model_correct_action"]["mse"]
        denominator = max(correct_mse, 1e-12)
        row.update(
            {
                "gain_vs_persistence": row["persistence"]["mse"] / denominator,
                "gain_vs_zero_action": row["model_zero_action"]["mse"] / denominator,
                "gain_vs_shuffled_action": (
                    row["model_shuffled_action"]["mse"] / denominator
                ),
                "gain_vs_reversed_action": (
                    row["model_reversed_action"]["mse"] / denominator
                ),
                "gain_vs_repeat_last_action": (
                    row["model_repeat_last_action"]["mse"] / denominator
                ),
                "prediction_delta_zero_mse": deltas[horizon][
                    "prediction_delta_zero_mse"
                ].result(),
                "prediction_delta_shuffled_mse": deltas[horizon][
                    "prediction_delta_shuffled_mse"
                ].result(),
                "prediction_delta_reversed_mse": deltas[horizon][
                    "prediction_delta_reversed_mse"
                ].result(),
                "prediction_delta_repeat_last_mse": deltas[horizon][
                    "prediction_delta_repeat_last_mse"
                ].result(),
            }
        )
        shuffle_gap = row["gain_vs_shuffled_action"] - 1.0
        reverse_gap = row["gain_vs_reversed_action"] - 1.0
        repeat_last_gap = row["gain_vs_repeat_last_action"] - 1.0
        row["shuffle_degradation_percent"] = 100.0 * shuffle_gap
        row["reverse_degradation_percent"] = 100.0 * reverse_gap
        row["repeat_last_degradation_percent"] = 100.0 * repeat_last_gap
        row["action_control_success_10pct"] = bool(shuffle_gap >= 0.10)
        per_horizon[str(horizon)] = row
    return {
        "window_count": len(dataset),
        "per_horizon": per_horizon,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(
    checkpoint: Path, device: torch.device
) -> tuple[WorldModel, dict]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    required = {"step", "model", "model_config", "train_config"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"checkpoint is missing keys: {missing}")
    if state["train_config"].get("task") != "LiftCube-v0":
        raise ValueError(
            f"checkpoint task must be LiftCube-v0, got "
            f"{state['train_config'].get('task')!r}"
        )
    cfg = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(WorldModelConfig()),
            OmegaConf.create(state["model_config"]),
        )
    )
    if cfg.C not in (3, 6):
        raise ValueError(f"LiftCube evaluator requires C=3 or C=6, got {cfg.C}")
    if cfg.action.envs["maniskill2"].dim != ACTION_DIM:
        raise ValueError(
            f"LiftCube evaluator requires action dim {ACTION_DIM}, got "
            f"{cfg.action.envs['maniskill2'].dim}"
        )
    model = WorldModel(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_report(
    *,
    checkpoint: Path,
    data_root: Path,
    state: dict,
    dataset: LiftCubeRolloutDataset,
    metrics: dict[str, object],
    horizons: tuple[int, ...],
    batch_size: int,
    seed: int,
    elapsed_sec: float,
) -> dict[str, object]:
    manifest = data_root / "manifest.json"
    return {
        "schema_version": 1,
        "task": "LiftCube-v0",
        "split": "val",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "step": int(state["step"]),
        },
        "data": {
            "root": str(data_root.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "episode_count": len(dataset.episode_ids),
            "include_rgb": bool(dataset.include_rgb),
        },
        "model_config": state["model_config"],
        "evaluation": {
            "T": int(dataset.T),
            "skip_frame": int(getattr(dataset, "skip_frame", 1)),
            "K": int(dataset.point_count),
            "horizons": list(horizons),
            "window_count": int(metrics["window_count"]),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "elapsed_sec": float(elapsed_sec),
        },
        "per_horizon": metrics["per_horizon"],
    }


def print_summary(report: dict[str, object]) -> None:
    print(
        "H  model_mse  persist_mse  zero_mse  shuffled_mse  reversed_mse  "
        "repeat_last_mse  shuffle_gap%  reverse_gap%  repeat_gap%  action_ok"
    )
    for horizon in report["evaluation"]["horizons"]:
        row = report["per_horizon"][str(horizon)]
        print(
            f"{horizon:<2} "
            f"{row['model_correct_action']['mse']:.6e} "
            f"{row['persistence']['mse']:.6e} "
            f"{row['model_zero_action']['mse']:.6e} "
            f"{row['model_shuffled_action']['mse']:.6e} "
            f"{row['model_reversed_action']['mse']:.6e} "
            f"{row['model_repeat_last_action']['mse']:.6e} "
            f"{row['shuffle_degradation_percent']:.2f} "
            f"{row['reverse_degradation_percent']:.2f} "
            f"{row['repeat_last_degradation_percent']:.2f} "
            f"{row['action_control_success_10pct']}"
        )


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Evaluate LiftCube multi-step latent rollouts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    horizons = tuple(sorted(set(args.horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    device = torch.device(args.device)
    model, state = load_checkpoint_model(args.checkpoint, device)
    model_channels = int(
        getattr(model.cfg, "C", state.get("model_config", {}).get("C", 3))
    )
    dataset = LiftCubeRolloutDataset(
        args.data_root,
        T=model.cfg.T,
        point_count=model.cfg.K,
        horizon=max(horizons),
        skip_frame=getattr(model.cfg, "skip_frame", 1),
        include_rgb=model_channels == 6,
    )
    started = time.perf_counter()
    metrics = evaluate_rollouts(
        model,
        dataset,
        batch_size=args.batch_size,
        device=device,
        horizons=horizons,
        seed=args.seed,
    )
    elapsed_sec = time.perf_counter() - started
    report = build_report(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        state=state,
        dataset=dataset,
        metrics=metrics,
        horizons=horizons,
        batch_size=args.batch_size,
        seed=args.seed,
        elapsed_sec=elapsed_sec,
    )
    output = args.output or args.checkpoint.parent / "rollout_metrics.json"
    atomic_write_json(output, report)
    print_summary(report)
    print(f"wrote {output}")
    return report


if __name__ == "__main__":
    main()
