"""Frozen latent probes for task-relevant LiftCube physical state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.maniskill import ManiSkillSequenceDataset
from planning.frozen_probe import (
    StandardizedLinearProbe,
    binary_metrics,
    continuous_metrics,
    fit_logistic_probe,
    fit_ridge_probe,
)
from scripts.eval_maniskill_group1_rollout import load_checkpoint_model, sha256_file


GRIPPER_CLOSED_THRESHOLD_M = 0.02


class ProbeWindowDataset(ManiSkillSequenceDataset):
    """World-model windows with physical labels at the exact target frame."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        T: int,
        point_count: int,
        skip_frame: int,
        include_rgb: bool = False,
        cache_size: int = 2,
    ):
        super().__init__(
            root,
            split=split,
            T=T,
            point_count=point_count,
            skip_frame=skip_frame,
            include_rgb=include_rgb,
            include_proprio=True,
            cache_size=cache_size,
        )
        self.split = split

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        if not isinstance(sample, dict):
            raise RuntimeError("probe dataset requires proprio-enabled samples")
        episode_id, start = self.index[index]
        target_frame = int(start + self.T * self.skip_frame)
        episode = self._load_episode(episode_id)
        qpos = np.asarray(episode["qpos"][target_frame], dtype=np.float32)
        tcp_position = np.asarray(
            episode["tcp_pose"][target_frame, :3], dtype=np.float32
        )
        object_height = np.float32(episode["object_pos"][target_frame, 2])
        object_position = np.asarray(
            episode["object_pos"][target_frame], dtype=np.float32
        )
        gripper_gap = np.float32(abs(float(qpos[-2])) + abs(float(qpos[-1])))
        return {
            **sample,
            "episode_id": torch.tensor(episode_id, dtype=torch.int64),
            "frame_id": torch.tensor(target_frame, dtype=torch.int64),
            "object_height": torch.tensor(object_height, dtype=torch.float32),
            "object_position": torch.from_numpy(object_position.copy()),
            "tcp_position": torch.from_numpy(tcp_position.copy()),
            "gripper_gap": torch.tensor(gripper_gap, dtype=torch.float32),
            "gripper_closed": torch.tensor(
                int(gripper_gap <= GRIPPER_CLOSED_THRESHOLD_M),
                dtype=torch.int64,
            ),
        }


def pool_scene_tokens(tokens: torch.Tensor | np.ndarray):
    """Use the checkpoint's fixed mean over the scene-token dimension."""
    if tokens.ndim != 3:
        raise ValueError(f"scene tokens must have shape [B,K,D], got {tokens.shape}")
    return tokens.mean(dim=1) if isinstance(tokens, torch.Tensor) else tokens.mean(axis=1)


def append_current_proprio(
    scene_features: torch.Tensor | np.ndarray,
    proprio_normalized: torch.Tensor | np.ndarray,
):
    """Concatenate a pooled scene vector with aligned current robot state."""
    if scene_features.ndim != 2 or proprio_normalized.ndim != 2:
        raise ValueError("scene features and proprio must both be [B,D]")
    if scene_features.shape[0] != proprio_normalized.shape[0]:
        raise ValueError("scene features and proprio batch sizes must match")
    if isinstance(scene_features, torch.Tensor) != isinstance(
        proprio_normalized, torch.Tensor
    ):
        raise TypeError("scene features and proprio must use the same array type")
    if isinstance(scene_features, torch.Tensor):
        return torch.cat([scene_features, proprio_normalized], dim=1)
    return np.concatenate([scene_features, proprio_normalized], axis=1)


@dataclass(frozen=True)
class ExtractedFeatures:
    representations: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    existing_heads: dict[str, np.ndarray]
    episode_id: np.ndarray
    frame_id: np.ndarray


@dataclass(frozen=True)
class ContactSidecar:
    episode_id: np.ndarray
    frame_id: np.ndarray
    cube_robot_contact: np.ndarray
    object_pos_error_m: np.ndarray
    tcp_pos_error_m: np.ndarray


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _forward_probe_boundaries(
    model: torch.nn.Module,
    *,
    points: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor,
    object_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run only the frozen representation path, avoiding training-loss work."""
    if not all(
        hasattr(model, name)
        for name in ("_aggregate", "_predict_sequence", "next_proprio_head")
    ):
        return model(
            points,
            actions,
            env="maniskill2",
            proprio=proprio,
            object_pos=object_pos,
        )
    latent_all = model._aggregate(points)
    normalized_proprio = model.proprio_encoder.normalize(proprio)
    pred_all, temporal_all, decoder_all = model._predict_sequence(
        latent_all[:, : model.cfg.T],
        actions,
        "maniskill2",
        proprio_normalized=normalized_proprio[:, : model.cfg.T],
        return_features=True,
    )
    pooled_decoder = decoder_all.mean(dim=2)
    predicted_proprio_normalized = model.next_proprio_head(pooled_decoder)
    return {
        "target": latent_all[:, model.cfg.T],
        "pred": pred_all[:, -1],
        "latent2": temporal_all[:, -1],
        "next_proprio": model.proprio_encoder.denormalize(
            predicted_proprio_normalized
        ),
        "object_pos": model.object_position_head(pooled_decoder),
    }


def extract_frozen_features(
    model: torch.nn.Module,
    dataset: ProbeWindowDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> ExtractedFeatures:
    """Extract train/validation features without updating the world model."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if getattr(model, "proprio_encoder", None) is None:
        raise ValueError("frozen probe extraction requires an enabled proprio encoder")
    if int(model.cfg.T) != dataset.T:
        raise ValueError(f"model T={model.cfg.T} does not match dataset T={dataset.T}")

    representation_chunks: dict[str, list[np.ndarray]] = {
        "true_spatial_next": [],
        "predicted_spatial_next": [],
        "temporal_current": [],
        "current_proprio_normalized": [],
    }
    label_chunks: dict[str, list[np.ndarray]] = {
        "object_position": [],
        "object_height": [],
        "tcp_position": [],
        "gripper_gap": [],
        "gripper_closed": [],
    }
    head_chunks: dict[str, list[np.ndarray]] = {
        "object_position": [],
        "next_proprio": [],
    }
    episode_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            points = batch["points"].to(device)
            actions = batch["actions"].to(device)
            proprio = batch["proprio"].to(device)
            object_pos = batch["object_pos"].to(device)
            output = _forward_probe_boundaries(
                model,
                points=points,
                actions=actions,
                proprio=proprio,
                object_pos=object_pos,
            )
            current_proprio = model.proprio_encoder.normalize(proprio)[
                :, dataset.T - 1
            ]
            representation_chunks["true_spatial_next"].append(
                _numpy(pool_scene_tokens(output["target"]))
            )
            representation_chunks["predicted_spatial_next"].append(
                _numpy(pool_scene_tokens(output["pred"]))
            )
            representation_chunks["temporal_current"].append(
                _numpy(pool_scene_tokens(output["latent2"]))
            )
            representation_chunks["current_proprio_normalized"].append(
                _numpy(current_proprio)
            )
            for name in label_chunks:
                label_chunks[name].append(_numpy(batch[name]))
            head_chunks["object_position"].append(
                _numpy(output["object_pos"][:, -1])
            )
            head_chunks["next_proprio"].append(
                _numpy(output["next_proprio"][:, -1])
            )
            episode_chunks.append(_numpy(batch["episode_id"]))
            frame_chunks.append(_numpy(batch["frame_id"]))

    return ExtractedFeatures(
        representations={
            name: np.concatenate(chunks, axis=0)
            for name, chunks in representation_chunks.items()
        },
        labels={
            name: np.concatenate(chunks, axis=0) for name, chunks in label_chunks.items()
        },
        existing_heads={
            name: np.concatenate(chunks, axis=0) for name, chunks in head_chunks.items()
        },
        episode_id=np.concatenate(episode_chunks, axis=0).astype(np.int64),
        frame_id=np.concatenate(frame_chunks, axis=0).astype(np.int64),
    )


def _validate_contact_arrays(sidecar: ContactSidecar, tolerance: float) -> None:
    arrays = {
        "episode_id": np.asarray(sidecar.episode_id),
        "frame_id": np.asarray(sidecar.frame_id),
        "cube_robot_contact": np.asarray(sidecar.cube_robot_contact),
        "object_pos_error_m": np.asarray(sidecar.object_pos_error_m),
        "tcp_pos_error_m": np.asarray(sidecar.tcp_pos_error_m),
    }
    lengths = {name: value.shape[0] for name, value in arrays.items() if value.ndim == 1}
    if len(lengths) != len(arrays) or len(set(lengths.values())) != 1:
        raise ValueError("contact sidecar arrays must be same-length vectors")
    if not np.isin(arrays["cube_robot_contact"], (0, 1)).all():
        raise ValueError("cube_robot_contact must contain only 0 and 1")
    errors = np.concatenate(
        [arrays["object_pos_error_m"], arrays["tcp_pos_error_m"]]
    ).astype(np.float64)
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("contact alignment errors must be finite and non-negative")
    if np.any(errors > tolerance):
        raise ValueError(
            f"contact alignment error exceeds tolerance {tolerance:.3e} m; "
            f"maximum is {float(errors.max()):.3e} m"
        )
    keys = list(
        zip(
            arrays["episode_id"].astype(np.int64).tolist(),
            arrays["frame_id"].astype(np.int64).tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (episode_id, frame_id) in contact sidecar")


def load_contact_sidecar(
    path: str | Path,
    *,
    alignment_tolerance_m: float,
) -> ContactSidecar:
    """Load and strictly validate a simulator-derived contact sidecar."""
    if alignment_tolerance_m < 0:
        raise ValueError("alignment_tolerance_m must be non-negative")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing contact sidecar: {path}")
    required = {
        "episode_id",
        "frame_id",
        "cube_robot_contact",
        "object_pos_error_m",
        "tcp_pos_error_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"contact sidecar is missing arrays: {missing}")
        sidecar = ContactSidecar(
            episode_id=np.asarray(archive["episode_id"], dtype=np.int64),
            frame_id=np.asarray(archive["frame_id"], dtype=np.int64),
            cube_robot_contact=np.asarray(
                archive["cube_robot_contact"], dtype=np.uint8
            ),
            object_pos_error_m=np.asarray(
                archive["object_pos_error_m"], dtype=np.float64
            ),
            tcp_pos_error_m=np.asarray(
                archive["tcp_pos_error_m"], dtype=np.float64
            ),
        )
    _validate_contact_arrays(sidecar, float(alignment_tolerance_m))
    return sidecar


def attach_contact_labels(
    extracted: ExtractedFeatures,
    sidecar: ContactSidecar,
) -> ExtractedFeatures:
    """Join exact cube-robot contact labels by episode and compact frame."""
    _validate_contact_arrays(sidecar, float("inf"))
    contact_by_key = {
        (int(episode_id), int(frame_id)): int(contact)
        for episode_id, frame_id, contact in zip(
            sidecar.episode_id,
            sidecar.frame_id,
            sidecar.cube_robot_contact,
        )
    }
    contact: list[int] = []
    for episode_id, frame_id in zip(extracted.episode_id, extracted.frame_id):
        key = (int(episode_id), int(frame_id))
        if key not in contact_by_key:
            raise ValueError(
                f"missing contact label for episode {key[0]} frame {key[1]}"
            )
        contact.append(contact_by_key[key])
    labels = dict(extracted.labels)
    labels["cube_robot_contact"] = np.asarray(contact, dtype=np.int64)
    return ExtractedFeatures(
        representations=extracted.representations,
        labels=labels,
        existing_heads=extracted.existing_heads,
        episode_id=extracted.episode_id,
        frame_id=extracted.frame_id,
    )


def _probe_sha256(probe: StandardizedLinearProbe) -> str:
    digest = hashlib.sha256()
    digest.update(probe.kind.encode("ascii"))
    for array in (
        probe.feature_mean,
        probe.feature_scale,
        probe.coef,
        probe.intercept,
    ):
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(str(probe.output_was_1d).encode("ascii"))
    return digest.hexdigest()


def _fit_probe(
    kind: str,
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
    logistic_l2: float,
    logistic_max_iter: int,
) -> StandardizedLinearProbe:
    if kind == "continuous":
        return fit_ridge_probe(features, target, alpha=ridge_alpha)
    return fit_logistic_probe(
        features,
        target,
        l2=logistic_l2,
        max_iter=logistic_max_iter,
    )


def _probe_metrics(
    kind: str,
    target: np.ndarray,
    prediction: np.ndarray,
    train_target: np.ndarray,
) -> dict[str, object]:
    if kind == "continuous":
        return continuous_metrics(
            target,
            prediction,
            train_mean=np.asarray(train_target).mean(axis=0),
        )
    return binary_metrics(
        target,
        prediction,
        train_positive_rate=float(np.asarray(train_target).mean()),
    )


def _evaluate_fitted_probe(
    *,
    probe: StandardizedLinearProbe,
    kind: str,
    train_target: np.ndarray,
    val_target: np.ndarray,
    val_features: np.ndarray,
    fitted_on: str,
    include_parameters: bool,
) -> tuple[dict[str, object], np.ndarray]:
    prediction = probe.predict(val_features)
    result = _probe_metrics(kind, val_target, prediction, train_target)
    result.update(
        {
            "probe_sha256": _probe_sha256(probe),
            "probe_fitted_on": fitted_on,
        }
    )
    if include_parameters:
        result["probe"] = probe.to_dict()
    return result, prediction


def evaluate_probe_suite(
    train: ExtractedFeatures,
    val: ExtractedFeatures,
    *,
    ridge_alpha: float = 1e-3,
    logistic_l2: float = 1e-3,
    logistic_max_iter: int = 500,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fit on training episodes and evaluate frozen validation features."""
    train_episodes = sorted(set(np.asarray(train.episode_id, dtype=np.int64).tolist()))
    val_episodes = sorted(set(np.asarray(val.episode_id, dtype=np.int64).tolist()))
    overlap = sorted(set(train_episodes).intersection(val_episodes))
    if overlap:
        raise ValueError(f"train and validation episodes overlap: {overlap}")
    target_kinds = {
        "object_height": "continuous",
        "tcp_position": "continuous",
        "gripper_gap": "continuous",
        "gripper_closed": "binary",
        "cube_robot_contact": "binary",
    }
    required_labels = set(target_kinds) | {"object_position"}
    for split_name, extracted in (("train", train), ("val", val)):
        missing = sorted(required_labels - set(extracted.labels))
        if missing:
            raise ValueError(f"{split_name} features are missing labels: {missing}")

    summary: dict[str, object] = {
        "schema_version": 1,
        "fit_split": "train",
        "eval_split": "val",
        "train": {
            "sample_count": int(len(train.episode_id)),
            "episode_ids": train_episodes,
        },
        "val": {
            "sample_count": int(len(val.episode_id)),
            "episode_ids": val_episodes,
        },
        "probe_config": {
            "ridge_alpha": float(ridge_alpha),
            "logistic_l2": float(logistic_l2),
            "logistic_max_iter": int(logistic_max_iter),
            "token_pooling": "mean",
            "gripper_closed_threshold_m": GRIPPER_CLOSED_THRESHOLD_M,
        },
        "targets": {},
        "existing_heads": {},
    }
    predictions: dict[str, np.ndarray] = {}
    train_true = np.asarray(train.representations["true_spatial_next"])
    val_true = np.asarray(val.representations["true_spatial_next"])
    val_predicted = np.asarray(val.representations["predicted_spatial_next"])
    train_temporal = np.asarray(train.representations["temporal_current"])
    val_temporal = np.asarray(val.representations["temporal_current"])
    train_proprio = np.asarray(train.representations["current_proprio_normalized"])
    val_proprio = np.asarray(val.representations["current_proprio_normalized"])
    train_full = append_current_proprio(train_true, train_proprio)
    val_true_full = append_current_proprio(val_true, val_proprio)
    val_predicted_full = append_current_proprio(val_predicted, val_proprio)
    full_state_targets = {
        "tcp_position",
        "gripper_gap",
        "gripper_closed",
        "cube_robot_contact",
    }

    for target_name, kind in target_kinds.items():
        train_target = np.asarray(train.labels[target_name])
        val_target = np.asarray(val.labels[target_name])
        target_summary: dict[str, object] = {
            "type": kind,
            "representations": {},
        }
        if kind == "binary" and len(np.unique(train_target)) < 2:
            target_summary["available"] = False
            target_summary["reason"] = "training target contains only one class"
            summary["targets"][target_name] = target_summary
            continue

        true_probe = _fit_probe(
            kind,
            train_true,
            train_target,
            ridge_alpha=ridge_alpha,
            logistic_l2=logistic_l2,
            logistic_max_iter=logistic_max_iter,
        )
        for representation_name, features, include_parameters in (
            ("true_spatial_next", val_true, True),
            ("predicted_spatial_next_transfer", val_predicted, False),
        ):
            result, prediction = _evaluate_fitted_probe(
                probe=true_probe,
                kind=kind,
                train_target=train_target,
                val_target=val_target,
                val_features=features,
                fitted_on="train:true_spatial_next",
                include_parameters=include_parameters,
            )
            target_summary["representations"][representation_name] = result
            predictions[f"{target_name}__{representation_name}"] = prediction

        temporal_probe = _fit_probe(
            kind,
            train_temporal,
            train_target,
            ridge_alpha=ridge_alpha,
            logistic_l2=logistic_l2,
            logistic_max_iter=logistic_max_iter,
        )
        result, prediction = _evaluate_fitted_probe(
            probe=temporal_probe,
            kind=kind,
            train_target=train_target,
            val_target=val_target,
            val_features=val_temporal,
            fitted_on="train:temporal_current",
            include_parameters=True,
        )
        target_summary["representations"]["temporal_current"] = result
        predictions[f"{target_name}__temporal_current"] = prediction

        if target_name in full_state_targets:
            full_probe = _fit_probe(
                kind,
                train_full,
                train_target,
                ridge_alpha=ridge_alpha,
                logistic_l2=logistic_l2,
                logistic_max_iter=logistic_max_iter,
            )
            for representation_name, features, include_parameters in (
                ("true_spatial_plus_current_proprio", val_true_full, True),
                (
                    "predicted_spatial_plus_current_proprio_transfer",
                    val_predicted_full,
                    False,
                ),
            ):
                result, prediction = _evaluate_fitted_probe(
                    probe=full_probe,
                    kind=kind,
                    train_target=train_target,
                    val_target=val_target,
                    val_features=features,
                    fitted_on="train:true_spatial_plus_current_proprio",
                    include_parameters=include_parameters,
                )
                target_summary["representations"][representation_name] = result
                predictions[f"{target_name}__{representation_name}"] = prediction
        summary["targets"][target_name] = target_summary

    val_object = np.asarray(val.existing_heads["object_position"])
    val_next_proprio = np.asarray(val.existing_heads["next_proprio"])
    val_head_gap = np.abs(val_next_proprio[:, 7:9]).sum(axis=1)
    head_predictions = {
        "object_position": val_object,
        "object_height": val_object[:, 2],
        "tcp_position": val_next_proprio[:, 18:21],
        "gripper_gap": val_head_gap,
        "gripper_closed": (val_head_gap <= GRIPPER_CLOSED_THRESHOLD_M).astype(
            np.float64
        ),
    }
    for target_name, prediction in head_predictions.items():
        kind = "binary" if target_name == "gripper_closed" else "continuous"
        summary["existing_heads"][target_name] = _probe_metrics(
            kind,
            np.asarray(val.labels[target_name]),
            prediction,
            np.asarray(train.labels[target_name]),
        )
        predictions[f"existing_head__{target_name}"] = prediction
    return summary, predictions


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# LiftCube Frozen Latent Probe Report",
        "",
        "This is a frozen representation diagnostic, not a physical-control success test.",
        "",
        "## Probe metrics",
        "",
        "| Target | Representation | Primary metric | Value |",
        "|---|---|---:|---:|",
    ]
    for target_name, target in summary.get("targets", {}).items():
        if not target.get("available", True):
            lines.append(
                f"| {target_name} | unavailable | reason | {target.get('reason')} |"
            )
            continue
        for representation_name, result in target["representations"].items():
            metric_name = "rmse" if target["type"] == "continuous" else "auroc"
            value = result.get(metric_name)
            rendered = "unavailable" if value is None else f"{float(value):.6g}"
            lines.append(
                f"| {target_name} | {representation_name} | {metric_name} | {rendered} |"
            )
    lines.extend(
        [
            "",
            "## Existing checkpoint heads",
            "",
            "| Target | Primary metric | Value |",
            "|---|---:|---:|",
        ]
    )
    for target_name, result in summary.get("existing_heads", {}).items():
        metric_name = "balanced_accuracy" if target_name == "gripper_closed" else "rmse"
        value = result.get(metric_name)
        rendered = "unavailable" if value is None else f"{float(value):.6g}"
        lines.append(f"| {target_name} | {metric_name} | {rendered} |")
    lines.extend(
        [
            "",
            "Interpretation: compare `true_spatial_next` with "
            "`predicted_spatial_next_transfer`; both use the same fitted probe. "
            "A large transfer drop isolates predictor distortion.",
            "",
        ]
    )
    return "\n".join(lines)


def save_probe_outputs(
    output_dir: str | Path,
    *,
    summary: dict[str, object],
    predictions: dict[str, np.ndarray],
    val: ExtractedFeatures,
) -> None:
    output_dir = Path(output_dir)
    arrays: dict[str, np.ndarray] = {
        "episode_id": np.asarray(val.episode_id, dtype=np.int64),
        "frame_id": np.asarray(val.frame_id, dtype=np.int64),
    }
    arrays.update(
        {f"label__{name}": np.asarray(value) for name, value in val.labels.items()}
    )
    arrays.update(
        {
            f"prediction__{name}": np.asarray(value)
            for name, value in predictions.items()
        }
    )
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"output array {name} contains NaN or Inf")
    _atomic_write_npz(output_dir / "predictions.npz", arrays)
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_text(output_dir / "REPORT.md", build_markdown_report(summary))


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contact-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alignment-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--logistic-l2", type=float, default=1e-3)
    parser.add_argument("--logistic-max-iter", type=int, default=500)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, checkpoint_state = load_checkpoint_model(args.checkpoint, device)
    common = {
        "T": int(model.cfg.T),
        "point_count": int(model.cfg.K),
        "skip_frame": int(model.cfg.skip_frame),
        "include_rgb": int(model.cfg.C) == 6,
    }
    train_dataset = ProbeWindowDataset(args.data_root, split="train", **common)
    val_dataset = ProbeWindowDataset(args.data_root, split="val", **common)
    train = extract_frozen_features(
        model, train_dataset, batch_size=args.batch_size, device=device
    )
    val = extract_frozen_features(
        model, val_dataset, batch_size=args.batch_size, device=device
    )
    sidecar = load_contact_sidecar(
        args.contact_sidecar,
        alignment_tolerance_m=args.alignment_tolerance_m,
    )
    train = attach_contact_labels(train, sidecar)
    val = attach_contact_labels(val, sidecar)
    summary, predictions = evaluate_probe_suite(
        train,
        val,
        ridge_alpha=args.ridge_alpha,
        logistic_l2=args.logistic_l2,
        logistic_max_iter=args.logistic_max_iter,
    )
    repeat_summary, repeat_predictions = evaluate_probe_suite(
        train,
        val,
        ridge_alpha=args.ridge_alpha,
        logistic_l2=args.logistic_l2,
        logistic_max_iter=args.logistic_max_iter,
    )
    if json.dumps(summary, sort_keys=True) != json.dumps(repeat_summary, sort_keys=True):
        raise RuntimeError("determinism check failed: repeated summary differs")
    for name in predictions:
        if name not in repeat_predictions or not np.array_equal(
            predictions[name], repeat_predictions[name]
        ):
            raise RuntimeError(
                f"determinism check failed: repeated prediction differs for {name}"
            )
    summary.update(
        {
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint),
                "step": int(checkpoint_state["step"]),
            },
            "data": {
                "root": str(args.data_root.resolve()),
                "manifest_sha256": sha256_file(args.data_root / "manifest.json"),
            },
            "contact_sidecar": {
                "path": str(args.contact_sidecar.resolve()),
                "sha256": sha256_file(args.contact_sidecar),
                "alignment_tolerance_m": float(args.alignment_tolerance_m),
            },
            "extraction": {
                "device": str(device),
                "batch_size": int(args.batch_size),
                **common,
            },
            "determinism": {
                "repeat_fit_exact_summary_match": True,
                "repeat_fit_exact_prediction_match": True,
            },
        }
    )
    save_probe_outputs(
        args.output_dir, summary=summary, predictions=predictions, val=val
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
