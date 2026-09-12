"""Evaluate the trained LiftCube contact head with class-conditional metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.maniskill import CrossEpisodeBatchSampler, ManiSkillSequenceDataset
from scripts.eval_maniskill_group1_rollout import (
    atomic_write_json,
    load_checkpoint_model,
    sha256_file,
)
from scripts.train_maniskill_group1 import _wrong_actions


def contact_classification_metrics(
    probability: np.ndarray, target: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("probability and target must have the same non-empty shape")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must be finite and in [0,1]")
    if not np.all((target == 0) | (target == 1)):
        raise ValueError("target must be binary")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    target = target.astype(bool)
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    positive = tp + fn
    negative = tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive if positive else float("nan")
    specificity = tn / negative if negative else float("nan")
    order = np.argsort(probability, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    # Average tied ranks so AUROC remains tie-correct.
    for value in np.unique(probability):
        mask = probability == value
        ranks[mask] = ranks[mask].mean()
    auroc = (
        (ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative)
        if positive and negative
        else float("nan")
    )
    eps = 1e-7
    bce = -np.mean(
        target * np.log(np.clip(probability, eps, 1 - eps))
        + (~target) * np.log(np.clip(1 - probability, eps, 1 - eps))
    )
    return {
        "samples": int(probability.size),
        "positive": int(positive),
        "negative": int(negative),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / probability.size),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "auroc": float(auroc),
        "bce": float(bce),
        "mean_probability_positive": float(probability[target].mean())
        if positive
        else float("nan"),
        "mean_probability_negative": float(probability[~target].mean())
        if negative
        else float("nan"),
    }


@torch.no_grad()
def evaluate_contact_head(
    checkpoint: Path,
    data_root: Path,
    contact_sidecar: Path,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    model, state = load_checkpoint_model(checkpoint, device)
    if model.contact_head is None:
        raise RuntimeError("checkpoint has no contact head")
    dataset = ManiSkillSequenceDataset(
        data_root,
        split=split,
        T=int(model.cfg.T),
        point_count=int(model.cfg.K),
        skip_frame=int(model.cfg.skip_frame),
        include_rgb=int(model.cfg.C) == 6,
        include_proprio=True,
        contact_sidecar=contact_sidecar,
        phase_repeat=1,
    )
    batches = CrossEpisodeBatchSampler(
        dataset, batch_size, shuffle=False, drop_last=False, seed=0
    )
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0)
    probabilities = []
    targets = []
    model.eval()
    for batch in loader:
        actions = batch["actions"].to(device)
        output = model(
            batch["points"].to(device),
            actions,
            env="maniskill2",
            negative_actions=_wrong_actions(actions, int(model.cfg.skip_frame)),
            proprio=batch["proprio"].to(device),
            object_pos=batch["object_pos"].to(device),
            cube_robot_contact=batch["cube_robot_contact"].to(device),
        )
        probabilities.append(output["contact_probability"].cpu().numpy())
        targets.append(batch["cube_robot_contact"][:, 1:].numpy())
    probability = np.concatenate(probabilities, axis=0)
    target = np.concatenate(targets, axis=0)
    return {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "step": int(state["step"]),
        },
        "data_root": str(data_root.resolve()),
        "contact_sidecar": {
            "path": str(contact_sidecar.resolve()),
            "sha256": sha256_file(contact_sidecar),
        },
        "split": split,
        "windows": len(dataset),
        "overall": contact_classification_metrics(probability, target),
        "per_horizon": [
            {
                "horizon": index + 1,
                **contact_classification_metrics(probability[:, index], target[:, index]),
            }
            for index in range(probability.shape[1])
        ],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contact-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = evaluate_contact_head(
        args.checkpoint,
        args.data_root,
        args.contact_sidecar,
        split=args.split,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
