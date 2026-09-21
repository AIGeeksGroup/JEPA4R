"""Reload one PCD/MS2 checkpoint and score an explicit UR5e holdout dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from module import SIGReg
from utils import PointCloudPreprocessor, get_column_normalizer


def checkpoint_filename(epoch: int) -> str:
    if epoch < 1:
        raise ValueError("epoch must be positive")
    return f"weights_epoch_{epoch}.pt"


def weighted_mean(values):
    if not values:
        raise ValueError("no values")
    weight = sum(count for _, count in values)
    if weight <= 0:
        raise ValueError("total weight must be positive")
    return sum(value * count for value, count in values) / weight


def fixed_full_batches(size: int, batch_size: int, seed: int):
    if size < 0:
        raise ValueError("size must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    retained = size - size % batch_size
    indices = torch.randperm(size, generator=generator)[:retained].tolist()
    return [
        indices[start : start + batch_size]
        for start in range(0, retained, batch_size)
    ]


def episode_diversity(clip_indices, batches):
    if not batches:
        raise ValueError("no batches")
    unique_counts = [
        len({clip_indices[index][0] for index in batch}) for batch in batches
    ]
    return {
        "min_unique_episodes_per_batch": min(unique_counts),
        "max_unique_episodes_per_batch": max(unique_counts),
        "mean_unique_episodes_per_batch": sum(unique_counts) / len(unique_counts),
    }


class LatentStatistics:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_squares = None
        self.cross_product = None

    def update(self, tensor):
        values = tensor.detach().reshape(-1, tensor.shape[-1]).to(
            device="cpu", dtype=torch.float64
        )
        if not torch.isfinite(values).all():
            raise AssertionError("non-finite latent value")
        if self.sum is None:
            dimensions = values.shape[1]
            self.sum = torch.zeros(dimensions, dtype=torch.float64)
            self.sum_squares = torch.zeros(dimensions, dtype=torch.float64)
            self.cross_product = torch.zeros(
                (dimensions, dimensions), dtype=torch.float64
            )
        if values.shape[1] != self.sum.shape[0]:
            raise ValueError("latent dimension changed")
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.sum_squares += values.square().sum(dim=0)
        self.cross_product += values.T @ values

    def report(self):
        if not self.count:
            raise ValueError("no latent values")
        mean = self.sum / self.count
        variance = (self.sum_squares / self.count - mean.square()).clamp_min(0)
        covariance = self.cross_product / self.count - torch.outer(mean, mean)
        covariance = (covariance + covariance.T) / 2
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        if not torch.isfinite(eigenvalues).all():
            raise AssertionError("non-finite covariance eigenvalue")
        tolerance = (
            eigenvalues.max()
            * max(eigenvalues.numel(), 1)
            * torch.finfo(eigenvalues.dtype).eps
        )
        positive = eigenvalues[eigenvalues > tolerance]
        if positive.numel():
            probabilities = positive / positive.sum()
            effective_rank = torch.exp(
                -(probabilities * probabilities.log()).sum()
            ).item()
        else:
            effective_rank = 0.0
        return {
            "tokens": self.count,
            "dimensions": mean.numel(),
            "feature_mean": mean.tolist(),
            "feature_variance": variance.tolist(),
            "mean_abs_feature_mean": mean.abs().mean().item(),
            "mean_feature_variance": variance.mean().item(),
            "covariance_effective_rank": effective_rank,
        }


def batch_norm_statistics(model):
    reports = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        mean = module.running_mean
        variance = module.running_var
        finite = (
            mean is not None
            and variance is not None
            and torch.isfinite(mean).all().item()
            and torch.isfinite(variance).all().item()
        )
        reports.append(
            {
                "name": name,
                "type": type(module).__name__,
                "num_features": module.num_features,
                "num_batches_tracked": (
                    int(module.num_batches_tracked.item())
                    if module.num_batches_tracked is not None
                    else None
                ),
                "finite": bool(finite),
                "running_mean_abs_max": (
                    float(mean.abs().max()) if mean is not None else None
                ),
                "running_var_min": (
                    float(variance.min()) if variance is not None else None
                ),
                "running_var_max": (
                    float(variance.max()) if variance is not None else None
                ),
            }
        )
    return reports


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--holdout-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint-run", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()

    filename = checkpoint_filename(args.epoch)
    checkpoint_root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    checkpoint_path = checkpoint_root / args.checkpoint_run / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    dataset_kwargs = {
        "frameskip": 5,
        "num_steps": 4,
        "keys_to_load": ["pcd_xyz", "pcd_rgb", "action", "observation"],
        "keys_to_cache": ["action", "observation"],
    }
    train_dataset = swm.data.HDF5Dataset(path=args.train_dataset, **dataset_kwargs)
    holdout = swm.data.HDF5Dataset(path=args.holdout_dataset, **dataset_kwargs)
    transform = spt.data.transforms.Compose(
        PointCloudPreprocessor(),
        get_column_normalizer(train_dataset, "action", "action"),
        get_column_normalizer(train_dataset, "observation", "observation"),
    )
    holdout.transform = transform
    batches = fixed_full_batches(len(holdout), args.batch_size, args.seed)
    if args.max_batches:
        batches = batches[: args.max_batches]
    loader = torch.utils.data.DataLoader(
        holdout,
        batch_sampler=batches,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(
        f"{args.checkpoint_run}/{filename}"
    ).to(device).eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device).eval()
    losses = []
    sigreg_losses = []
    latent_statistics = LatentStatistics()
    examples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {
                key: torch.nan_to_num(value.to(device), 0.0)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            output = model.encode(batch)
            prediction = model.predict(
                output["emb"][:, :3],
                output["act_emb"][:, :3],
            )
            target = output["emb"][:, 1:]
            loss = (prediction - target).pow(2).mean()
            sigreg_loss = sigreg(output["emb"].transpose(0, 1))
            tensors = [
                output["emb"],
                output["act_emb"],
                prediction,
                target,
                loss,
                sigreg_loss,
            ]
            if not all(torch.isfinite(value).all() for value in tensors):
                raise AssertionError(f"non-finite model output in batch {batch_index}")
            count = int(output["emb"].shape[0])
            losses.append((float(loss), count))
            sigreg_losses.append((float(sigreg_loss), count))
            latent_statistics.update(output["emb"])
            examples += count

    batch_norm = batch_norm_statistics(model)
    if not all(item["finite"] for item in batch_norm):
        raise AssertionError("non-finite BatchNorm running statistics")
    report = {
        "status": "PASS",
        "protocol_version": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_dataset": str(args.train_dataset.resolve()),
        "holdout_dataset": str(args.holdout_dataset.resolve()),
        "device": str(device),
        "batches": len(losses),
        "examples": examples,
        "mean_prediction_loss": weighted_mean(losses),
        "mean_sigreg_loss": weighted_mean(sigreg_losses),
        "max_batches": args.max_batches,
        "sampling": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "drop_last": True,
            "dataset_examples": len(holdout),
            "retained_examples": examples,
            "dropped_examples": len(holdout) - examples,
            **episode_diversity(holdout.clip_indices, batches),
        },
        "latent": latent_statistics.report(),
        "batch_norm": batch_norm,
    }
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
