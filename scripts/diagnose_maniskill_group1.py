"""Diagnose Group 1 latent health after scratch training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ManiSkillSequenceDataset
from models import WorldModel, WorldModelConfig


def representation_diagnostics(latents: torch.Tensor, max_samples: int = 32768) -> dict[str, object]:
    """Report latent spread, effective rank and non-finite values."""
    if latents.dim() != 2:
        raise ValueError(f"latents must have shape [samples, dim], got {tuple(latents.shape)}")
    flat = latents.detach().float()
    if flat.shape[0] > max_samples:
        flat = flat[:max_samples]
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    finite = torch.isfinite(flat).all().item()
    if not finite:
        return {
            "samples": int(flat.shape[0]),
            "latent_std": float("nan"),
            "latent_mean": float("nan"),
            "effective_rank": 0,
            "rank_fraction": 0.0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "collapse_flag": True,
        }
    centered = flat - flat.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    threshold = singular.max().clamp_min(1e-8) * 1e-2
    rank = int((singular > threshold).sum().item())
    dim = flat.shape[-1]
    std = float(flat.std().item())
    return {
        "samples": int(flat.shape[0]),
        "latent_std": std,
        "latent_mean": float(flat.mean().item()),
        "effective_rank": rank,
        "rank_fraction": rank / dim,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "collapse_flag": bool(std < 1e-3 or rank < dim * 0.1),
    }


def _load_model(checkpoint: Path, device: torch.device) -> tuple[WorldModel, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(WorldModelConfig()), OmegaConf.create(state["model_config"]))
    )
    model = WorldModel(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def run_diagnosis(checkpoint: Path, data_root: Path, max_batches: int, batch_size: int, device: torch.device) -> dict[str, object]:
    model, state = _load_model(checkpoint, device)
    train_cfg = state["train_config"]
    task_root = data_root / "LiftCube-v0"
    dataset = ManiSkillSequenceDataset(task_root, split="val", T=model.cfg.T, point_count=model.cfg.K)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    latents = []
    with torch.no_grad():
        for index, (points, _) in enumerate(loader):
            _, latent2 = model.encode(points[:, : model.cfg.T].to(device))
            latents.append(latent2.cpu().reshape(-1, model.cfg.D))
            if max_batches > 0 and index + 1 >= max_batches:
                break
    if not latents:
        raise RuntimeError("validation loader is empty")
    result = representation_diagnostics(torch.cat(latents, dim=0))
    result.update({"task": "LiftCube-v0", "split": "val", "checkpoint_step": int(state["step"]), "train_config": train_cfg})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_diagnosis(args.checkpoint, args.data_root, args.max_batches, args.batch_size, device)
    output = args.output or args.checkpoint.parent / "diagnostics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
