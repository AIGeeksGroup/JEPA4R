"""Offline validation metrics for Group 1's latent prediction."""

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


def offline_metrics(prediction: torch.Tensor, target: torch.Tensor, persistence: torch.Tensor) -> dict[str, float]:
    """Compare action-conditioned predictions with a latent persistence baseline."""
    if prediction.shape != target.shape or persistence.shape != target.shape:
        raise ValueError("prediction, target and persistence must have the same shape")
    pred_mse = float((prediction - target).pow(2).mean().item())
    persistence_mse = float((persistence - target).pow(2).mean().item())
    return {
        "pred_mse": pred_mse,
        "persistence_mse": persistence_mse,
        "prediction_gain": persistence_mse / max(pred_mse, 1e-12),
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


def run_evaluation(checkpoint: Path, data_root: Path, max_batches: int, batch_size: int, device: torch.device) -> dict[str, object]:
    model, state = _load_model(checkpoint, device)
    task_root = data_root / "LiftCube-v0"
    dataset = ManiSkillSequenceDataset(task_root, split="val", T=model.cfg.T, point_count=model.cfg.K)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_sum = persistence_sum = 0.0
    count = 0
    with torch.no_grad():
        for index, (points, actions) in enumerate(loader):
            points, actions = points.to(device), actions.to(device)
            output = model(points, actions, env="maniskill2")
            latent1_all = model._aggregate(points)
            metrics = offline_metrics(output["pred"], output["target"], latent1_all[:, model.cfg.T - 1])
            batch_count = output["target"].numel()
            prediction_sum += metrics["pred_mse"] * batch_count
            persistence_sum += metrics["persistence_mse"] * batch_count
            count += batch_count
            if max_batches > 0 and index + 1 >= max_batches:
                break
    if count == 0:
        raise RuntimeError("validation loader is empty")
    pred_mse = prediction_sum / count
    persistence_mse = persistence_sum / count
    result = {
        "task": "LiftCube-v0",
        "split": "val",
        "checkpoint_step": int(state["step"]),
        "batches": index + 1,
        "pred_mse": pred_mse,
        "persistence_mse": persistence_mse,
        "prediction_gain": persistence_mse / max(pred_mse, 1e-12),
    }
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
    result = run_evaluation(args.checkpoint, args.data_root, args.max_batches, args.batch_size, device)
    output = args.output or args.checkpoint.parent / "offline_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
