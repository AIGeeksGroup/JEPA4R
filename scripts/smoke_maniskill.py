"""Run one data-to-WorldModel optimization step for all three ManiSkill2 tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ManiSkillSequenceDataset, action_stats
from losses import LossConfig
from models import WorldModel, WorldModelConfig
from models.action import ActionConfig, EnvSpec
from models.aggregator import AggregatorConfig
from models.ordering import OrderingConfig
from models.predictor import PredictorConfig
from models.temporal import TemporalConfig


TASKS = ("LiftCube-v0", "PickCube-v0", "StackCube-v0")


def make_config(T: int, K: int, mean, std) -> WorldModelConfig:
    D = 32
    return WorldModelConfig(
        K=K,
        D=D,
        T=T,
        C=3,
        aggregator=AggregatorConfig(
            ordering=OrderingConfig(order="hilbert", grid_size=0.02, origin=[-0.75, -0.60, -0.05], depth=16),
            order=["z", "hilbert-trans"],
            stride=[2],
            enc_depths=[1, 1],
            enc_channels=[16, 32],
            enc_num_head=[2, 4],
            dec_depths=[1],
            dec_channels=[D],
            dec_num_head=[4],
            patch_size=K,
            drop_path=0.0,
        ),
        temporal=TemporalConfig(num_layers=1, num_heads=4),
        action=ActionConfig(envs={"maniskill2": EnvSpec(dim=7, mean=mean.tolist(), std=std.tolist())}),
        predictor=PredictorConfig(num_layers=1, num_heads=4),
        loss=LossConfig(num_proj=64, chunk=16),
    )


def run_task(task_root: Path, T: int, K: int, batch_size: int, steps: int, device: torch.device) -> None:
    mean, std = action_stats(task_root)
    dataset = ManiSkillSequenceDataset(task_root, split="train", T=T, point_count=K)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = WorldModel(make_config(T, K, mean, std)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    iterator = iter(loader)
    last_loss = None
    for _ in range(steps):
        points, actions = next(iterator, None)
        if points is None:
            iterator = iter(loader)
            points, actions = next(iterator)
        output = model(points.to(device), actions.to(device), env="maniskill2")
        if not torch.isfinite(output["loss"]):
            raise RuntimeError(f"{task_root.name}: non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"{task_root.name}: non-finite or missing gradients")
        optimizer.step()
        last_loss = float(output["loss"].detach().cpu())
    print(f"{task_root.name}: windows={len(dataset)} batch_points={tuple(points.shape)} batch_actions={tuple(actions.shape)} steps={steps} loss={last_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    for task in TASKS:
        run_task(args.data_root / task, args.T, args.K, args.batch_size, args.steps, device)
    print("ManiSkill2 data-to-model smoke: PASS")


if __name__ == "__main__":
    main()
