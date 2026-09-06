"""Train JEPA4R on compact LiftCube-v0 trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ManiSkillSequenceDataset, action_stats
from data.maniskill import CrossEpisodeBatchSampler
from losses import LossConfig
from models import WorldModel, WorldModelConfig
from models.action import ActionConfig, EnvSpec
from models.aggregator import AggregatorConfig
from models.ordering import OrderingConfig
from models.predictor import PredictorConfig
from models.temporal import TemporalConfig


@dataclass
class LiftCubeTrainConfig:
    task: str = "LiftCube-v0"
    env: str = "maniskill2"
    data_root: str = "data/maniskill2/v0.5.3/compact"
    run_root: str = "runs"
    run_name: str = "liftcube_baseline"
    steps: int = 10000
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50
    val_every: int = 500
    checkpoint_every: int = 500
    val_batches: int = 64
    device: str = "cuda"


def validate_liftcube_manifest(root: Path) -> dict:
    """Require the supported LiftCube split: 90 train and 10 validation episodes."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing LiftCube manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "LiftCube-v0":
        raise ValueError(f"Group 1 requires task LiftCube-v0, got {manifest.get('task')!r}")
    episode_ids = [int(x) for x in manifest.get("episode_ids", [])]
    train_ids = [int(x) for x in manifest.get("train_episode_ids", [])]
    val_ids = [int(x) for x in manifest.get("val_episode_ids", [])]
    if len(episode_ids) != 100 or len(train_ids) != 90 or len(val_ids) != 10:
        raise ValueError(
            "LiftCube training requires 100 total episodes split as 90 train / 10 val; "
            f"got {len(episode_ids)} / {len(train_ids)} / {len(val_ids)}"
        )
    if len(set(episode_ids)) != 100 or set(train_ids) | set(val_ids) != set(episode_ids):
        raise ValueError("LiftCube episode ids must cover the manifest exactly")
    if set(train_ids) & set(val_ids):
        raise ValueError("LiftCube train and validation episodes must be disjoint")
    return manifest


def load_liftcube_config(path: Path, overrides: Sequence[str]) -> tuple[WorldModelConfig, LiftCubeTrainConfig]:
    """Load model and training config with OmegaConf dotlist overrides."""
    model_cfg = WorldModelConfig.from_yaml(str(path), overrides)
    cfg = OmegaConf.merge(OmegaConf.structured(LiftCubeTrainConfig()), OmegaConf.load(path).train)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o[len("train.") :] for o in overrides if o.startswith("train.")]))
    train_cfg = OmegaConf.to_object(cfg)
    if train_cfg.task != "LiftCube-v0" or train_cfg.env != "maniskill2":
        raise ValueError("LiftCube config must use task=LiftCube-v0 and env=maniskill2")
    return model_cfg, train_cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_gradients(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return bool(grads) and all(torch.isfinite(g).all().item() for g in grads)


def _wrong_actions(actions: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Keep valid action chunks while breaking their state alignment."""
    if actions.dim() != 3 or chunk_size < 1 or actions.shape[1] % chunk_size:
        raise ValueError("actions must be [B,M,A] with M divisible by chunk_size")
    if actions.shape[0] > 1:
        return actions.roll(1, dims=0)
    chunks = actions.view(1, actions.shape[1] // chunk_size, chunk_size, actions.shape[2])
    if chunks.shape[1] < 2:
        raise ValueError("batch-size-one ranking requires at least two action chunks")
    return chunks.roll(1, dims=1).reshape_as(actions)


def _forward(model: WorldModel, points: torch.Tensor, actions: torch.Tensor, env: str) -> dict:
    negative = _wrong_actions(actions, model.cfg.skip_frame) if model.cfg.loss.ranking_weight > 0 else None
    return model(points, actions, env=env, negative_actions=negative)


def run_preflight(model: WorldModel, loader: DataLoader, device: torch.device, env: str) -> dict:
    """Run one non-updating forward/backward pass and report memory usage."""
    points, actions = next(iter(loader))
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = _forward(model, points.to(device), actions.to(device), env)
    if not torch.isfinite(output["loss"]).item():
        raise RuntimeError("preflight produced a non-finite loss")
    output["loss"].backward()
    if not _finite_gradients(model):
        raise RuntimeError("preflight produced missing or non-finite gradients")
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    model.zero_grad(set_to_none=True)
    return {
        "points_shape": list(points.shape),
        "actions_shape": list(actions.shape),
        "loss": float(output["loss"].detach().cpu()),
        "pred_loss": float(output["pred_loss"].detach().cpu()),
        "latent_loss": float(output["latent_loss"].detach().cpu()),
        "flow_loss": float(output["flow_loss"].detach().cpu()),
        "rank_loss": float(output["rank_loss"].detach().cpu()),
        "reg_loss": float(output["reg_loss"].detach().cpu()),
        "peak_memory_allocated": peak_allocated,
        "peak_memory_reserved": peak_reserved,
    }


def _set_action_stats(model_cfg: WorldModelConfig, mean: np.ndarray, std: np.ndarray) -> None:
    spec: EnvSpec = model_cfg.action.envs["maniskill2"]
    spec.mean = mean.tolist()
    spec.std = std.tolist()


def _evaluate(model: WorldModel, loader: DataLoader, device: torch.device, env: str, max_batches: int) -> dict:
    model.eval()
    total = {"loss": 0.0, "pred_loss": 0.0, "latent_loss": 0.0, "flow_loss": 0.0,
             "rank_loss": 0.0, "reg_loss": 0.0}
    count = 0
    with torch.no_grad():
        for points, actions in loader:
            output = _forward(model, points.to(device), actions.to(device), env)
            for key in total:
                total[key] += float(output[key].detach().cpu())
            count += 1
            if max_batches > 0 and count >= max_batches:
                break
    if count == 0:
        raise RuntimeError("validation loader is empty")
    return {key: value / count for key, value in total.items()}


def _save_checkpoint(path: Path, model: WorldModel, optimizer: torch.optim.Optimizer, step: int, best_val: float, model_cfg, train_cfg, mean, std) -> None:
    state = {
        "step": step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": OmegaConf.to_container(OmegaConf.structured(model_cfg), resolve=True),
        "train_config": OmegaConf.to_container(OmegaConf.structured(train_cfg), resolve=True),
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _prepare_run_dir(train_cfg: LiftCubeTrainConfig, run_dir: Path | None, resume: Path | None) -> Path:
    if run_dir is not None:
        target = run_dir
    else:
        target = Path(train_cfg.run_root) / train_cfg.run_name
    if target.exists() and any(target.iterdir()) and resume is None:
        raise FileExistsError(f"run directory is not empty; pass --resume explicitly: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def train(model_cfg: WorldModelConfig, train_cfg: LiftCubeTrainConfig, data_root: Path, run_dir: Path, device: torch.device, resume: Path | None) -> None:
    root = data_root / "LiftCube-v0"
    manifest = validate_liftcube_manifest(root)
    mean, std = action_stats(root, split="train")
    _set_action_stats(model_cfg, mean, std)
    train_ds = ManiSkillSequenceDataset(root, split="train", T=model_cfg.T, point_count=model_cfg.K, skip_frame=model_cfg.skip_frame)
    val_ds = ManiSkillSequenceDataset(root, split="val", T=model_cfg.T, point_count=model_cfg.K, skip_frame=model_cfg.skip_frame)
    train_batches = CrossEpisodeBatchSampler(
        train_ds, train_cfg.batch_size, shuffle=True, drop_last=True, seed=train_cfg.seed,
    )
    val_batches = CrossEpisodeBatchSampler(
        val_ds, train_cfg.batch_size, shuffle=False, drop_last=False, seed=train_cfg.seed,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_batches, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_sampler=val_batches, num_workers=train_cfg.num_workers)
    model = WorldModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    start_step, best_val = 0, float("inf")
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state.get("best_val", best_val))
    (run_dir / "config.json").write_text(json.dumps({"model": OmegaConf.to_container(OmegaConf.structured(model_cfg), resolve=True), "train": OmegaConf.to_container(OmegaConf.structured(train_cfg), resolve=True)}, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    iterator = iter(train_loader)
    start_time = time.time()
    step = start_step
    while step < train_cfg.steps:
        try:
            points, actions = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            points, actions = next(iterator)
        model.train()
        output = _forward(model, points.to(device), actions.to(device), train_cfg.env)
        if not torch.isfinite(output["loss"]).item():
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        if not _finite_gradients(model):
            raise RuntimeError(f"non-finite or missing gradients at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        step += 1
        record = {"step": step, "train_loss": float(output["loss"].detach().cpu()), "pred_loss": float(output["pred_loss"].detach().cpu()), "latent_loss": float(output["latent_loss"].detach().cpu()), "flow_loss": float(output["flow_loss"].detach().cpu()), "rank_loss": float(output["rank_loss"].detach().cpu()), "reg_loss": float(output["reg_loss"].detach().cpu()), "elapsed_sec": time.time() - start_time}
        if step % train_cfg.val_every == 0 or step == train_cfg.steps:
            validation = _evaluate(model, val_loader, device, train_cfg.env, train_cfg.val_batches)
            record.update({f"val_{key}": value for key, value in validation.items()})
            if validation["loss"] < best_val:
                best_val = validation["loss"]
                _save_checkpoint(run_dir / "best.pt", model, optimizer, step, best_val, model_cfg, train_cfg, mean, std)
        if step % train_cfg.checkpoint_every == 0 or step == train_cfg.steps:
            _save_checkpoint(run_dir / "last.pt", model, optimizer, step, best_val, model_cfg, train_cfg, mean, std)
        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.steps:
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            print(f"step {step:5d} | loss {record['train_loss']:.4f} | pred {record['pred_loss']:.4e} | rank {record['rank_loss']:.4e} | reg {record['reg_loss']:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/liftcube_baseline.yaml"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--preflight", action="store_true", help="run one forward/backward pass and stop")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    model_cfg, train_cfg = load_liftcube_config(args.config, args.overrides)
    _seed_everything(train_cfg.seed)
    data_root = Path(train_cfg.data_root)
    root = data_root / "LiftCube-v0"
    manifest = validate_liftcube_manifest(root)
    mean, std = action_stats(root, split="train")
    _set_action_stats(model_cfg, mean, std)
    dataset = ManiSkillSequenceDataset(root, split="train", T=model_cfg.T, point_count=model_cfg.K, skip_frame=model_cfg.skip_frame)
    batches = CrossEpisodeBatchSampler(
        dataset, train_cfg.batch_size, shuffle=False, drop_last=True, seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=train_cfg.num_workers)
    requested = torch.device(train_cfg.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    print(f"task={manifest['task']} train_episodes={len(manifest['train_episode_ids'])} val_episodes={len(manifest['val_episode_ids'])} windows={len(dataset)} device={device}")
    model = WorldModel(model_cfg).to(device)
    preflight = run_preflight(model, loader, device, train_cfg.env)
    print(json.dumps({"preflight": preflight}, sort_keys=True))
    if args.preflight:
        return
    run_dir = _prepare_run_dir(train_cfg, args.run_dir, args.resume)
    train(model_cfg, train_cfg, data_root, run_dir, device, args.resume)


if __name__ == "__main__":
    main()
