"""Prepare and train one compact ManiSkill task with the Group 1 model."""

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
from data.maniskill import load_proprio_stats
from losses import LossConfig
from models import WorldModel, WorldModelConfig
from models.action import ActionConfig, EnvSpec
from models.aggregator import AggregatorConfig
from models.ordering import OrderingConfig
from models.predictor import PredictorConfig
from models.temporal import TemporalConfig


@dataclass
class Group1TrainConfig:
    task: str = "LiftCube-v0"
    expected_episodes: int = 100
    env: str = "maniskill2"
    data_root: str = "/workspace/datasets/maniskill2/v0.5.3/compact"
    run_root: str = "/workspace/users/lewm/runs/4d-jepa-architecture"
    run_name: str = "group1_scratch_liftcube"
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
    include_rgb: bool = False
    include_proprio: bool = False
    contact_sidecar: str | None = None
    contact_alignment_tolerance_m: float = 1e-4
    contact_phase_repeat: int = 1


def task_data_root(data_root: Path, task: str) -> Path:
    """Resolve one task directory without permitting path traversal."""
    if not task or Path(task).name != task:
        raise ValueError(f"invalid ManiSkill task name: {task!r}")
    return data_root / task


def validate_group1_manifest(
    root: Path,
    *,
    expected_task: str = "LiftCube-v0",
    expected_episodes: int = 100,
) -> dict:
    """Enforce a configured task, exact episode count, and a 90/10 split."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Group 1 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_episodes < 2:
        raise ValueError("expected_episodes must be at least two")
    if manifest.get("task") != expected_task:
        raise ValueError(
            f"Group 1 requires task {expected_task}, got {manifest.get('task')!r}"
        )
    episode_ids = [int(x) for x in manifest.get("episode_ids", [])]
    train_ids = [int(x) for x in manifest.get("train_episode_ids", [])]
    val_ids = [int(x) for x in manifest.get("val_episode_ids", [])]
    expected_val = max(1, int(round(expected_episodes * 0.1)))
    expected_train = expected_episodes - expected_val
    if (
        len(episode_ids) != expected_episodes
        or len(train_ids) != expected_train
        or len(val_ids) != expected_val
    ):
        raise ValueError(
            f"Group 1 requires {expected_episodes} total episodes split as "
            f"{expected_train} train / {expected_val} val; "
            f"got {len(episode_ids)} / {len(train_ids)} / {len(val_ids)}"
        )
    if (
        len(set(episode_ids)) != expected_episodes
        or len(set(train_ids)) != expected_train
        or len(set(val_ids)) != expected_val
        or set(train_ids) | set(val_ids) != set(episode_ids)
    ):
        raise ValueError("Group 1 episode ids must cover the manifest exactly")
    if set(train_ids) & set(val_ids):
        raise ValueError("Group 1 train and validation episodes must be disjoint")
    return manifest


def load_group1_config(path: Path, overrides: Sequence[str]) -> tuple[WorldModelConfig, Group1TrainConfig]:
    """Load model and training config with OmegaConf dotlist overrides."""
    model_cfg = WorldModelConfig.from_yaml(str(path), overrides)
    cfg = OmegaConf.merge(OmegaConf.structured(Group1TrainConfig()), OmegaConf.load(path).train)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o[len("train.") :] for o in overrides if o.startswith("train.")]))
    train_cfg = OmegaConf.to_object(cfg)
    if train_cfg.env != "maniskill2":
        raise ValueError("Group 1 config must use env=maniskill2")
    task_data_root(Path(train_cfg.data_root), train_cfg.task)
    if train_cfg.expected_episodes < 2:
        raise ValueError("train.expected_episodes must be at least two")
    _validate_data_channels(model_cfg, train_cfg)
    return model_cfg, train_cfg


def _validate_data_channels(
    model_cfg: WorldModelConfig, train_cfg: Group1TrainConfig
) -> None:
    expected = 6 if train_cfg.include_rgb else 3
    if model_cfg.C != expected:
        raise ValueError(
            f"train.include_rgb={train_cfg.include_rgb} requires model.C={expected}, "
            f"got {model_cfg.C}"
        )
    if model_cfg.proprio.enabled != train_cfg.include_proprio:
        raise ValueError(
            f"train.include_proprio={train_cfg.include_proprio} must match "
            f"model.proprio.enabled={model_cfg.proprio.enabled}"
        )
    contact_enabled = model_cfg.loss.contact_weight > 0
    sidecar_enabled = train_cfg.contact_sidecar is not None
    if contact_enabled != sidecar_enabled:
        raise ValueError(
            "model.loss.contact_weight > 0 and train.contact_sidecar must be enabled together"
        )


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return (
            batch["points"],
            batch["actions"],
            batch["proprio"],
            batch["object_pos"],
            batch.get("cube_robot_contact"),
        )
    points, actions = batch
    return points, actions, None, None, None


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


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


BASE_METRICS = (
    "loss",
    "pred_loss",
    "latent_loss",
    "flow_loss",
    "rank_loss",
    "reg_loss",
)
AUX_METRICS = (
    "proprio_loss",
    "object_loss",
    "qpos_mse",
    "qvel_mse",
    "tcp_mse",
    "object_xyz_mse",
    "object_z_mse",
)
CONTACT_METRICS = (
    "contact_loss",
    "contact_accuracy",
)
RUNNER_METADATA_FILES = {
    "command.txt",
    "config.sha256",
    "git-head.txt",
    "git-status.before.txt",
    "input.sha256",
    "nvidia-smi.before.txt",
    "runtime.txt",
    "train.log",
    "train.pid",
}


def _forward(
    model: WorldModel,
    points: torch.Tensor,
    actions: torch.Tensor,
    env: str,
    *,
    proprio: torch.Tensor | None = None,
    object_pos: torch.Tensor | None = None,
    cube_robot_contact: torch.Tensor | None = None,
) -> dict:
    negative = _wrong_actions(actions, model.cfg.skip_frame) if model.cfg.loss.ranking_weight > 0 else None
    return model(
        points,
        actions,
        env=env,
        negative_actions=negative,
        proprio=proprio,
        object_pos=object_pos,
        cube_robot_contact=cube_robot_contact,
    )


def run_preflight(model: WorldModel, loader: DataLoader, device: torch.device, env: str) -> dict:
    """Run one non-updating forward/backward pass and report memory usage."""
    points, actions, proprio, object_pos, cube_robot_contact = _unpack_batch(next(iter(loader)))
    points = points.to(device)
    actions = actions.to(device).requires_grad_(True)
    if proprio is not None:
        proprio = proprio.to(device).requires_grad_(True)
    if object_pos is not None:
        object_pos = object_pos.to(device)
    if cube_robot_contact is not None:
        cube_robot_contact = cube_robot_contact.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = _forward(
        model,
        points,
        actions,
        env,
        proprio=proprio,
        object_pos=object_pos,
        cube_robot_contact=cube_robot_contact,
    )
    if not torch.isfinite(output["loss"]).item():
        raise RuntimeError("preflight produced a non-finite loss")
    output["loss"].backward()
    if not _finite_gradients(model):
        raise RuntimeError("preflight produced missing or non-finite gradients")
    if actions.grad is None or not torch.isfinite(actions.grad).all() or not actions.grad.abs().sum() > 0:
        raise RuntimeError("preflight produced missing/non-finite/zero action input gradients")
    if proprio is not None and (
        proprio.grad is None
        or not torch.isfinite(proprio.grad).all()
        or not proprio.grad.abs().sum() > 0
    ):
        raise RuntimeError("preflight produced missing/non-finite/zero proprio input gradients")
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    model.zero_grad(set_to_none=True)
    result = {
        "points_shape": list(points.shape),
        "actions_shape": list(actions.shape),
        "proprio_shape": None if proprio is None else list(proprio.shape),
        "object_pos_shape": None if object_pos is None else list(object_pos.shape),
        "cube_robot_contact_shape": None
        if cube_robot_contact is None
        else list(cube_robot_contact.shape),
        "action_input_grad_norm": float(actions.grad.norm().detach().cpu()),
        "proprio_input_grad_norm": 0.0
        if proprio is None
        else float(proprio.grad.norm().detach().cpu()),
        "peak_memory_allocated": peak_allocated,
        "peak_memory_reserved": peak_reserved,
    }
    for name in (*BASE_METRICS, *AUX_METRICS, *CONTACT_METRICS):
        if name in output:
            result[name] = float(output[name].detach().cpu())
    return result


def _set_action_stats(model_cfg: WorldModelConfig, mean: np.ndarray, std: np.ndarray) -> None:
    spec: EnvSpec = model_cfg.action.envs["maniskill2"]
    spec.mean = mean.tolist()
    spec.std = std.tolist()


def _set_proprio_stats(
    model_cfg: WorldModelConfig, mean: np.ndarray, std: np.ndarray
) -> None:
    if not model_cfg.proprio.enabled:
        raise ValueError("cannot set proprio statistics while proprioception is disabled")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (model_cfg.proprio.dim,) or std.shape != (
        model_cfg.proprio.dim,
    ):
        raise ValueError(
            f"proprio mean/std must have shape [{model_cfg.proprio.dim}]"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (
        std > 0
    ).all():
        raise ValueError("proprio mean/std must be finite and std must be positive")
    model_cfg.proprio.mean = mean.tolist()
    model_cfg.proprio.std = std.tolist()


def _evaluate(model: WorldModel, loader: DataLoader, device: torch.device, env: str, max_batches: int) -> dict:
    model.eval()
    metric_names = list(BASE_METRICS)
    if model.cfg.proprio.enabled:
        metric_names.extend(AUX_METRICS)
    if model.cfg.loss.contact_weight > 0:
        metric_names.extend(CONTACT_METRICS)
    total = {name: 0.0 for name in metric_names}
    count = 0
    with torch.no_grad():
        for batch in loader:
            points, actions, proprio, object_pos, cube_robot_contact = _unpack_batch(batch)
            output = _forward(
                model,
                points.to(device),
                actions.to(device),
                env,
                proprio=None if proprio is None else proprio.to(device),
                object_pos=None if object_pos is None else object_pos.to(device),
                cube_robot_contact=None
                if cube_robot_contact is None
                else cube_robot_contact.to(device),
            )
            for key in total:
                total[key] += float(output[key].detach().cpu())
            count += 1
            if max_batches > 0 and count >= max_batches:
                break
    if count == 0:
        raise RuntimeError("validation loader is empty")
    return {key: value / count for key, value in total.items()}


def _save_checkpoint(
    path: Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    model_cfg,
    train_cfg,
    mean,
    std,
    *,
    data_manifest: dict,
    proprio_mean: np.ndarray | None,
    proprio_std: np.ndarray | None,
    initialization: dict | None = None,
) -> None:
    state = {
        "step": step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": OmegaConf.to_container(OmegaConf.structured(model_cfg), resolve=True),
        "train_config": OmegaConf.to_container(OmegaConf.structured(train_cfg), resolve=True),
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "proprio_mean": None if proprio_mean is None else proprio_mean.tolist(),
        "proprio_std": None if proprio_std is None else proprio_std.tolist(),
        "data_manifest": data_manifest,
        "spatial_provenance": model.spatial_provenance(),
        "initialization": initialization,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _prepare_run_dir(train_cfg: Group1TrainConfig, run_dir: Path | None, resume: Path | None) -> Path:
    if run_dir is not None:
        target = run_dir
    else:
        target = Path(train_cfg.run_root) / train_cfg.run_name
    if target.exists() and resume is None:
        unexpected = {
            path.name for path in target.iterdir()
        } - RUNNER_METADATA_FILES
        if unexpected:
            raise FileExistsError(
                "run directory contains training outputs or unknown files; "
                f"pass --resume explicitly: {target}; found={sorted(unexpected)}"
            )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_initial_weights(
    model: WorldModel, checkpoint: Path, device: torch.device
) -> dict:
    """Initialize a contact-enabled model from a compatible legacy checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise RuntimeError(f"initial checkpoint has no model state: {checkpoint}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_missing = (
        ["contact_head.bias", "contact_head.weight"]
        if model.contact_head is not None
        else []
    )
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            "incompatible initial checkpoint: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(checkpoint.resolve()),
        "source_step": int(state.get("step", 0)),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def train(
    model_cfg: WorldModelConfig,
    train_cfg: Group1TrainConfig,
    data_root: Path,
    run_dir: Path,
    device: torch.device,
    resume: Path | None,
    init_checkpoint: Path | None = None,
) -> None:
    root = task_data_root(data_root, train_cfg.task)
    manifest = validate_group1_manifest(
        root,
        expected_task=train_cfg.task,
        expected_episodes=train_cfg.expected_episodes,
    )
    mean, std = action_stats(root, split="train")
    _set_action_stats(model_cfg, mean, std)
    proprio_mean = None
    proprio_std = None
    if train_cfg.include_proprio:
        proprio_mean, proprio_std = load_proprio_stats(root)
        _set_proprio_stats(model_cfg, proprio_mean, proprio_std)
    train_ds = ManiSkillSequenceDataset(
        root, split="train", T=model_cfg.T, point_count=model_cfg.K,
        skip_frame=model_cfg.skip_frame, include_rgb=train_cfg.include_rgb,
        include_proprio=train_cfg.include_proprio,
        contact_sidecar=train_cfg.contact_sidecar,
        contact_alignment_tolerance_m=train_cfg.contact_alignment_tolerance_m,
        phase_repeat=train_cfg.contact_phase_repeat,
    )
    val_ds = ManiSkillSequenceDataset(
        root, split="val", T=model_cfg.T, point_count=model_cfg.K,
        skip_frame=model_cfg.skip_frame, include_rgb=train_cfg.include_rgb,
        include_proprio=train_cfg.include_proprio,
        contact_sidecar=train_cfg.contact_sidecar,
        contact_alignment_tolerance_m=train_cfg.contact_alignment_tolerance_m,
        phase_repeat=1,
    )
    train_batches = CrossEpisodeBatchSampler(
        train_ds, train_cfg.batch_size, shuffle=True, drop_last=True, seed=train_cfg.seed,
    )
    val_batches = CrossEpisodeBatchSampler(
        val_ds, train_cfg.batch_size, shuffle=False, drop_last=False, seed=train_cfg.seed,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_batches, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_sampler=val_batches, num_workers=train_cfg.num_workers)
    model = WorldModel(model_cfg).to(device)
    initialization = None
    if init_checkpoint is not None:
        initialization = _load_initial_weights(model, init_checkpoint, device)
    trainable = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    start_step, best_val = 0, float("inf")
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state.get("best_val", best_val))
    (run_dir / "config.json").write_text(json.dumps({
        "model": OmegaConf.to_container(OmegaConf.structured(model_cfg), resolve=True),
        "train": OmegaConf.to_container(OmegaConf.structured(train_cfg), resolve=True),
        "spatial_provenance": model.spatial_provenance(),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in trainable),
            "frozen": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        },
        "initialization": initialization,
    }, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    iterator = iter(train_loader)
    start_time = time.time()
    step = start_step
    while step < train_cfg.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        points, actions, proprio, object_pos, cube_robot_contact = _unpack_batch(batch)
        model.train()
        output = _forward(
            model,
            points.to(device),
            actions.to(device),
            train_cfg.env,
            proprio=None if proprio is None else proprio.to(device),
            object_pos=None if object_pos is None else object_pos.to(device),
            cube_robot_contact=None
            if cube_robot_contact is None
            else cube_robot_contact.to(device),
        )
        if not torch.isfinite(output["loss"]).item():
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        if not _finite_gradients(model):
            raise RuntimeError(f"non-finite or missing gradients at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        optimizer.step()
        step += 1
        record = {
            "step": step,
            "train_loss": float(output["loss"].detach().cpu()),
            "elapsed_sec": time.time() - start_time,
        }
        for name in (*BASE_METRICS[1:], *AUX_METRICS, *CONTACT_METRICS):
            if name in output:
                record[name] = float(output[name].detach().cpu())
        if step % train_cfg.val_every == 0 or step == train_cfg.steps:
            validation = _evaluate(model, val_loader, device, train_cfg.env, train_cfg.val_batches)
            record.update({f"val_{key}": value for key, value in validation.items()})
            if validation["loss"] < best_val:
                best_val = validation["loss"]
                _save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    best_val,
                    model_cfg,
                    train_cfg,
                    mean,
                    std,
                    data_manifest=manifest,
                    proprio_mean=proprio_mean,
                    proprio_std=proprio_std,
                    initialization=initialization,
                )
        if step % train_cfg.checkpoint_every == 0 or step == train_cfg.steps:
            _save_checkpoint(
                run_dir / f"checkpoint_{step:06d}.pt",
                model, optimizer, step, best_val, model_cfg, train_cfg, mean, std,
                data_manifest=manifest,
                proprio_mean=proprio_mean,
                proprio_std=proprio_std,
                initialization=initialization,
            )
            _save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                step,
                best_val,
                model_cfg,
                train_cfg,
                mean,
                std,
                data_manifest=manifest,
                proprio_mean=proprio_mean,
                proprio_std=proprio_std,
                initialization=initialization,
            )
        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.steps:
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            aux = (
                ""
                if "proprio_loss" not in record
                else f" | proprio {record['proprio_loss']:.4e} | object {record['object_loss']:.4e}"
            )
            print(
                f"step {step:5d} | loss {record['train_loss']:.4f} | "
                f"pred {record['pred_loss']:.4e} | rank {record['rank_loss']:.4e} | "
                f"reg {record['reg_loss']:.4f}{aux}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/maniskill_group1_scratch.yaml"))
    parser.add_argument("--run-dir", type=Path, default=None)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume", type=Path, default=None)
    checkpoints.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--preflight", action="store_true", help="run one forward/backward pass and stop")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    model_cfg, train_cfg = load_group1_config(args.config, args.overrides)
    _seed_everything(train_cfg.seed)
    data_root = Path(train_cfg.data_root)
    root = task_data_root(data_root, train_cfg.task)
    manifest = validate_group1_manifest(
        root,
        expected_task=train_cfg.task,
        expected_episodes=train_cfg.expected_episodes,
    )
    mean, std = action_stats(root, split="train")
    _set_action_stats(model_cfg, mean, std)
    if train_cfg.include_proprio:
        proprio_mean, proprio_std = load_proprio_stats(root)
        _set_proprio_stats(model_cfg, proprio_mean, proprio_std)
    dataset = ManiSkillSequenceDataset(
        root, split="train", T=model_cfg.T, point_count=model_cfg.K,
        skip_frame=model_cfg.skip_frame, include_rgb=train_cfg.include_rgb,
        include_proprio=train_cfg.include_proprio,
        contact_sidecar=train_cfg.contact_sidecar,
        contact_alignment_tolerance_m=train_cfg.contact_alignment_tolerance_m,
        phase_repeat=train_cfg.contact_phase_repeat,
    )
    batches = CrossEpisodeBatchSampler(
        dataset, train_cfg.batch_size, shuffle=False, drop_last=True, seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=train_cfg.num_workers)
    requested = torch.device(train_cfg.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    print(f"task={manifest['task']} train_episodes={len(manifest['train_episode_ids'])} val_episodes={len(manifest['val_episode_ids'])} windows={len(dataset)} device={device}")
    model = WorldModel(model_cfg).to(device)
    if args.init_checkpoint is not None:
        _load_initial_weights(model, args.init_checkpoint, device)
    preflight = run_preflight(model, loader, device, train_cfg.env)
    print(json.dumps({"preflight": preflight}, sort_keys=True))
    if args.preflight:
        return
    run_dir = _prepare_run_dir(train_cfg, args.run_dir, args.resume)
    train(
        model_cfg,
        train_cfg,
        data_root,
        run_dir,
        device,
        args.resume,
        args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
