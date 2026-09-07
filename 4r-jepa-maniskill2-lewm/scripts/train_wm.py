#!/usr/bin/env python
"""LeWM's training driver with a model replacement and point-cloud column.

Production defaults are read directly from le-wm/config, not tuned here.
--smoke is an explicitly separate CPU integration test, not a reproduction
of the full published training run.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT / "vendor/le-wm", ROOT / "vendor/stable-worldmodel-lewm-compatible",
             ROOT / "vendor/ManiSkill2-Learn", ROOT / "vendor/pytorch3d"]:
    sys.path.insert(0, str(path))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "artifacts/worldmodel"))
os.environ.setdefault("WANDB_MODE", "disabled")

from omegaconf import OmegaConf
from ms2_lewm.official import LEWM, lewm_config, load_source
from ms2_lewm.temporal import resolve_temporal_config


def model_forward(self, batch, stage, cfg):
    # This replaces only LeWM's model/objective forward callback.
    output = self.model(batch)
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
                  on_step=True, sync_dist=True)
    return output


def make_config(dataset, run_name, smoke=False):
    cfg = lewm_config("ogb")
    temporal_config = resolve_temporal_config()
    cfg.history_size = temporal_config["history_size"]
    cfg.output_model_name = run_name
    cfg.subdir = run_name
    cfg.data.dataset.name = str(Path(dataset).resolve())
    # Keep RGB for the official storage/evaluation infrastructure. The model
    # consumes the extra pointcloud column without ImageNet normalization.
    cfg.data.dataset.keys_to_load = ["pixels", "pointcloud", "action", "observation"]
    cfg.data.dataset.keys_to_cache = ["action", "observation"]
    # Explicit user correction: the added temporal encoder consumes consecutive
    # frames in both pretraining and frozen-encoder PPO. The original LeWM
    # dataset's frameskip=5 is not an encoder hyperparameter.
    cfg.data.dataset.frameskip = 1
    # Persist the chosen encoder config with every export; activation functions
    # have no state-dict parameters, so weights alone cannot identify the activation.
    cfg.model = {"_target_": "ms2_lewm.model.WorldModel", "action_dim": 7, "history_stride": 1,
                 "rl_config_path": "configs/ppo_pointnet_identity.py", "temporal_config": temporal_config}
    if smoke:
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        cfg.trainer.max_epochs = 2
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 1
        cfg.num_workers = 0
        cfg.loader.batch_size = 4
        cfg.loader.num_workers = 0
        cfg.loader.persistent_workers = False
        cfg.loader.prefetch_factor = None
        cfg.loader.pin_memory = False
    return cfg


def run(args):
    if not args.smoke:
        dataset_path = Path(args.dataset)
        if dataset_path.is_file() and dataset_path.suffix.lower() in (".h5", ".hdf5"):
            import h5py
            with h5py.File(dataset_path, "r") as handle:
                if bool(handle.attrs.get("SYNTHETIC_UNIT_TEST_ONLY", False)):
                    raise ValueError("Synthetic unit-test fixtures cannot be used for production training.")
        # Resource admission only: no trainer/model/optimizer setting is changed.
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
        ], text=True, timeout=10)
        devices = [{"index": int(row.split(",")[0]), "free_mib": int(row.split(",")[1])}
                   for row in raw.strip().splitlines()]
        admission = {"devices": devices, "required_free_mib": 16384,
                     "passed": bool(devices) and all(d["free_mib"] >= 16384 for d in devices)}
        (ROOT / "artifacts/last_training_resource_check.json").write_text(json.dumps(admission, indent=2))
        if not admission["passed"]:
            raise RuntimeError("GPU training deferred to protect existing tasks: need 16384 MiB free.")
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from ms2_lewm.model import WorldModel

    cfg = make_config(args.dataset, args.run_name, args.smoke)
    # The functions below are imported from pristine public le-wm/utils.py.
    official_utils = load_source("_ms2_official_lewm_utils", LEWM / "utils.py")
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None,
                                   cache_dir=os.environ.get("LOCAL_DATASET_DIR"), **dataset_cfg)
    transforms = [official_utils.get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels") or col == "pointcloud":
            continue
        transforms.append(official_utils.get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    cfg.model.action_dim = int(cfg.data.dataset.frameskip * dataset.get_dim("action"))

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    if not len(train) or not len(val):
        raise ValueError("Dataset is too small for the official split/batch size; collect more trajectories.")

    world_model = WorldModel(action_dim=cfg.model.action_dim, rl_config_path=cfg.model.rl_config_path,
                            history_stride=cfg.model.history_stride, temporal_config=cfg.model.temporal_config)
    initial_parameters = ({name: p.detach().cpu().clone() for name, p in world_model.named_parameters()}
                          if args.smoke else None)
    # The matching official stable_worldmodel/scripts/train/lewm.py supplies
    # these arguments; public le-wm/train.py omits the required scheduler args.
    # Preserve that official implementation, including its epoch interval.
    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {"model_opt": {
        "modules": "model", "optimizer": dict(cfg.optimizer),
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                      "warmup_steps": max(1, int(0.01 * total_steps)), "max_steps": total_steps},
        "interval": "epoch",
    }}
    data_module = spt.data.DataModule(train=train, val=val)
    training_module = spt.Module(model=world_model,
                                forward=partial(model_forward, cfg=cfg), optim=optimizers)
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    callback = official_utils.SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)
    trainer = pl.Trainer(**cfg.trainer, callbacks=[callback], num_sanity_val_steps=1,
                         logger=logger, enable_checkpointing=True)
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(trainer=trainer, module=training_module, data=data_module,
                          ckpt_path=args.resume or (ckpt_path if ckpt_path.exists() else None))
    manager()
    # A resumable optimizer/scheduler checkpoint in addition to official model export.
    trainer.save_checkpoint(ckpt_path)
    summary = {
        "mode": "cpu_integration_smoke" if args.smoke else "official_hyperparameters",
        "global_step": trainer.global_step, "train_windows": len(train_set),
        "val_windows": len(val_set), "action_dim": cfg.model.action_dim,
        "latent_dim": world_model.latent_dim, "history_size": world_model.history_size,
        "history_stride": world_model.history_stride, "data_frameskip": int(cfg.data.dataset.frameskip),
        "checkpoint": str(ckpt_path),
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        "finite_parameters": all(bool(torch.isfinite(p).all()) for p in world_model.parameters()),
    }
    if initial_parameters is not None:
        groups = {"frame_encoder": "encoder.frame_encoder.",
                  "temporal_encoder": "encoder.temporal_encoder.", "predictor": "predictor."}
        summary["parameter_l1_change"] = {
            group: sum(float((p.detach().cpu() - initial_parameters[name]).abs().sum())
                       for name, p in world_model.named_parameters() if name.startswith(prefix))
            for group, prefix in groups.items()
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", default="pointcloud_lewm_identity")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if args.config_only:
        print(OmegaConf.to_yaml(make_config(args.dataset, args.run_name, args.smoke)))
    else:
        run(args)
