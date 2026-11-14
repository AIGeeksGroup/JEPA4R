"""PLDM training on Cube (the train stage of the Cube workflow): the PLDM authors' model, objective, optimizer and
learning-rate schedule (third_party/pldm, method.py) in stable-pretraining's trainer, 10 epochs of batch 128 in BF16,
on the training views of data_manifest.json; LeWM's SaveCkptCallback exports weights_epoch_<n>.pt every epoch.
usage: python -m jepa4r.baselines.cube.pldm.train --cube-count <n> --data-manifest <file> --output <dir>"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import os
from pathlib import Path

from .shared_data import contract, loaders
from jepa4r.models.cube.training_runtime import build_environment

VENDOR = Path(__file__).resolve().parents[4] / "third_party/pldm"  # the PLDM authors' source


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + "\n")


def method_module(method):
    return importlib.import_module(".method", __package__)


def factory_config(args):
    return dict(_target_=__package__ + ".method.build_model", official_source=str(VENDOR), pretrained=False)


def native_module(method, module, model, profile_name, data_binding=None):
    """A stable-pretraining Module whose optimizer is built here; its training_step stays the method's own."""
    import stable_pretraining as spt

    class MethodModule(spt.Module):
        def configure_optimizers(self):
            if method == "pldm":
                opt = module.build_optimizer(self.model)
                self.native_author_schedule = module.build_scheduler(
                    opt, epochs=10, batch_steps=contract(profile_name, data_binding)["train_batches"], batch_size=128
                )
                return opt
            opt, scheduler = module.make_optimizer_and_scheduler(self.model.parameters())
            return [opt], [dict(scheduler=scheduler, interval="step")]

    def pldm_forward(self, batch, stage):
        out = self.model(batch["pixels"], batch["actions"])
        for key, value in out.items():
            if "loss" in key:
                self.log(stage + "/" + key, value, on_step=True, on_epoch=True, sync_dist=True)
        return out

    return MethodModule(model=model, forward=pldm_forward if method == "pldm" else module.forward, optim=None)


def author_schedule_before_step(pl_module, step):
    pl_module.native_author_schedule.adjust_learning_rate(int(step))
    return int(step)


def trainer_options(callbacks):
    return dict(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=True,
    )


def run(args):
    import lightning as pl
    import stable_pretraining as spt
    from omegaconf import OmegaConf

    if spt.__version__ != "0.1.8":
        raise ValueError("stable-pretraining 0.1.8 is required")
    pl.seed_everything(0, workers=True)
    module = method_module(args.method)
    train, val, stats = loaders(
        args.profile, args.data_root, module, args.workers, args.prefetch, data_binding=args.data_binding
    )
    from jepa4r.models import world_model as exports  # LeWM's utils.py SaveCkptCallback is in world_model.py

    cfg = factory_config(args)
    model = module.build_model(cfg["official_source"], pretrained=False)
    world = native_module(args.method, module, model, args.profile, args.data_binding)
    label = args.profile + "_" + args.method + "_seed0"
    export_root = args.output / "checkpoints" / label
    export_root.mkdir(parents=True, exist_ok=False)
    write(args.output / "action_statistics.json", stats)

    class AuthorSchedule(pl.pytorch.callbacks.Callback):
        """PLDM's learning-rate schedule, applied before every optimizer step."""

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if args.method == "pldm":
                author_schedule_before_step(pl_module, trainer.global_step)

    callbacks = [exports.SaveCkptCallback(label, OmegaConf.create(cfg), epoch_interval=1), AuthorSchedule()]
    trainer = pl.Trainer(**trainer_options(callbacks))
    data = spt.data.DataModule(train=train, val=val)
    spt.Manager(trainer=trainer, module=world, data=data, seed=0)()
    return export_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-count", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--prefetch", type=int, default=2)
    args = parser.parse_args()
    args.profile = "cube" + str(args.cube_count)
    args.data_binding = args.data_manifest.resolve()
    args.data_root = args.data_binding.parent
    args.method = "pldm"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(build_environment(args.data_root, args.output))
    with (args.output / "stdout.log").open("x", encoding="utf8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            run(args)
    return 0
