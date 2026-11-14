"""train stage of the Cube workflow: each method's own trainer on the training views of the collected data (JEPA4R /
LeWM: LeWM's trainer with the recipes of configs/method/cube/recipes; PLDM: jepa4r/baselines/cube/pldm/train.py;
LeJEPA: jepa4r/baselines/cube/native/tiny_train.py), 10 epochs; the last epoch's export is recorded in WM_EXPORT.json.
usage: python -m jepa4r.models.cube.fresh_train --method <m> --data-manifest <file> --output <dir> [--execute]"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys

from jepa4r.data.cube import fresh_data
from .training_runtime import build_environment, wm_trainer, load_recipe, load_tiny_trainer

METHODS = ("pldm", "lejepa", "lewm", "jepa4r")
EPOCHS = 10


def describe_training(method, data_manifest, output_root, *, workers=None, prefetch=None):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (one of {METHODS})")
    binding = fresh_data.load_binding(data_manifest)
    cubes = binding["cube_count"]
    result = dict(
        method=method,
        cube_count=cubes,
        data_manifest=str(Path(data_manifest).resolve()),
        output=str(Path(output_root).resolve()),
        workers=workers,
        prefetch=prefetch,
        epochs=EPOCHS,
    )
    if method in ("lewm", "jepa4r"):
        modality = "rgb" if method == "lewm" else "pcd"
        cfg = load_recipe(method)
        label = "fresh_cube" + str(cubes) + "_" + method
        cfg["output_model_name"] = cfg["subdir"] = label
        for role, split in [("dataset", "train"), ("validation_dataset", "validation")]:
            cfg["data"][role] = dict(
                name=binding["views"][split + "_" + modality]["path"],
                num_steps=4,
                frameskip=5,
                keys_to_load=["pixels", "action"] if modality == "rgb" else ["pcd_xyz", "pcd_rgb", "action"],
                keys_to_cache=["action"],
            )
        if workers is not None:
            cfg["loader"]["num_workers"] = cfg["num_workers"] = cfg["validation_num_workers"] = workers
        if cfg["loader"]["num_workers"] == 0:
            cfg["loader"].pop("prefetch_factor", None)
            cfg["loader"].pop("persistent_workers", None)
        elif prefetch is not None:
            cfg["loader"]["prefetch_factor"] = prefetch
        result.update(config=cfg, epochs=cfg["trainer"]["max_epochs"])
    return result


@contextmanager
def _argv(values):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


@contextmanager
def route_loader(native, plan):
    """Open the training views through fresh_data's reader when the trainer asks for them by path."""
    binding = fresh_data.load_binding(plan["data_manifest"])
    original = native.swm.data.load_dataset
    modality = "pcd" if plan["method"] == "jepa4r" else "rgb"
    paths = {Path(row["path"]).resolve(): key for key, row in binding["views"].items() if key.endswith("_" + modality)}

    def load(name, **kwargs):
        kwargs.pop("format", None)
        return fresh_data.make_reader(plan["data_manifest"], paths[Path(name).resolve()], **kwargs)

    native.swm.data.load_dataset = load
    try:
        yield
    finally:
        native.swm.data.load_dataset = original


def _pldm(plan, output):
    from jepa4r.baselines.cube.pldm import train

    native_output = output / "native"
    arguments = ["pldm", "--cube-count", str(plan["cube_count"]), "--data-manifest", plan["data_manifest"]]
    arguments += ["--output", str(native_output)]
    arguments += ["--workers", str(plan["workers"] if plan["workers"] is not None else 20)]
    arguments += ["--prefetch", str(plan["prefetch"] if plan["prefetch"] is not None else 2)]
    with _argv(arguments):
        train.main()
    directory = native_output / "checkpoints" / f"cube{plan['cube_count']}_pldm_seed0"
    return directory / f"weights_epoch_{plan['epochs']}.pt", directory / "config.json"


def _tiny(plan, output):
    module = load_tiny_trainer()
    binding = fresh_data.load_binding(plan["data_manifest"])
    label = "lejepa_fresh_cube" + str(plan["cube_count"])
    values = [module.__file__, "--task", label, "--dataset", binding["views"]["train_rgb"]["path"], "--name", label]
    values += ["--epochs", str(plan["epochs"])]
    if plan["workers"] is not None:
        values.extend(["--workers", str(plan["workers"])])
    original_loader = module.DataLoader
    if plan["prefetch"] is not None:

        def loader(*args, **kwargs):
            kwargs["prefetch_factor"] = plan["prefetch"]
            return original_loader(*args, **kwargs)

        module.DataLoader = loader
    try:
        with route_loader(module, plan), _argv(values):
            module.main()
    finally:
        module.DataLoader = original_loader
    directory = output / "native" / "checkpoints" / label
    return directory / f"weights_epoch_{plan['epochs']}.pt", directory / "config.json"


def _lewm(plan, output):
    from omegaconf import OmegaConf

    cfg = plan["config"]
    with wm_trainer() as native:
        with route_loader(native, plan):
            native.run(OmegaConf.create(cfg))
    directory = output / "native" / "checkpoints" / cfg["output_model_name"]
    return directory / f"weights_epoch_{plan['epochs']}.pt", directory / "config.json"


def run_job(plan):
    """Train the world model and write WM_EXPORT.json (the last epoch's weights and their config)."""
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    previous_env = dict(os.environ)
    with (output / "stdout_stderr.log").open("w", encoding="utf8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                os.environ.update(build_environment(Path(plan["data_manifest"]).parent, output / "native"))
                runner = _pldm if plan["method"] == "pldm" else _tiny if plan["method"] == "lejepa" else _lewm
                checkpoint, config = runner(plan, output)
            finally:
                os.environ.clear()
                os.environ.update(previous_env)
    record = dict(
        method=plan["method"],
        cube_count=plan["cube_count"],
        epochs=plan["epochs"],
        checkpoint=str(checkpoint.resolve()),
        config=str(config.resolve()),
        data_manifest=plan["data_manifest"],
    )
    (output / "WM_EXPORT.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = describe_training(args.method, args.data_manifest, args.output, workers=args.workers, prefetch=args.prefetch)
    print(json.dumps(plan, indent=2))
    return run_job(plan) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
