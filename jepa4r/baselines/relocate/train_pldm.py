"""PLDM on Adroit Relocate: stable-worldmodel's PLDM training script (scripts/train/pldm.py of the framework checkout,
JEPA4R_SWM) with its pldm.yaml on the Relocate RGB data: 10 epochs, seed 3072, outputs under --output (STABLEWM_HOME).
usage: python -m jepa4r.baselines.relocate.train_pldm --framework-root <swm> --dataset <train_rgb.h5> --output <dir>"""

import argparse
import ast
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import types

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework-root", required=True, type=Path, help="the stable-worldmodel checkout")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="the run's STABLEWM_HOME")
    parser.add_argument("--task", choices=("adroit_relocate",), default="adroit_relocate")
    parser.add_argument("--name", default=None, help="checkpoint name; default pldm_TASK")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, choices=(1, 2, 3), default=2)
    return parser


def build_plan(args, parser):
    if args.name is None:
        args.name = "pldm_" + args.task
    framework = args.framework_root.expanduser().resolve()
    dataset, output = args.dataset.expanduser().resolve(), args.output.expanduser().resolve()
    overrides = [
        "data=ogb",
        "seed=3072",
        f"data.dataset.name={json.dumps(str(dataset))}",
        "data.dataset.keys_to_load=[pixels,action]",
        "data.dataset.keys_to_cache=[action]",
        "~data.dataset.keys_to_merge",
        f"subdir={json.dumps(args.name)}",
        f"output_model_name={json.dumps(args.name)}",
        "trainer.max_epochs=10",
        f"num_workers={args.workers}",
        "loader.persistent_workers=" + str(args.workers > 0).lower(),
    ]
    overrides += [f"loader.prefetch_factor={args.prefetch_factor}"] if args.workers else ["~loader.prefetch_factor"]
    return dict(
        framework_root=str(framework),
        native_script=str(framework / "scripts/train/pldm.py"),
        dataset=str(dataset),
        output=str(output),
        name=args.name,
        hydra_overrides=overrides,
    )


def logger_compatible_tree(source, filename):
    """The training script's AST with its one ``logger.info`` call in ``run`` made module-independent (a logger-scope
    fix); the file on disk is unchanged."""
    tree = ast.parse(source, filename=filename)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if function is None:
        raise ValueError("framework entry has no original run function")
    calls = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "info"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "logger"
    ]
    if len(calls) > 1:
        raise ValueError("logger scope compatibility must select at most one original call")
    if calls:
        calls[0].func = ast.parse("logging.getLogger(__name__).info", mode="eval").body
    return ast.fix_missing_locations(tree)


def load_native(path):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = logger_compatible_tree(source, str(path))
    name = "_relocate_framework_pldm"
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), ""
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not Path(module.swm.__file__).resolve().is_relative_to(path.parents[2].resolve()):
        raise RuntimeError("stable_worldmodel was imported from another checkout than the framework root")
    return module


@contextmanager
def explicit_training_seed(native, seed):
    import lightning as pl

    seed = int(seed)
    pl.seed_everything(seed, workers=True)
    original = native.spt.Manager

    def manager(*args, **kwargs):
        kwargs["seed"] = seed
        return original(*args, **kwargs)

    native.spt.Manager = manager
    try:
        yield
    finally:
        native.spt.Manager = original


def execute(plan):
    framework, dataset, output = map(Path, (plan["framework_root"], plan["dataset"], plan["output"]))
    previous_path = list(sys.path)
    previous_env = {key: os.environ.get(key) for key in ("STABLEWM_HOME", "SPT_CACHE_DIR")}
    sys.path[:0] = [str(PACKAGE_ROOT), str(framework)]
    try:
        native = load_native(plan["native_script"])
        hydra = importlib.import_module("hydra")
        helpers = importlib.import_module("jepa4r.baselines.relocate.data")
        with hydra.initialize_config_dir(version_base=None, config_dir=str(framework / "scripts/train/config")):
            cfg = hydra.compose(config_name="pldm", overrides=plan["hydra_overrides"])
        output.mkdir(parents=True, exist_ok=True)
        os.environ.update(STABLEWM_HOME=str(output), SPT_CACHE_DIR=str(output / "_spt_cache"))
        native.spt.set(
            cache_dir=str(output / "_spt_cache"),
            cleanup={"checkpoints": False, "logs": False, "callback_artifacts": False},
        )
        with helpers.dataset_handle_hook(native.swm, dataset), explicit_training_seed(native, int(cfg.seed)):
            native.run.__wrapped__(cfg)
    finally:
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    execute(build_plan(args, parser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
