"""World-model training and behavior cloning of the Adroit Relocate workflow.

train  the world model on the collected Relocate data: JEPA4R / LeWM with LeWM's trainer and the Relocate configs of
       configs/method/relocate/train (JEPA4R: point clouds with the DexArt PointNet), PLDM with stable-worldmodel's
       script (jepa4r/baselines/relocate/train_pldm.py), LeJEPA Tiny (jepa4r/baselines/relocate/train_lejepa.py);
       10 epochs, exports under <output>/checkpoints/relocate
bc     CortexBench's frozen-encoder BC loop (eai-vc, run_bc.py / run_adroit_bc3d.py for point clouds) with the run's
       encoder as the embedding; the Hydra overrides after "--" (seed, demos, epochs, evaluation) as the workflow
       gives them
usage: python -m jepa4r.policy.bc.relocate.adroit_learning train|bc --model <m> --output <dir> ..."""

import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]


def configure_runtime():
    extra = os.environ.get("ADROIT_WM_EXTRA_SITE")
    if extra:
        sys.path.append(extra)
    paths = [ROOT.parent, ROOT / "policy/bc"]
    eai = os.environ.get("CORTEXBENCH_EAI_VC_ROOT")
    if eai:
        paths.extend(
            [Path(eai) / "cortexbench", Path(eai) / "vc_models", Path(eai) / "cortexbench/mujoco_vc/visual_imitation"]
        )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def wm_overrides(model):
    """The Hydra overrides of the Relocate train config (configs/method/relocate/train/lewm.yaml) for JEPA4R / LeWM."""
    return [
        "data=" + ("adroit_relocate_pcd" if model == "jepa4r" else "adroit_relocate"),
        "model=" + model,
        "subdir=relocate",
        "output_model_name=relocate",
        "wandb.enabled=false",
        "num_workers=16",
        "loader.persistent_workers=true",
        "loader.prefetch_factor=2",
    ]


def train(args):
    import stable_pretraining as spt
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    os.environ.update(STABLEWM_HOME=str(args.output), SPT_CACHE_DIR=str(args.output / "_cache"))
    arguments = ["--task", "adroit_relocate", "--dataset", str(args.data_root / "train_rgb.h5")]
    arguments += ["--output", str(args.output), "--name", "relocate", "--workers", "16"]
    if args.model in ("jepa4r", "lewm"):
        import hydra

        native = importlib.import_module("jepa4r.models.world_model")
        overrides = wm_overrides(args.model)
        os.environ["RELOCATE_" + ("PCD" if args.model == "jepa4r" else "RGB") + "_TRAIN"] = str(
            args.data_root / ("train_" + ("pcd" if args.model == "jepa4r" else "rgb") + ".h5")
        )
        with hydra.initialize_config_dir(
            version_base=None, config_dir=str(ROOT.parent / "configs/method/relocate/train")
        ):
            cfg = hydra.compose(config_name="lewm", overrides=overrides)
        spt.set(
            cache_dir=str(args.output / "_cache"),
            cleanup={"checkpoints": False, "logs": False, "callback_artifacts": False},
        )
        native.run_random_split(cfg, None, manager_seed=int(cfg.seed), original_layout=True)
    elif args.model == "pldm":
        native = importlib.import_module("jepa4r.baselines.relocate.train_pldm")
        parser = native.make_parser()
        native.execute(
            native.build_plan(
                parser.parse_args(arguments + ["--framework-root", os.environ["ADROIT_FRAMEWORK_ROOT"]]), parser
            )
        )
    else:
        native = importlib.import_module("jepa4r.baselines.relocate.train_lejepa")
        parser = native.make_parser()
        native.execute(native.build_plan(parser.parse_args(arguments), parser))


def bc(args):
    import wm_embeddings
    import jepa_baselines

    selector = {
        "lewm": "lewm_adroit_relocate",
        "jepa4r": "jepa4r_adroit_relocate",
        "pldm": "pldm_relocate",
        "lejepa": "lejepa_relocate",
        "visreg": "visreg_vitb",
        "vjepa": "vjepa21_vitb",
    }[args.model]
    os.environ["STABLEWM_HOME"] = str(args.output / "assets")
    for key in (
        "RELOCATE_RGB_CHECKPOINT",
        "RELOCATE_PCD_CHECKPOINT",
        "RELOCATE_PLDM_CHECKPOINT",
        "RELOCATE_LEJEPA_CHECKPOINT",
    ):
        os.environ[key] = str(args.checkpoint)
    if os.environ.get("ADROIT_VJEPA_ROOT"):
        jepa_baselines.VJEPA2_REPO = Path(os.environ["ADROIT_VJEPA_ROOT"])
    original_load = {}
    for module, name in (
        (wm_embeddings, "load_lewm"),
        (wm_embeddings, "load_jepa4r"),
        (jepa_baselines, "load_pldm"),
        (jepa_baselines, "load_lejepa"),
        (jepa_baselines, "load_visreg"),
        (jepa_baselines, "load_vjepa21"),
    ):
        original = getattr(module, name)
        original_load[(module, name)] = original

        def load(*a, _original=original, _name=name, **kw):  # the run's checkpoint / epoch; the encoder stays frozen
            if _name in ("load_visreg", "load_vjepa21"):
                kw["checkpoint"] = str(args.checkpoint)
            else:
                kw["epoch"] = args.epoch
            result = _original(*a, **kw)
            for p in result[0].parameters():
                p.requires_grad_(False)
            return result

        setattr(module, name, load)
    tail = args.entry_args[1:] if args.entry_args[:1] == ["--"] else args.entry_args
    sys.argv = [
        str(ROOT / "policy/bc" / ("run_adroit_bc3d.py" if args.model == "jepa4r" else "run_bc.py")),
        *tail,
        "embedding=" + selector,
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        for (module, name), original in original_load.items():
            setattr(module, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("train", "bc"))
    parser.add_argument("--model", required=True, choices=("jepa4r", "lewm", "pldm", "lejepa", "visreg", "vjepa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epoch", type=int, default=10)
    args, tail = parser.parse_known_args()
    args.entry_args = tail
    configure_runtime()
    {"train": train, "bc": bc}[args.stage](args)


if __name__ == "__main__":
    main()
