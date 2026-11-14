"""LeJEPA on Adroit Relocate: LeJEPA's Tiny trainer (train_lejepa_relocate.py) on the Relocate RGB frames -- ViT-Tiny/14
at 224, four augmented views, a 512-d downstream head and a 16-d training projector, batch 256, 10 epochs, AdamW
(lr 2e-3, weight decay 0.05), lambda 0.02, BF16, seed 3072; checkpoints under --output (STABLEWM_HOME).
usage: python -m jepa4r.baselines.relocate.train_lejepa --dataset <train_rgb.h5> --output <dir>"""

import argparse
import importlib
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
NATIVE_SCRIPT = PACKAGE_ROOT / "jepa4r/baselines/relocate/train_lejepa_relocate.py"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="the run's STABLEWM_HOME")
    parser.add_argument("--task", choices=("adroit_relocate",), default="adroit_relocate")
    parser.add_argument("--name", default=None, help="checkpoint name; default lejepa_TASK")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, choices=(1, 2, 3), default=2)
    return parser


def build_plan(args, parser):
    if args.name is None:
        args.name = "lejepa_" + args.task
    dataset, output = args.dataset.expanduser().resolve(), args.output.expanduser().resolve()
    native = ["--task", args.task, "--dataset", str(dataset), "--name", args.name, "--epochs", "10", "--bs", "256"]
    native += ["--V", "4", "--lr", "0.002", "--lamb", "0.02", "--proj-dim", "16", "--workers", str(args.workers)]
    native += ["--seed", "3072"]
    return dict(
        dataset=str(dataset),
        output=str(output),
        native_argv=native,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
    )


def loader_factory(original, workers, prefetch, close_parent):
    def create(*args, **kwargs):
        dataset = args[0] if args else kwargs.get("dataset")
        close_parent(getattr(dataset, "ds", dataset))
        kwargs = dict(kwargs, num_workers=workers, persistent_workers=workers > 0)
        if workers:
            kwargs["prefetch_factor"] = prefetch
        else:
            kwargs.pop("prefetch_factor", None)
        return original(*args, **kwargs)

    return create


def execute(plan):
    dataset, output = Path(plan["dataset"]), Path(plan["output"])
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    native = importlib.import_module("jepa4r.baselines.relocate.train_lejepa_relocate")
    helpers = importlib.import_module("jepa4r.baselines.relocate.data")
    output.mkdir(parents=True, exist_ok=True)
    old_argv, old_home, old_loader = sys.argv, os.environ.get("STABLEWM_HOME"), native.DataLoader
    try:
        sys.argv = [str(NATIVE_SCRIPT), *plan["native_argv"]]
        os.environ["STABLEWM_HOME"] = str(output)
        native.DataLoader = loader_factory(
            old_loader, plan["workers"], plan["prefetch_factor"], helpers.close_parent_hdf
        )
        with helpers.dataset_handle_hook(native.swm, dataset):
            native.main()
    finally:
        sys.argv, native.DataLoader = old_argv, old_loader
        if old_home is None:
            os.environ.pop("STABLEWM_HOME", None)
        else:
            os.environ["STABLEWM_HOME"] = old_home


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    execute(build_plan(args, parser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
