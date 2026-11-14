"""World-model training on the RH56 twin's data with the world-model training entry point
(jepa4r/models/world_model.py, used unchanged) and the Relocate train configs (configs/method/relocate/train: point
clouds with the DexArt PointNet for route pcd, the RGB frames for route rgb).

The split follows the episodes' own train / val labels (manifest.json of the published data), the action normalizer is
fitted on the train rows only; batch 128, 10 epochs, seed 3072 (Python, NumPy, Torch and the training manager), BF16 on
CUDA. The checkpoints go to <output>/worldmodel_cache/checkpoints/rh56_relocate_<route>/weights_epoch_<n>.pt.
usage: python -m jepa4r.envs.twins.rh56.wm_bridge --manifest <file> --dataset <h5> --route rgb|pcd --output <dir>"""

from contextlib import contextmanager
from pathlib import Path
import argparse
import importlib
import json
import os
import random
import sys

import numpy as np

from jepa4r.envs.twins.rh56._runtime import temporary_bindings

BATCH_SIZE = 128
EPOCHS = 10


class EpisodePartition:
    """The train / val episodes of the published data and their clips and rows."""

    def __init__(self, dataset, manifest):
        records = sorted(manifest["episodes"], key=lambda r: r["episode_index"])
        self.episode_ids = {s: [r["episode_index"] for r in records if r["split"] == s] for s in ("train", "val")}
        episode_sets = {s: set(ids) for s, ids in self.episode_ids.items()}
        self.clip_indices = {
            s: [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in episode_sets[s]] for s in self.episode_ids
        }
        self.rows = {
            s: [(int(dataset.offsets[ep]), int(dataset.offsets[ep] + dataset.lengths[ep])) for ep in ids]
            for s, ids in self.episode_ids.items()
        }


class TrainColumnView:
    """Feed only the train rows to the trainer's column normalizer (get_column_normalizer, unchanged)."""

    def __init__(self, dataset, partition):
        self.dataset, self.partition = dataset, partition
        self._cache = {}

    def get_col_data(self, col):
        if col not in self._cache:
            data = np.asarray(self.dataset.get_col_data(col))
            self._cache[col] = np.concatenate([data[lo:hi] for lo, hi in self.partition.rows["train"]])
        return self._cache[col]


@contextmanager
def install_episode_split(train_module, dataset_path, manifest):
    """Replace the trainer's random split and normalizer fit for this dataset by the episode split (both restored
    after)."""
    expected = Path(dataset_path).resolve()
    normalizer = train_module.get_column_normalizer
    splitter = train_module.spt.data.random_split
    bound = {}

    def selected(dataset):
        path = getattr(dataset, "h5_path", None)
        return path is not None and Path(path).resolve() == expected

    def partition(dataset):
        if "partition" not in bound:
            bound["partition"] = EpisodePartition(dataset, manifest)
        return bound["partition"]

    def fit(dataset, source, target):
        if not selected(dataset):
            return normalizer(dataset, source, target)
        return normalizer(TrainColumnView(dataset, partition(dataset)), source, target)

    def split(dataset, lengths, generator=None):
        if not selected(dataset):
            return splitter(dataset, lengths, generator=generator)
        episodes = partition(dataset)
        # The parent process opened the HDF5 file to read the episodes; each data-loader worker opens its own handle.
        parent_handle = getattr(dataset, "h5_file", None)
        if parent_handle is not None:
            parent_handle.close()
            dataset.h5_file = None
        from torch.utils.data import Subset

        return [Subset(dataset, episodes.clip_indices[s]) for s in ("train", "val")]

    with temporary_bindings(
        (train_module, "get_column_normalizer", fit), (train_module.spt.data, "random_split", split)
    ):
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--package-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="the jepa4r package root; defaults to this checkout",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--route", required=True, choices=("rgb", "pcd"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args(argv)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import stable_pretraining as spt
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    package = Path(args.package_root).resolve()
    sys.path.insert(0, str(package.parent))
    train = importlib.import_module("jepa4r.models.world_model")
    train.spt = spt  # the training code in world_model.py imports it lazily
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    overrides = (
        ["data=adroit_relocate", "model=lewm"] if args.route == "rgb" else ["data=adroit_relocate_pcd", "model=jepa4r"]
    )
    with initialize_config_dir(version_base=None, config_dir=str(package.parent / "configs/method/relocate/train")):
        cfg = compose(config_name="jepa_wm", overrides=overrides)
    run_name = f"rh56_relocate_{args.route}"
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.data.dataset.name = str(Path(args.dataset).resolve())
        cfg.data.dataset.keys_to_load = (
            ["pixels", "action"] if args.route == "rgb" else ["pcd_xyz", "pcd_rgb", "action"]
        )
        cfg.data.dataset.keys_to_cache = ["action"]
        cfg.output_model_name = run_name
        cfg.subdir = run_name
        cfg.num_workers = args.workers
        cfg.loader.batch_size = BATCH_SIZE
        cfg.loader.num_workers = args.workers
        cfg.loader.persistent_workers = args.workers > 0
        cfg.loader.pin_memory = args.device == "cuda"
        if args.workers:
            cfg.loader.prefetch_factor = args.prefetch_factor
        else:
            cfg.loader.pop("prefetch_factor", None)
        cfg.wandb = {"enabled": False}
        cfg.trainer.max_epochs = EPOCHS
        cfg.trainer.pop("limit_train_batches", None)
        cfg.trainer.pop("limit_val_batches", None)
        cfg.trainer.accelerator = "gpu" if args.device == "cuda" else "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.precision = "bf16-mixed" if args.device == "cuda" else "32-true"
        cfg.trainer.enable_progress_bar = False
    os.environ["STABLEWM_HOME"] = str(out / "worldmodel_cache")
    spt.set(
        cache_dir=str(out / "spt_cache"), cleanup={"checkpoints": False, "logs": False, "callback_artifacts": False}
    )
    original_manager = train.spt.Manager

    def manager(*manager_args, **manager_kwargs):
        manager_kwargs["seed"] = int(args.seed)
        return original_manager(*manager_args, **manager_kwargs)

    with temporary_bindings((train.spt, "Manager", manager)), install_episode_split(train, args.dataset, manifest):
        train.run_random_split(cfg, None, original_layout=True)
    print(json.dumps(dict(route=args.route, checkpoints=str(out / "worldmodel_cache/checkpoints" / run_name))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
