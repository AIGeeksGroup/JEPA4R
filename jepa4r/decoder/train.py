"""Decoder training (stage train_decoder), linear-probe-then-fine-tune order [kumar2022lpft]:

stage A  train the new parameters (set generator, affine map, rgb branch; 5.75M), backbone frozen
stage B  add rank-4 LoRA on q / v of all attention blocks and continue
Targets: observed clouds of data/wm/<env>; inputs: latents of the frozen world model cfg.decoder.wm (epoch 10).
Returns {"val_nn_stage_a": cm, "val_nn_stage_b": cm} (paper: 0.87 -> 0.79).

From the paper's reconstruction code: the data (PairedFrames, build_dataset, Episodes: one frame for the
encoder + the point cloud of the same simulator state) and the training run (``run_stage``: one stage = one
Lightning run of 14,140 batches of 32 frames on the 0.9 / 0.1 frame split of seed 3072, one AdamW per parameter group,
the trained parts saved per epoch). ``recon_config`` composes each stage's run config (the reconstruction runs'
recon.yaml + data/ogb_pcd.yaml + decoder/page4d.yaml + exp/{frozen,lora}.yaml) from cfg.decoder and
cfg.env; ``train`` runs stage A into <out>/stage_a and stage B into <out>/stage_b and reports each stage's last
validation readout (validate/nn_cm: mean two-sided NN distance, cm, over 100 validation batches), the numbers of Sec.
4.4.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf

try:  # stable-worldmodel (third_party/stable-worldmodel) is needed to run, not to import this module
    import stable_worldmodel as swm
except ImportError:  # pragma: no cover
    swm = None


class PairedFrames(torch.utils.data.Dataset):
    """Row i of ``inputs`` + row i of ``targets``; only ``obs_key`` and ``target`` are kept."""

    def __init__(self, inputs, targets, obs_key, align_on=("qpos",)):
        assert np.array_equal(inputs.lengths, targets.lengths) and np.array_equal(inputs.offsets, targets.offsets), (
            "input and target datasets do not hold the same episodes (ep_len / ep_offset differ)"
        )
        for col in align_on:
            a, b = inputs.get_col_data(col), targets.get_col_data(col)
            assert a.shape == b.shape and np.array_equal(a, b), (
                f"input and target datasets are not the same rows: '{col}' differs (max |diff| "
                f"{np.abs(a - b).max() if a.shape == b.shape else 'shape'})"
            )
        self.inputs, self.targets, self.obs_key = inputs, targets, obs_key
        self.lengths, self.offsets = targets.lengths, targets.offsets

    def get_col_data(self, col):  # per-row columns (object poses, actions) are the same rows in both
        return self.targets.get_col_data(col)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {self.obs_key: self.inputs[idx][self.obs_key], "target": self.targets[idx]["target"]}


def build_dataset(cfg, obs_key, input_keys, input_preprocessor):
    """cfg: ``config/recon/data/*.yaml``. One sample = one frame: ``{obs_key: (1, ...), 'target': (1, N, C)}``."""
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    target_preprocessor = hydra.utils.instantiate(cfg.target_preprocessor)

    def load(name, keys, cache=()):
        return swm.data.load_dataset(
            name,
            transform=None,
            cache_dir=cache_dir,
            num_steps=1,
            frameskip=1,
            keys_to_load=list(keys),
            keys_to_cache=list(cache),
        )

    if cfg.target == cfg.input:
        dataset = load(cfg.input, dict.fromkeys([*input_keys, *cfg.target_keys]))
        # target first: the encoder's own preprocessor may drop the columns it consumes
        dataset.transform = spt.data.transforms.Compose(target_preprocessor, input_preprocessor)
        return dataset

    inputs, targets = load(cfg.input, input_keys, cfg.align_on), load(cfg.target, cfg.target_keys, cfg.align_on)
    inputs.transform, targets.transform = input_preprocessor, target_preprocessor
    return PairedFrames(inputs, targets, obs_key, cfg.align_on)


class Episodes:
    """Whole trajectories out of a frame dataset built by ``build_dataset`` (one dataset, or a row-aligned pair)."""

    def __init__(self, frames):
        self.frames = frames
        self.lengths = [int(n) for n in frames.lengths]
        self.first = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])  # index of step 0 of every episode
        self._cols = {}

    def __len__(self):
        return len(self.lengths)

    def rows(self, ep, stride=1):
        return self.first[ep] + np.arange(0, self.lengths[ep], stride)

    def load(self, ep, stride=1):
        """every ``stride``-th frame of episode ``ep``: ``{obs_key: (K, ...), 'target': (K, N, C)}``."""
        items = [self.frames[int(i)] for i in self.rows(ep, stride)]
        return {k: torch.cat([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}

    def column(self, name, ep=None, stride=1):
        """a per-row column of the dataset (e.g. ``action``, ``privileged_block_0_pos``), raw: the rows of one
        episode, or all rows. ``KeyError`` if the dataset has no such column."""
        if name not in self._cols:
            self._cols[name] = np.asarray(self.frames.get_col_data(name))
            for d in (self.frames, getattr(self.frames, "inputs", None), getattr(self.frames, "targets", None)):
                # reopened lazily; an open handle must not cross a DataLoader fork
                if getattr(d, "h5_file", None) is not None:
                    d.h5_file.close()
                    d.h5_file = None
        return self._cols[name] if ep is None else self._cols[name][self.rows(ep, stride)]


def run_stage(cfg, run_dir):
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger

    from jepa4r.decoder.model import input_pipeline, load_world_model, recon_forward
    from jepa4r.models.world_model import SaveCkptCallback

    ##############################
    # frozen world model     ##
    ##############################

    world_model = load_world_model(cfg.ckpt)  # 2D / 3D / 4D checkpoint -> latent1, never updated
    input_keys, input_preprocessor = input_pipeline(cfg)  # as the checkpoint was trained, see its config.yaml

    #########################
    # dataset       ##
    #########################

    dataset = build_dataset(cfg.data, world_model.obs_key, input_keys, input_preprocessor)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    # decoder / optim     ##
    ##############################

    decoder = hydra.utils.instantiate(cfg.decoder)

    # one optimizer per entry of cfg.optim_groups (name: {modules: regex, lr}), so the new parts and the LoRA adapters
    # of the pretrained backbone have their own learning rate
    base = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizers = {
        name: {
            "modules": g.modules,
            "optimizer": {**base, "lr": g.lr},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
        for name, g in cfg.optim_groups.items()
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=decoder,  # the only trained part, and what SaveCkptCallback saves
        world_model=world_model,
        forward=partial(recon_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    # training       ##
    ##########################

    ckpt_dir = Path(run_dir) / "checkpoints"

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name="",
        cfg=cfg.decoder,
        epoch_interval=1,
        cache_dir=run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = ckpt_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


def recon_config(cfg, stage, init_from=None):
    """The run config of one stage ('a' or 'b') as train_recon.py received it."""
    from jepa4r.config import stage_dir
    from jepa4r.decoder.model import decoder_kwargs

    d, t, env = cfg.decoder, cfg.decoder.train, cfg.env
    st = t[f"stage_{stage}"]
    wm = (
        stage_dir("train_wm", env.name, str(d.wm.method), None, str(d.wm.tag))
        / "checkpoints"
        / f"weights_epoch_{int(d.wm.epoch)}.pt"
    )
    data = str(stage_dir("collect_wm_data", env.name) / env.wm_data.files.pointcloud)
    aff = env.pointcloud.affine
    groups = {"new": {"modules": str(t.param_groups.new), "lr": float(st.lr)}}
    if stage == "b":
        groups["lora"] = {"modules": str(t.param_groups.lora), "lr": float(st.lora_lr)}
    return OmegaConf.create(
        {
            "ckpt": str(wm),
            "output_model_name": f"{d.name}_stage_{stage}",
            "subdir": "",
            "input_keys": None,
            "input_preprocessor": None,  # as the world model was trained (its run's config.yaml)
            "num_workers": int(t.num_workers),
            "train_split": float(t.split),
            "seed": int(t.seed),
            "embed_dim": int(d.latent_dim),
            "num_points": int(d.head.num_points),
            "trainer": {
                "max_epochs": int(st.epochs),
                "limit_train_batches": int(st.steps),
                "limit_val_batches": int(t.val_batches),
                "devices": "auto",
                "accelerator": "gpu",
                "precision": str(t.precision),
                "gradient_clip_val": float(t.gradient_clip),
            },
            "loader": {
                "batch_size": int(st.batch_size),
                "num_workers": int(t.num_workers),
                "persistent_workers": True,
                "prefetch_factor": 3,
                "pin_memory": True,
            },
            "optimizer": {
                "type": "AdamW",
                "lr": float(t.optimizer.lr),
                "weight_decay": float(t.optimizer.weight_decay),
                "betas": [float(b) for b in t.optimizer.betas],
            },
            "wandb": {"enabled": False},
            "data": {
                "input": data,
                "target": data,
                "target_keys": ["pcd_xyz", "pcd_rgb"],
                "align_on": ["qpos"],
                "target_preprocessor": {
                    "_target_": "jepa4r.data.clips.get_pcd_preprocessor",
                    "source_xyz": "pcd_xyz",
                    "source_rgb": "pcd_rgb",
                    "target": "target",
                    "center": [float(c) for c in aff.center],
                    "scale": float(aff.scale),
                    "use_rgb": True,
                    "drop_sources": False,
                },
            },
            "decoder": {
                "_target_": f"{d.backbone.impl}.{d.backbone.decoder_class}",
                "weights": str(d.backbone.weights),
                **decoder_kwargs(cfg, "frozen" if stage == "a" else "lora", init_from),
            },
            "optim_groups": groups,
        }
    )


def train(cfg, out_dir) -> dict:
    from jepa4r.utils.io import last_logged

    out_dir, result = Path(out_dir), {}
    init_from = None
    for stage in ("a", "b"):
        run_dir = out_dir / f"stage_{stage}"
        weights = run_dir / "checkpoints" / f"weights_epoch_{int(cfg.decoder.train[f'stage_{stage}'].epochs)}.pt"
        if not weights.exists():  # a finished stage is not trained again
            run_dir.mkdir(parents=True, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(run_dir)  # Lightning's default logs stay in the run (train_recon.py ran in its Hydra run dir)
            try:
                run_stage(recon_config(cfg, stage, init_from), run_dir)
            finally:
                os.chdir(cwd)
        result[f"val_nn_stage_{stage}"] = last_logged(run_dir, "validate/nn_cm_epoch")
        init_from = weights
    return result
