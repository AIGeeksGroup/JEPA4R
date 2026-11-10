"""One current Cube episode-heldout native LeWM trainer."""
import os
from functools import partial
from pathlib import Path
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from module import SIGReg
from utils import get_column_normalizer, SaveCkptCallback


def _prepare_fresh_run(cfg):
    seed = cfg.get("fresh_seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or int(seed) != seed or int(seed) != cfg.seed:
        raise ValueError("fresh_seed must be an integer equal to cfg.seed")
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(int(seed), workers=True)
    return int(seed)


def _split_datasets(cfg, dataset, transform, cache_dir, rnd_gen):
    validation = cfg.data.get("validation_dataset")
    if validation is None:
        raise ValueError("current protocol requires independent validation episodes")
    validation_cfg = OmegaConf.to_container(validation, resolve=True)
    validation_name = validation_cfg.pop("name")
    validation_set = swm.data.load_dataset(
        validation_name, transform=transform, cache_dir=cache_dir, **validation_cfg
    )
    return dataset, validation_set


def _validation_loader_kwargs(cfg):
    kwargs = dict(cfg.loader)
    workers = cfg.get("validation_num_workers")
    if workers is not None:
        kwargs["num_workers"] = int(workers)
        if int(workers) == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
    return kwargs


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def run(cfg):
    fresh_seed = _prepare_fresh_run(cfg)
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [hydra.utils.instantiate(cfg.data.preprocessor)]  # pixels or point cloud, see config/train/data
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col.startswith("pcd"):
                continue  # observation columns are handled by the preprocessor above
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = _split_datasets(cfg, dataset, transform, cache_dir, rnd_gen)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **_validation_loader_kwargs(cfg), shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(hydra.utils.get_method(cfg.forward) if cfg.get("forward") else lejepa_forward, cfg=cfg),  # e.g. jepa_agg.agg_forward
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        seed=fresh_seed,
    )

    manager()
    return
