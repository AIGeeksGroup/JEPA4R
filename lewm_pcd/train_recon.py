from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from recon.data import build_dataset
from recon.model import input_pipeline, load_world_model, recon_forward
from utils import SaveCkptCallback


@hydra.main(version_base=None, config_path="./config/recon", config_name="recon")
def run(cfg):
    ##############################
    ##   frozen world model     ##
    ##############################

    world_model = load_world_model(cfg.ckpt)  # 2D / 3D / 4D checkpoint -> latent1, never updated
    input_keys, input_preprocessor = input_pipeline(cfg)  # as the checkpoint was trained, see its config.yaml

    #########################
    ##       dataset       ##
    #########################

    dataset = build_dataset(cfg.data, world_model.obs_key, input_keys, input_preprocessor)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##      decoder / optim     ##
    ##############################

    decoder = hydra.utils.instantiate(cfg.decoder)

    # one optimizer per entry of cfg.optim_groups (name: {modules: regex, lr}), so the new parts and the LoRA adapters
    # of the pretrained backbone have their own learning rate
    base = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizers = {
        name: {"modules": g.modules, "optimizer": {**base, "lr": g.lr}, "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch"}
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
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'recon', run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=f"recon/{cfg.output_model_name}", cfg=cfg.decoder, epoch_interval=1,
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
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
