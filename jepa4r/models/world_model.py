"""JEPA world model (impl of method kind ``jepa_wm``): JEPA4R and LeWM, which differ only in the encoder.

f_theta = encoder (cfg.method.encoder) + projector;  g_phi = predictor;  action encoder   (Sec. 3, Sec. 4.2)
Training (LeWM recipe on stable-worldmodel [maes2026stableworldmodel]): clips of 4 frames 5 env steps apart,
teacher forcing, AdamW lr 5e-5, batch 128, 10 epochs, bf16, no stop-gradient / EMA target,
    L = 1/(3d) * sum_{t=1..3} ||z_hat_{t+1} - z_{t+1}||^2 + 0.09 * SIGReg(Z)        (Eq. 1)
Interface used by the rest of the code:
    encode(obs [B, ...]) -> z [B, 192]
    predict(z_hist [B, T, 192], a_hist [B, T, A*5]) -> z_next [B, 192]           (T <= history)
    rollout(z_ctx [B, 3, 192], action_blocks [B, H, A*5]) -> z_hat [B, H, 192]   (autoregressive; Sec. 4.4)

``JEPA`` is LeWM's ``jepa.py`` (dict interface: ``encode(info)`` fills info['emb'] / info['act_emb'],
``predict(emb, act_emb)``, ``rollout(info, action_sequence, history_size)``); training is LeWM's ``train.py`` run on
stable-worldmodel / stable-pretraining, in its two original variants:
  run_random_split   one dataset file, random 0.9 / 0.1 clip split with torch.Generator().manual_seed(seed), the same
                     generator shuffling the train loader (+ the Cube workflow's seed hook)
  run_episode_split  Cube 1-4: an independent validation file (whole held-out episodes), seed 0 set
                     with Lightning's seed_everything before the data are built
Checkpoints: ``<out_dir>/checkpoints/{config.json, weights_epoch_<n>.pt}`` (stable-worldmodel ``save_pretrained``, the
format of the released HF checkpoints, which ``load_checkpoint`` also reads).
"""
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from torch import nn

from jepa4r.data.clips import get_column_normalizer
from jepa4r.models.sigreg import SIGReg

try:  # Lightning is needed to train, not to import this module (tests import every impl module)
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover
    Callback = object

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        obs_key="pixels",
    ):
        super().__init__()

        self.obs_key = obs_key  # observation key fed to the encoder ('pixels' or 'pcd')
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info[self.obs_key].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        pixels_emb = self.encoder(pixels)  # (B*T, D) -- see encoders/ for the contract
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert self.obs_key in info, f"{self.obs_key} not in info_dict"
        H = info[self.obs_key].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal[self.obs_key] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


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

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, cache_dir=None):
        super().__init__()
        self.cache_dir = cache_dir
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
            cache_dir=self.cache_dir,
        )


def run_random_split(cfg, run_dir, *, manager_seed=None, original_layout=False):
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.loggers import WandbLogger

    if manager_seed is not None:
        manager_seed = int(manager_seed)
        pl.seed_everything(manager_seed, workers=True)
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
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

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
        # e.g. jepa_agg.agg_forward
        forward=partial(hydra.utils.get_method(cfg.forward) if cfg.get("forward") else lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    if original_layout:  # the original layout, which the Relocate / RH56 workflow reads back
        run_id = cfg.get("subdir") or ""
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    else:
        run_dir = Path(run_dir)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    config_dir = run_dir if original_layout else run_dir / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.yaml", "w") as f:  # next to the weights (subdir == output_model_name)
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name if original_layout else '', cfg=legacy_config(cfg.model), epoch_interval=1,
        cache_dir=None if original_layout else run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager_seed_kwargs = {} if manager_seed is None else {"seed": manager_seed}
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        **manager_seed_kwargs,
    )

    manager()
    return

def _prepare_fresh_run(cfg, run_dir, exist_ok=True):
    import lightning as pl

    seed = cfg.get("fresh_seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or int(seed) != seed or int(seed) != cfg.seed:
        raise ValueError("fresh_seed must be an integer equal to cfg.seed")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=exist_ok)  # the stage directory exists already (jepa4r-run creates it)
    pl.seed_everything(int(seed), workers=True)
    return int(seed)


def _split_datasets(cfg, dataset, transform, cache_dir, rnd_gen):
    import stable_worldmodel as swm

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


def run_episode_split(cfg, run_dir, *, resume_checkpoint=None, state_callback=None, original_layout=False):
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.loggers import WandbLogger

    if original_layout:  # the source's own layout, which the moved Cube workflow (jepa4r/models/cube) reads back
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.get("subdir") or "")
    fresh_seed = _prepare_fresh_run(cfg, run_dir, exist_ok=not original_layout)
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
        # e.g. jepa_agg.agg_forward
        forward=partial(hydra.utils.get_method(cfg.forward) if cfg.get("forward") else lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(run_dir)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    config_dir = run_dir if original_layout else run_dir / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.yaml", "w") as f:  # next to the weights (subdir == output_model_name)
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name if original_layout else '', cfg=legacy_config(cfg.model), epoch_interval=1,
        cache_dir=None if original_layout else run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback] + ([] if state_callback is None else [state_callback]),
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=resume_checkpoint if resume_checkpoint is not None else ckpt_path if ckpt_path.exists() else None,
        seed=fresh_seed,
    )

    manager()
    return


# --------------------------------------------------------------------------- jepa4r interfaces (adapters)
# Hydra targets written into config.json by LeWM's train.py (released HF checkpoints, local runs) -> the moved classes.
LEGACY_TARGETS = {
    'jepa.JEPA': 'jepa4r.models.world_model.JEPA',
    'module.ARPredictor': 'jepa4r.models.predictor.ARPredictor',
    'module.Embedder': 'jepa4r.models.heads.Embedder',
    'module.MLP': 'jepa4r.models.heads.MLP',
    'encoders.DexArtPointNet': 'jepa4r.models.encoders.pointnet.DexArtPointNet',
    'encoders.ViTCLS': 'jepa4r.models.encoders.vit.ViTCLS',
    'encoders.PointNet2': 'jepa4r.models.encoders.pointnet2.PointNet2',
    'encoders.DGCNN': 'jepa4r.models.encoders.dgcnn.DGCNN',
    'encoders.PointPatchTransformer': 'jepa4r.models.encoders.point_patch_transformer.PointPatchTransformer',
    'utils.get_pcd_preprocessor': 'jepa4r.data.clips.get_pcd_preprocessor',
    'utils.get_img_preprocessor': 'jepa4r.data.clips.get_img_preprocessor',
}


_LEGACY_OF = {new: old for old, new in LEGACY_TARGETS.items()}


def legacy_config(config):
    """A model block with the moved targets written back as LeWM's train.py names them, so that config.json has the
    layout of the paper runs' checkpoints (the released ones and every loader moved here read those names)."""
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)

    def back(node):
        if isinstance(node, dict):
            return {k: (_LEGACY_OF.get(v, v) if k == '_target_' else back(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [back(v) for v in node]
        return node
    return back(config)


def _retarget(node):
    if isinstance(node, dict):
        return {k: (LEGACY_TARGETS.get(v, v) if k == '_target_' else _retarget(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_retarget(v) for v in node]
    return node


def instantiate(config):
    """hydra.utils.instantiate of a LeWM model block, old (released) or new targets."""
    import hydra

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    return hydra.utils.instantiate(OmegaConf.create(_retarget(config)))


def encoder_config(enc_cfg, latent_dim, in_dim=6):
    """configs/encoder/<name>.yaml -> the encoder block of LeWM's config/train/model/encoder/*.yaml."""
    t = LEGACY_TARGETS
    if enc_cfg.arch == 'pointnet':
        return {'_target_': t['encoders.DexArtPointNet'], 'in_dim': in_dim, 'mlp_spec': list(enc_cfg.hidden),
                'out_dim': latent_dim}
    if enc_cfg.arch == 'vit':
        return {'_target_': t['encoders.ViTCLS'], 'size': str(enc_cfg.variant), 'patch_size': int(enc_cfg.patch_size),
                'image_size': int(enc_cfg.get('image_size', 224)), 'pretrained': bool(enc_cfg.get('pretrained', False)),
                'use_mask_token': False}
    h = OmegaConf.to_container(enc_cfg.hparams, resolve=True)
    name = {'pointnet2': 'encoders.PointNet2', 'dgcnn': 'encoders.DGCNN',
            'point_patch_transformer': 'encoders.PointPatchTransformer'}[enc_cfg.arch]
    return {'_target_': t[name], 'in_dim': in_dim, **h, 'out_dim': latent_dim}


def model_config(method_cfg, action_dim=None):
    """The Hydra model block of LeWM's config/train/model/lewm.yaml / lewm_pcd.yaml for this method."""
    d, p = int(method_cfg.latent_dim), method_cfg.predictor
    t = LEGACY_TARGETS

    def mlp(head):
        return {'_target_': t['module.MLP'], 'input_dim': d, 'output_dim': d, 'hidden_dim': int(head.hidden_dim),
                'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True}}

    return {
        '_target_': t['jepa.JEPA'],
        'obs_key': 'pcd' if method_cfg.obs == 'pointcloud' else 'pixels',
        'encoder': encoder_config(method_cfg.encoder, d),
        'predictor': {'_target_': t['module.ARPredictor'], 'num_frames': int(method_cfg.history), 'input_dim': d,
                      'hidden_dim': d, 'output_dim': d, 'depth': int(p.depth), 'heads': int(p.heads),
                      'mlp_dim': int(p.mlp_dim), 'dim_head': int(p.dim_head), 'dropout': float(p.dropout),
                      'emb_dropout': float(p.emb_dropout)},
        'action_encoder': {'_target_': t['module.Embedder'],
                           'input_dim': '???' if action_dim is None else int(method_cfg.frameskip) * int(action_dim),
                           'emb_dim': int(method_cfg.action_encoder.emb_dim)},
        'projector': mlp(method_cfg.projector),
        'pred_proj': mlp(method_cfg.pred_proj),
    }


def build_world_model(method_cfg, action_dim):
    return instantiate(model_config(method_cfg, action_dim))


def lewm_train_config(cfg):
    """The configuration LeWM's train.py (config/train/lewm.yaml + data/<task>.yaml + model/<encoder>.yaml) receives,
    composed from cfg.method (recipe) and cfg.env (dataset files, affine, split)."""
    from jepa4r.config import stage_dir
    from jepa4r.data.clips import dataset_kwargs, preprocessor_kwargs

    m, wm = cfg.method, cfg.env.wm_data
    obs, data_dir = m.obs, stage_dir('collect_wm_data', cfg.env.name)
    files = wm.files
    data = {'dataset': {'name': str(data_dir / files[obs]), **dataset_kwargs(m, obs)},
            'preprocessor': preprocessor_kwargs(cfg.env, obs)}
    split = str(wm.get('split', 'random_clips'))
    if split == 'episodes':  # Cube 1-4: whole validation episodes in their own file
        data['validation_dataset'] = {'name': str(data_dir / files[f'{obs}_val']), **dataset_kwargs(m, obs)}
    t = m.train
    lw = {
        'output_model_name': str(m.name),
        'subdir': '',
        'num_workers': int(t.num_workers),
        'train_split': float(t.train_split),
        'seed': int(t.seed),
        'img_size': 224,
        'embed_dim': int(m.latent_dim),
        'pcd_in_dim': 6,
        'history_size': int(m.history),
        'num_preds': 1,
        'trainer': {
            'max_epochs': int(t.epochs),
            'devices': 'auto',
            'accelerator': 'gpu',
            'precision': str(t.precision),
            'gradient_clip_val': float(t.gradient_clip),
        },
        'loader': {
            'batch_size': int(t.batch_size),
            'num_workers': int(t.num_workers),
            'persistent_workers': True,
            'prefetch_factor': 3,
            'pin_memory': True,
        },
        'optimizer': {'type': 'AdamW', 'lr': float(t.lr), 'weight_decay': float(t.weight_decay)},
        'loss': {
            'sigreg': {
                'weight': float(m.loss.sigreg_weight),
                'kwargs': {'knots': int(m.loss.sigreg_knots), 'num_proj': int(m.loss.sigreg_projections)},
            }
        },
        'wandb': {'enabled': False},
        'data': data,
        'model': model_config(m),
    }
    if split == 'episodes':
        lw['fresh_seed'] = int(t.seed)
        lw['validation_num_workers'] = int(t.num_workers)
    return OmegaConf.create(lw), split


def train(cfg, out_dir) -> dict:
    """Stage train_wm for kind jepa_wm: one checkpoint per epoch in out_dir; returns final losses."""
    lw, split = lewm_train_config(cfg)
    out_dir = Path(out_dir)
    cwd = os.getcwd()
    os.chdir(out_dir)  # LeWM ran in its Hydra run dir (hydra.job.chdir): Lightning's default logs stay in the run
    try:
        if split == 'episodes':
            run_episode_split(lw, out_dir)
        else:
            run_random_split(lw, out_dir)
    finally:
        os.chdir(cwd)
    return training_summary(out_dir)


def training_summary(out_dir) -> dict:
    """Last logged train / validation losses (Lightning's default log in out_dir) and the saved checkpoints."""
    from jepa4r.utils.io import last_logged

    out_dir = Path(out_dir)
    result = {'checkpoints': sorted(p.name for p in (out_dir / 'checkpoints').glob('weights_epoch_*.pt'))}
    for key in ('train/loss', 'train/pred_loss', 'train/sigreg_loss', 'val/loss', 'val/pred_loss'):
        value = last_logged(out_dir, key)
        if value is not None:
            result[key.replace('/', '_')] = value
    return result


def load_checkpoint(path, epoch=10, device='cpu', strict=True):
    """A LeWM checkpoint directory (config.json + weights_epoch_<epoch>.pt; the layout of save_pretrained and of the
    released HF checkpoints) -> the JEPA model in eval mode."""
    path = Path(path)
    if (path / 'checkpoints' / 'config.json').is_file():
        path = path / 'checkpoints'
    model = instantiate(json.loads((path / 'config.json').read_text(encoding='utf-8')))
    state = torch.load(path / f'weights_epoch_{epoch}.pt', map_location=device, weights_only=True)
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()


def load_world_model(env, method, tag="default", epoch=10):
    """Frozen world model from outputs/train_wm/<env>/<method>/<tag>/ (decoder, imagination)."""
    from jepa4r.config import stage_dir

    return load_checkpoint(stage_dir('train_wm', env, method, None, tag), epoch=epoch)


class FrozenWorldModelEncoder:
    """f_theta (encoder + projector) of a frozen LeWM checkpoint: preprocessed observations [B, ...] -> [B, 192]."""

    def __init__(self, model, obs, device='cpu'):
        self.model, self.obs, self.device = model.to(device).eval(), obs, device
        self.latent_dim = int(model.projector.net[-1].out_features) if hasattr(model.projector, 'net') else None

    @torch.no_grad()
    def encode(self, batch):
        x = torch.as_tensor(batch).to(self.device).float()
        return self.model.projector(self.model.encoder(x))


def load_frozen_encoder(cfg):
    """FrozenEncoder (jepa4r/policy/frozen.py) = f_theta of the epoch-10 checkpoint, for train_policy (read where
    jepa4r.policy.frozen.checkpoint says: the train_wm run, or on Cube 1-4 the Cube workflow's accepted export)."""
    from jepa4r.policy.frozen import checkpoint

    model = load_checkpoint(checkpoint(cfg), epoch=int(cfg.method.get('policy_epoch', 10)))
    return FrozenWorldModelEncoder(model, cfg.method.obs, device=str(cfg.get('device', 'cpu')))
