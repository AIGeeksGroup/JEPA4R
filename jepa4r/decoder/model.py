"""PointCloudDecoder: set generator -> frozen backbone (+ LoRA in stage B) -> head; z [B, 192] -> P_hat [B, 1200, 6].

build_decoder(decoder_cfg, **kwargs)     the decoder of cfg.decoder.backbone (jepa4r/decoder/backbones)
decoder_kwargs(cfg, train, init_from)    its constructor arguments from cfg.decoder and the target frame of cfg.env
load_decoder(env, decoder, tag, stage)   a trained decoder of outputs/train_decoder/<env>/<decoder>/<tag>/stage_<a|b>/
load_decoder_checkpoint(path)            any decoder run (config.json + weights_epoch_<n>.pt), incl. the released ones
From the paper's reconstruction code: FrozenLatent (the frozen world model as observation -> per-frame latent),
input_pipeline (what the encoder reads, from the world model's run config), load_recon / open_run (a trained run with
its world model and frames), recon_forward (the training step). World models load through
jepa4r.models.world_model.load_checkpoint; the released decoder configs name recon.decoders.PAGE4DDecoder /
recon_backbones.* (DECODER_TARGETS maps them to the classes in this repository).
"""

import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn

from jepa4r.config import paths
from jepa4r.decoder.losses import nn_distance
from jepa4r.models.world_model import load_checkpoint

try:  # stable-worldmodel (third_party/stable-worldmodel) is needed to run, not to import this module
    import stable_worldmodel as swm
except ImportError:  # pragma: no cover
    swm = None

DECODER_TARGETS = {  # _target_ of the released decoder configs (HF AIGeeksGroup/JEPA4R */decoder/*/config.json)
    "recon.decoders.PAGE4DDecoder": "jepa4r.decoder.backbones.page4d.PAGE4DDecoder",
    "recon_backbones.VGGTDecoder": "jepa4r.decoder.backbones.vggt.VGGTDecoder",
    "recon_backbones.DA3Decoder": "jepa4r.decoder.backbones.depth_anything3.DA3Decoder",
}
PIXEL_SELECTION = "random_train_fixed_eval"  # PAGE4DDecoder._pick


class FrozenLatent(nn.Module):
    """A trained world model reduced to its per-frame latent. Never trains, never leaves eval mode (the
    projector's BatchNorm statistics and every dropout stay as trained) -- also when Lightning calls ``train()``."""

    def __init__(self, world_model):
        super().__init__()
        assert hasattr(world_model, "obs_key") and hasattr(world_model, "encode"), (
            f"{type(world_model).__name__} is not a jepa.JEPA world model"
        )
        self.world_model = world_model.requires_grad_(False).eval()
        self.obs_key = world_model.obs_key

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, obs):
        """obs (B, T, ...) -> latent1 (B, T, D); ``JEPA.encode`` for images or point clouds, inherited unchanged by
        the 4D model."""
        return self.world_model.encode({self.obs_key: obs})["emb"]


def load_world_model(ckpt):
    ckpt = Path(ckpt)  # <run>/weights_epoch_N.pt, absolute or under JEPA4R_CHECKPOINTS
    ckpt = ckpt if ckpt.is_absolute() else Path(paths()["checkpoints"], ckpt)
    return FrozenLatent(load_checkpoint(ckpt.parent, epoch=int(ckpt.stem.rsplit("_", 1)[1])))


def input_pipeline(cfg):
    """Columns and preprocessor that feed the encoder. By default those the checkpoint was trained with, read
    from its run's ``config.yaml`` (pixels, a point-cloud affine, Sonata's identity affine, ...);
    ``input_keys`` / ``input_preprocessor`` in the config override them."""
    keys, preprocessor = cfg.get("input_keys"), cfg.get("input_preprocessor")
    if keys is None or preprocessor is None:
        ckpt = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.ckpt)
        run_cfg = (ckpt.parent if ckpt.suffix == ".pt" else ckpt) / "config.yaml"
        assert run_cfg.exists(), f"{run_cfg} not found: set input_keys and input_preprocessor explicitly"
        data = OmegaConf.load(run_cfg).data
        if keys is None:  # observation columns, by the rule train.py uses
            keys = [k for k in data.dataset.keys_to_load if k.startswith("pixels") or k.startswith("pcd")]
        if preprocessor is None:
            preprocessor = data.preprocessor
    return list(keys), hydra.utils.instantiate(preprocessor)


def load_recon(name):
    """A trained reconstruction run -> (frozen world model, decoder, the run's config).
    ``name``: a run's checkpoint directory (its last epoch) or ``<dir>/weights_epoch_N.pt``."""
    path = Path(name)
    if path.suffix != ".pt":
        weights = sorted(path.glob("weights_epoch_*.pt"), key=lambda w: int(w.stem.rsplit("_", 1)[1]))
        assert weights, f"no weights_epoch_N.pt in {path}"
        path = weights[-1]
    cfg = OmegaConf.load(path.parent / "config.yaml")
    decoder = load_decoder_checkpoint(path).eval().requires_grad_(False)
    return load_world_model(cfg.ckpt), decoder, cfg


def open_run(name, input=None, target=None, device="cuda"):
    """Everything the evaluation scripts need from a trained run: (world model, decoder, config, ``Episodes``).
    ``input`` / ``target`` replace the run's training datasets, e.g. by other trajectories -- for a pixel checkpoint
    the pixel dataset and the point clouds of the same rows, checked exactly as in training."""
    from jepa4r.decoder.train import Episodes, build_dataset

    world_model, decoder, cfg = load_recon(name)
    if input is not None:
        cfg.data.input = input
    if target is not None:
        cfg.data.target = target
    input_keys, input_preprocessor = input_pipeline(cfg)
    frames = build_dataset(cfg.data, world_model.obs_key, input_keys, input_preprocessor)
    return world_model.to(device), decoder.to(device), cfg, Episodes(frames)


def recon_forward(self, batch, stage, cfg):
    """encode the frame with the frozen world model, decode its latent, compare with the target point cloud."""
    latent = self.world_model(batch[self.world_model.obs_key])[:, 0]  # (B, D)
    target = batch["target"][:, 0]  # (B, N, C)

    decoded = self.model(latent)
    loss, terms = self.model.loss(decoded, target)
    output = {"loss": loss, **terms}
    if stage != "fit":  # the same readout for every decoder: two-sided nearest-neighbour distance, in cm
        points = self.model.to_points(decoded, target.size(1))
        output["nn_cm"] = nn_distance(points, target).mean() / cfg.data.target_preprocessor.scale * 100

    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items()}, on_step=True, sync_dist=True)
    return output


def load_decoder_checkpoint(path):
    """config.json + weights_epoch_<n>.pt (``path``: the file, or its directory -> the last epoch) -> the decoder, as
    stable-worldmodel's load_pretrained builds it (instantiate the config, load the state dict strictly; the frozen
    pretrained weights come back from the config's ``weights`` in the constructor)."""
    path = Path(path)
    if path.suffix != ".pt":
        path = sorted(path.glob("weights_epoch_*.pt"), key=lambda w: int(w.stem.rsplit("_", 1)[1]))[-1]
    conf = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    conf["_target_"] = DECODER_TARGETS.get(conf["_target_"], conf["_target_"])
    decoder = hydra.utils.instantiate(conf)
    decoder.load_state_dict(torch.load(path, map_location="cpu"))
    return decoder


def decoder_kwargs(cfg, train="frozen", init_from=None):
    """Constructor arguments of the backbone decoders (the reconstruction runs' decoder/page4d.yaml and
    exp/{frozen,lora}.yaml) from cfg.decoder; the target frame = cfg.env.pointcloud.affine. ``train``: 'frozen' (stage
    A) or 'lora' (stage B, the new parts starting from the stage-A weights ``init_from``)."""
    d, sg, head, loss = cfg.decoder, cfg.decoder.set_generator, cfg.decoder.head, cfg.decoder.loss
    grid = math.isqrt(int(sg.num_queries))
    assert grid * grid == int(sg.num_queries), "num_queries must be a square grid"
    assert str(head.pixel_selection) == PIXEL_SELECTION, head.pixel_selection
    assert float(loss.chamfer) == 1.0
    aff = cfg.env.pointcloud.affine
    return dict(
        latent_dim=int(d.latent_dim),
        grid=grid,
        num_points=int(head.num_points),
        train=train,
        lora_rank=int(d.train.stage_b.lora_rank),
        init_from=None if init_from is None else str(init_from),
        color=bool(head.rgb_branch),
        color_weight=float(loss.rgb_l1),
        sw_weight=float(loss.sliced_w2),
        sw_projections=int(loss.sw_projections),
        gen_dim=int(sg.dim),
        gen_hidden=int(sg.hidden),
        gen_depth=int(sg.num_blocks),
        query_dim=int(sg.query_dim),
        target_center=[float(c) for c in aff.center],
        target_scale=float(aff.scale),
        view=[float(v) for v in head.init_view],
    )


def build_decoder(decoder_cfg, **kwargs):
    from jepa4r.decoder.backbones import build_backbone

    return build_backbone(decoder_cfg.backbone, **kwargs)


def load_decoder(env, decoder, tag="default", stage="b"):
    """Checkpoint from outputs/train_decoder/<env>/<decoder>/<tag>/."""
    from jepa4r.config import stage_dir

    return load_decoder_checkpoint(
        stage_dir("train_decoder", env, None, decoder, tag) / f"stage_{stage}" / "checkpoints"
    )


def count_parameters(decoder_cfg):
    """Sec. 4.4 counts from the modules alone (stage count_params: no checkpoints): new_params_M = the stage-A
    trainables (set generator, affine map, rgb branch), lora_params_M = the stage-B adapters, backbone_params_B = the
    frozen backbone as the decoder keeps it (patch projection, camera / depth / track and mask heads dropped). The
    PAGE-4D architecture (page4d, vggt) is built without its released weights, swapping the loader as VGGTDecoder does;
    the counts do not depend on the weights."""
    import jepa4r.decoder.backbones.page4d as p4

    if str(decoder_cfg.backbone.name) not in ("page4d", "vggt"):
        raise ValueError("decoder parameters are counted for the PAGE-4D architecture (page4d, vggt)")
    sg, head = decoder_cfg.set_generator, decoder_cfg.head
    loader, p4.load_page4d = (
        p4.load_page4d,
        lambda weights: p4.PAGE4D(enable_camera=False, enable_depth=False, enable_track=False),
    )
    try:
        dec = p4.PAGE4DDecoder(
            latent_dim=int(decoder_cfg.latent_dim),
            grid=math.isqrt(int(sg.num_queries)),
            num_points=int(head.num_points),
            train="lora",
            lora_rank=int(decoder_cfg.train.stage_b.lora_rank),
            color=bool(head.rgb_branch),
            gen_dim=int(sg.dim),
            gen_hidden=int(sg.hidden),
            gen_depth=int(sg.num_blocks),
            query_dim=int(sg.query_dim),
        )
    finally:
        p4.load_page4d = loader

    def is_lora(name):
        return any(t in name for t in p4.ADAPTER_KEYS)

    params = list(dec.named_parameters())
    new = sum(p.numel() for n, p in params if n.split(".")[0] in ("generator", "frame", "rgb"))
    lora = sum(p.numel() for n, p in params if is_lora(n))
    backbone = sum(p.numel() for n, p in params if n.split(".")[0] in p4.PRETRAINED and not is_lora(n))
    return {"new_params_M": new / 1e6, "lora_params_M": lora / 1e6, "backbone_params_B": backbone / 1e9}
