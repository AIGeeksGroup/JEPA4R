"""CortexBench loaders for the JEPA-family baselines, same interface as lewm_cortex.load_lewm: each returns
(model, embedding_dim, transforms, metadata); model takes [B, 3, H, W] and returns [B, D], frozen, in eval mode.

* load_pldm: a PLDM checkpoint of the framework (config.json + weights_epoch_N.pt; scripts/run_mw_baselines.sh trains
  one per MetaWorld task on the same data as LeWM). Taken where PLDM's own encode() takes it: the ViT encoder's CLS
  (interpolate_pos_encoding=True) through the projector, 192-d, the space its predictor works in, i.e. the same place
  LeWM is read in CortexBench. Preprocessing as LeWM's: bicubic resize to 224 + ImageNet normalisation.
* load_lejepa: a LeJEPA trained by scripts/baselines/train_lejepa_mw.py (same data as LeWM): the backbone CLS through the
  512-d head, the feature the official recipe probes linearly (not the projector). Preprocessing as LeWM's.
* load_vjepa21: Meta's V-JEPA 2.1 ViT-B (general video JEPA, frozen as released).
"""
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from hydra.utils import instantiate

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# the official V-JEPA 2 code (github.com/facebookresearch/vjepa2 @ 204698b), cloned here; not tracked, see README.md
VJEPA2_REPO = Path(__file__).resolve().parents[2] / 'third_party' / 'vjepa2'


def _transforms():
    return T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.ToTensor(),
                      T.Normalize(*IMAGENET)])


class PLDMEmbedding(torch.nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder, self.projector = encoder, projector

    def forward(self, x):
        cls = self.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return self.projector(cls)


_VIT_SIZES = {'tiny': (192, 12, 3), 'small': (384, 12, 6), 'base': (768, 12, 12), 'large': (1024, 24, 16)}


def _vit_hf(size='tiny', patch_size=16, image_size=224, pretrained=False, use_mask_token=True, **kwargs):
    """The construction of stable_pretraining.backbone.utils.vit_hf (same ViTConfig, add_pooling_layer=False), built
    with transformers directly: importing stable_pretraining.backbone.utils fails in the cortexbench env, whose timm is
    pinned to 0.6.11 for VC-1."""
    from transformers import ViTConfig, ViTModel
    hidden, layers, heads = _VIT_SIZES[size]
    cfg = ViTConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=hidden * 4,
                    image_size=image_size, patch_size=patch_size, **kwargs)
    model = ViTModel(cfg, add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


def _build(node):
    if node.get('_target_') == 'stable_pretraining.backbone.utils.vit_hf':
        return _vit_hf(**{k: v for k, v in node.items() if k != '_target_'})
    return instantiate(node)


def _sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pldm(checkpoint_dir: str, epoch: int = 10):
    ckdir = Path(checkpoint_dir)
    cfg = json.loads((ckdir / 'config.json').read_text())
    encoder, projector = _build(cfg['encoder']), _build(cfg['projector'])
    sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu', weights_only=False)
    encoder.load_state_dict(_sub(sd, 'encoder.'), strict=True)
    projector.load_state_dict(_sub(sd, 'projector.'), strict=True)
    model = PLDMEmbedding(encoder, projector).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        dim = model(torch.zeros(2, 3, 224, 224)).shape[-1]
    metadata = {'algo': 'pldm', 'model': 'vit_tiny_patch14 (PLDM encoder + projector)',
                'checkpoint': str(ckdir / f'weights_epoch_{epoch}.pt')}
    return model, int(dim), _transforms(), metadata


class LeJEPAEmbedding(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])


def load_lejepa(checkpoint_dir: str, epoch: int = 10):
    """HF ViT-tiny/14 (LeWM's backbone) CLS through the 512-d head."""
    ckdir = Path(checkpoint_dir)
    sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu', weights_only=False)
    backbone = _vit_hf(size='tiny', patch_size=14, image_size=224, use_mask_token=False)
    head = torch.nn.Linear(192, 512)
    backbone.load_state_dict(sd['backbone'], strict=True)
    head.load_state_dict(sd['head'], strict=True)
    model = LeJEPAEmbedding(backbone, head).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    metadata = {'algo': 'lejepa', 'model': 'vit_tiny_patch14 + Linear(192,512) (LeJEPA)',
                'checkpoint': str(ckdir / f'weights_epoch_{epoch}.pt')}
    return model, 512, _transforms(), metadata


def build_vjepa21_encoder(checkpoint):
    """The encoder of the official hub's vjepa2_1_vit_base_384 (src/hub/backbones.py, _make_vjepa2_1_model): ViT-B/16,
    RoPE, single frames through the image tokenizer (img_temporal_dim_size=1); the checkpoint's ema_encoder, strict."""
    if str(VJEPA2_REPO) not in sys.path:
        sys.path.insert(0, str(VJEPA2_REPO))
    from app.vjepa_2_1.models import vision_transformer as vit_encoder
    enc = vit_encoder.vit_base(
        patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
        wide_SiLU=True, uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)['ema_encoder']
    sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}  # the hub's _clean_backbone_key
    enc.load_state_dict(sd, strict=True)
    return enc


class VJEPA21Embedding(torch.nn.Module):
    """A frame [B, 3, H, W] -> image-path input [B, 3, 1, H, W] -> mean of the last layer's patch tokens (after the
    encoder's own final LayerNorm), 768-d. V-JEPA 2.1 has no CLS token."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x[:, :, None]).mean(dim=1)


def load_vjepa21(checkpoint: str, resolution: int = 384):
    """V-JEPA 2.1 ViT-B (vjepa2_1_vitb_dist_vitG_384.pt), frozen. Preprocessing as its official evaluations: ImageNet
    normalisation at 384 (its released resolution, used by all its downstream evaluations); CortexBench's 256 x 256
    frame is resized whole (bicubic), not cropped, keeping the full view like the other encoders."""
    model = VJEPA21Embedding(build_vjepa21_encoder(checkpoint)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    transforms = T.Compose([T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                            T.ToTensor(), T.Normalize(*IMAGENET)])
    metadata = {'algo': 'vjepa2.1', 'model': 'vit_base_patch16 (V-JEPA 2.1, distilled from ViT-G, 384)',
                'checkpoint': str(checkpoint), 'resolution': resolution}
    return model, 768, transforms, metadata
