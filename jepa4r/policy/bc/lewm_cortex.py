# Moved from cortexbench@9331494:lewm_pcd/bench/cortexbench/lewm_cortex.py (verbatim except: imports / paths adapted to
# the package; lines wrapped at 120 columns)
"""vc_models entry point for LeWM encoders in CortexBench (the frozen-PVR imitation protocol of eai-vc).

CortexBench loads an encoder through `hydra.utils.call(<vc_models conf/model/NAME.yaml>)`, which must return
(model, embedding_dim, transforms, metadata); the model maps a batch of preprocessed frames (B, 3, 224, 224) to one
vector per frame. For LeWM that vector is the encoder's CLS token passed through the projector: the 192-d latent the
LeWM predictor and planner work in. The model is rebuilt exactly as stable_worldmodel's load_pretrained does it
(hydra-instantiate the checkpoint's config.json, then load weights_epoch_N.pt); needs lewm_pcd/ on PYTHONPATH.

Preprocessing: LeWM was trained on full 224 x 224 renders with ImageNet normalisation, so the 256 x 256 CortexBench
frames are resized (not centre-cropped, which would narrow the field of view the model was trained on) to 224 and
normalised the same way. random_init=True gives the same architecture with untrained weights (the analogue of the
paper's 'Random (ViT) Frozen' rows).
"""

import json
from pathlib import Path

import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from torch import nn

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _moved(node):
    """config.json's 'module.MLP' -> the moved class (jepa4r.models.heads.MLP); everything else unchanged."""
    if isinstance(node, dict):
        return {
            k: ('jepa4r.models.heads.MLP' if (k == '_target_' and v == 'module.MLP') else _moved(v))
            for k, v in node.items()
        }
    return node


VIT_SIZES = {'tiny': (192, 12, 3), 'small': (384, 12, 6), 'base': (768, 12, 12)}  # stable_pretraining vit_hf


class ViTCLS(nn.Module):
    """encoders.vit.ViTCLS rebuilt without stable_pretraining: its module-level `from transformers import
    TimmWrapperModel, ...` fails under the timm 0.6.11 that CortexBench pins for VC-1. Same HF ViTModel, same config,
    same state-dict keys (vit.*), same forward (CLS token, interpolate_pos_encoding=True)."""

    def __init__(self, size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False, **kwargs):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        assert not pretrained
        hidden, layers, heads = VIT_SIZES[size]
        config = ViTConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                           intermediate_size=4 * hidden, image_size=image_size, patch_size=patch_size, **kwargs)
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        self.vit.config.interpolate_pos_encoding = True

    def forward(self, x):
        return self.vit(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]


def _train_data(ckdir):
    """The dataset a LeWM checkpoint was trained on (the hydra config.yaml train.py writes next to it), for metadata."""
    p = Path(ckdir) / 'config.yaml'
    if not p.exists():
        return []
    import yaml
    return [yaml.safe_load(p.read_text())['data']['dataset']['name']]


class LeWMEmbedding(nn.Module):
    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()

    def forward(self, x):
        return self.projector(self.encoder(x))


def build_lewm_embedding(checkpoint_dir, epoch=10, use_projector=True, random_init=False, seed=0):
    ckdir = Path(checkpoint_dir)
    cfg = json.loads((ckdir / 'config.json').read_text())
    enc_cfg = {k: v for k, v in cfg['encoder'].items() if k != '_target_'}
    assert cfg['encoder']['_target_'] in ('encoders.ViTCLS', 'encoders.vit.ViTCLS'), cfg['encoder']['_target_']
    # the initial (random_init: final) weights come from `seed`; the caller's global RNG is left as it was (CortexBench
    # loads the encoder again right before its BC dataloader draws the shuffle order, which must follow the run's seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = ViTCLS(**enc_cfg)
        projector = instantiate(_moved(cfg['projector'])) if use_projector else None  # module.MLP (jepa4r.models.heads)
    if not random_init:
        sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu')
        encoder.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
        if projector is not None:
            projector.load_state_dict({k[len('projector.'):]: v for k, v in sd.items() if k.startswith('projector.')})
    return LeWMEmbedding(encoder, projector).eval()


def load_lewm(
    checkpoint_dir: str, epoch: int = 10, use_projector: bool = True, random_init: bool = False, seed: int = 0
):
    ckdir = Path(checkpoint_dir)
    model = build_lewm_embedding(ckdir, epoch, use_projector, random_init, seed)
    with torch.no_grad():
        dim = model(torch.zeros(2, 3, 224, 224)).shape[-1]
    transforms = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(*IMAGENET),
    ])
    metadata = {
        'algo': 'random' if random_init else 'lewm',
        'model': 'vit_tiny_patch14 (LeWM encoder + projector)',
        'data': [] if random_init else _train_data(ckdir),
        'checkpoint': None if random_init else str(ckdir / f'weights_epoch_{epoch}.pt'),
    }
    return model, int(dim), transforms, metadata


# ---------------------------------------------------------------- 3D (point cloud) LeWM
class PcdTransform:
    """(N, 6) numpy point cloud [xyz in metres, rgb 0..255] -> (N, 6) float tensor, as the 3D LeWM's training
    preprocessor (utils.PointCloudPreprocessor): (xyz - center) * scale ; rgb / 255."""

    def __init__(self, center, scale, use_rgb=True):
        self.center = torch.tensor([float(c) for c in center], dtype=torch.float32)
        self.scale, self.use_rgb = float(scale), use_rgb

    def __call__(self, pcd):
        x = torch.as_tensor(pcd, dtype=torch.float32)
        xyz = (x[:, :3] - self.center) * self.scale
        return torch.cat([xyz, x[:, 3:6] / 255.0], dim=-1) if self.use_rgb else xyz


# the 3D LeWM encoders (lewm_pcd/encoders, config/train/model/encoder/{dexart,ms2,ppt,pointnet2,dgcnn}.yaml):
# the _target_ in a checkpoint's config.json -> (file in lewm_pcd/encoders, class, name for the metadata)
PCD_ENCODERS = {
    'encoders.DexArtPointNet': ('pointnet.py', 'DexArtPointNet', 'DexArt PointNet'),
    'encoders.pointnet_dexart.DexArtPointNet': ('pointnet.py', 'DexArtPointNet', 'DexArt PointNet'),
    'encoders.ManiSkillPointNet': ('pointnet_ms2.py', 'ManiSkillPointNet', 'ManiSkill PointNet'),
    'encoders.pointnet_ms2.ManiSkillPointNet': ('pointnet_ms2.py', 'ManiSkillPointNet', 'ManiSkill PointNet'),
    'encoders.PointPatchTransformer': ('point_patch_transformer.py', 'PointPatchTransformer', 'PointPatchTransformer'),
    'encoders.ppt.PointPatchTransformer': (
        'point_patch_transformer.py',
        'PointPatchTransformer',
        'PointPatchTransformer',
    ),
    'encoders.PointNet2': ('pointnet2.py', 'PointNet2', 'PointNet++'),
    'encoders.pointnet2.PointNet2': ('pointnet2.py', 'PointNet2', 'PointNet++'),
    'encoders.DGCNN': ('dgcnn.py', 'DGCNN', 'DGCNN'),
    'encoders.dgcnn.DGCNN': ('dgcnn.py', 'DGCNN', 'DGCNN'),
}


def _pcd_encoder_class(target):
    """A 3D LeWM encoder class (PCD_ENCODERS) loaded from its file, without importing the encoders package (its
    __init__ imports encoders.vit -> stable_pretraining, which fails under the timm 0.6.11 CortexBench pins)."""
    import importlib.util
    fname, cls, _ = PCD_ENCODERS[target]
    path = Path(__file__).resolve().parents[2] / 'models' / 'encoders' / fname  # jepa4r/models/encoders
    spec = importlib.util.spec_from_file_location('lewm_' + fname[:-len('.py')], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def _dexart_pointnet_class():
    """encoders.DexArtPointNet (see _pcd_encoder_class)."""
    return _pcd_encoder_class('encoders.DexArtPointNet')


def load_lewm_pcd(
    checkpoint_dir: str, epoch: int = 10, center=(0.0, 0.65, 0.2), scale: float = 4.0, use_rgb: bool = True
):
    """The 3D LeWM's encoder (the one its config.json names: DexArt PointNet, ManiSkill PointNet,
    PointPatchTransformer, PointNet++ or DGCNN) + projector: one (N, 6) point cloud -> the 192-d LeWM latent.
    center / scale / use_rgb must be the checkpoint's data.preprocessor (checked against its config.yaml if present).
    CortexBench's load_pretrained_model wraps transforms for images (PIL); run_bc3d.py replaces that wrapper for
    models whose metadata says input 'pcd'."""
    ckdir = Path(checkpoint_dir)
    cfg = json.loads((ckdir / 'config.json').read_text())
    target = cfg['encoder']['_target_']
    assert target in PCD_ENCODERS, target
    if (ckdir / 'config.yaml').exists():
        import yaml
        pre = yaml.safe_load((ckdir / 'config.yaml').read_text())['data']['preprocessor']
        assert [float(c) for c in pre['center']] == [float(c) for c in center] and float(pre['scale']) == float(scale) \
            and bool(pre['use_rgb']) == bool(use_rgb), f'preprocessor mismatch: checkpoint {pre}'
    encoder = _pcd_encoder_class(target)(**{k: v for k, v in cfg['encoder'].items() if k != '_target_'})
    projector = instantiate(_moved(cfg['projector']))  # module.MLP (jepa4r.models.heads)
    sd = torch.load(ckdir / f'weights_epoch_{epoch}.pt', map_location='cpu')
    encoder.load_state_dict({k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')})
    projector.load_state_dict({k[len('projector.'):]: v for k, v in sd.items() if k.startswith('projector.')})
    model = LeWMEmbedding(encoder, projector).eval()
    with torch.no_grad():
        dim = model(torch.zeros(2, 1200, 6 if use_rgb else 3)).shape[-1]
    metadata = {'algo': 'lewm', 'input': 'pcd', 'model': f'{PCD_ENCODERS[target][2]} (LeWM 3D encoder + projector)',
                'data': _train_data(ckdir),
                'checkpoint': str(ckdir / f'weights_epoch_{epoch}.pt')}
    return model, int(dim), PcdTransform(center, scale, use_rgb), metadata
