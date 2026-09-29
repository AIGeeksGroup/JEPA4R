"""vc_models entry points for the VC-1 paper's other baselines, which the public eai-vc release does not ship:
R3M (ResNet-50), VIP (ResNet-50) and CLIP (ViT-B/16). Features as the paper's appendix states for the MuJoCo suites:
a ResNet gives its last conv layer's global-average-pooled features (2048-d; VIP's 1024-d fc head is not used), a ViT
its CLS token (CLIP: the vision tower's pooler output = layer-normed CLS, 768-d). Weights are the models' official
releases (R3M / VIP: the pytorch.s3 copies the official loaders use; CLIP: openai/clip-vit-base-patch16 on the HF
hub), in $STABLEWM_HOME/checkpoints/cortexbench_baselines. Preprocessing is each model's own: ImageNet normalisation after Resize(256) +
CenterCrop(224) for the ResNets (as R3M / VIP and VC-1 do), CLIP's resize-224 + its normalisation for CLIP.
(MVP is not here: its official download links on berkeley.box.com return 404.)
"""
import hashlib
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from PIL import Image

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_vc1(name: str, checkpoint: str):
    """Load VC-1 with the official eai-vc model and preprocessing code."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'official VC-1 checkpoint is unavailable: {checkpoint}')

    from vc_models.models.vit.vit import (
        load_mae_encoder,
        vit_base_patch16,
        vit_large_patch16,
    )
    from vc_models.transforms import vit_transforms

    builders = {'vc1_vitb': vit_base_patch16, 'vc1_vitl': vit_large_patch16}
    if name not in builders:
        raise ValueError(f'unknown VC-1 model: {name}')
    model = builders[name](img_size=224, use_cls=True, drop_path_rate=0.0)
    model = load_mae_encoder(model, checkpoint_path=str(checkpoint)).eval()
    transforms = vit_transforms()
    with torch.no_grad():
        embedding_dim = int(model(transforms(Image.new('RGB', (256, 256))).unsqueeze(0)).shape[-1])
    return model, embedding_dim, transforms, {
        'algo': 'mae',
        'model': name,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': _sha256(checkpoint),
        'loader': 'official vc_models.models.vit.vit.load_mae_encoder',
    }


def load_resnet_baseline(name: str, checkpoint: str):
    """R3M / VIP: torchvision ResNet-50 with the checkpoint's 'module.convnet.*' weights, fc removed -> 2048-d."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'official {name} checkpoint is unavailable: {checkpoint}')
    ck = torch.load(checkpoint, map_location='cpu')
    sd = ck['r3m'] if name == 'r3m' else ck['vip']
    conv = {k[len('module.convnet.'):]: v for k, v in sd.items() if k.startswith('module.convnet.') and '.fc.' not in k and not k.startswith('module.convnet.fc')}
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Identity()
    missing, unexpected = model.load_state_dict(conv, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(*IMAGENET)])
    return model.eval(), 2048, transforms, {
        'algo': name,
        'model': 'resnet50',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': _sha256(checkpoint),
        'loader': 'JEPA4R official-format ResNet adapter',
    }


class _CLIPCLS(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, x):
        return self.vision(pixel_values=x).pooler_output


def load_clip(model_id: str, cache_dir: str):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')  # weights are in cache_dir; no network during runs
    from transformers import CLIPVisionModel
    vision = CLIPVisionModel.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
    transforms = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(*CLIP_NORM)])
    return _CLIPCLS(vision).eval(), vision.config.hidden_size, transforms, {'algo': 'clip', 'model': model_id}
