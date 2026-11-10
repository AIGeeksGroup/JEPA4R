"""Released VISReg/V-JEPA2.1 full encoders and original full-FOV preprocessing."""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class RawRGBIdentity:
    def __call__(self, rows):
        return rows


class FrozenImageEncoder(torch.nn.Module):
    def __init__(self, encoder, spec):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.spec = spec
        self.output_dim = spec['output_dim']
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def preprocess_uint8(self, raw):
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (3, 224, 224) or raw.dtype != torch.uint8:
            raise ValueError('Expected real uint8 BCHW224 RGB')
        device = next(self.encoder.parameters()).device
        x = raw.to(device, non_blocking=True).float().div(255.)
        size = self.spec['native_size']
        x = resize(x, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        x = (x - mean) / std
        return x.unsqueeze(2) if self.spec['family'] == 'vjepa21_vitl1024' else x

    @torch.inference_mode()
    def encode_uint8(self, raw):
        device_type = next(self.encoder.parameters()).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = self.preprocess_uint8(raw)
            if self.spec['family'] == 'vjepa21_vitl1024':
                if self.spec.get('inference_precision') != 'fp16' or device_type != 'cuda':
                    raise ValueError('V-JEPA2.1 paper route requires explicit CUDA FP16 full forward')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    tokens = self.encoder(x)
                if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (len(raw), 576, 1024):
                    raise ValueError('Expected last-normalized image patch tokens B576x1024')
                features = tokens.mean(dim=1).float()
            else:
                tokens = self.encoder.forward_features(x)
                if tuple(tokens.shape) != (len(raw), 197, 768):
                    raise ValueError('Expected normalized VISReg tokens B197x768')
                features = tokens[:, 0, :]
        if features.dtype != torch.float32 or tuple(features.shape) != (len(raw), self.output_dim) or not torch.isfinite(features).all():
            raise ValueError('Expected finite FP32 BD backbone features')
        return features

    def forward(self, raw):
        return SimpleNamespace(logits=self.encode_uint8(raw))

    def snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def check_frozen(self, before):
        now = self.state_dict()
        if set(now) != set(before) or any(v.training for v in self.modules()) or any(v.requires_grad or v.grad is not None for v in self.parameters()):
            raise ValueError('Frozen encoder state/mode violation')
        for k, v in now.items():
            current = v.detach().cpu()
            if current.dtype != before[k].dtype or not torch.equal(current, before[k]):
                raise ValueError('Frozen parameter/buffer changed: ' + k)


def load_encoder(spec, device='cuda'):
    """Load explicit local assets; never invoke hub or an automatic download."""
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    spec = dict(spec)
    if spec['family'] not in ('visreg_vitb768', 'vjepa21_vitl1024'):
        raise ValueError('Unknown encoder family')
    checkpoint = Path(spec['checkpoint'])
    if sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA differs')
    if spec['family'] == 'vjepa21_vitl1024':
        root = Path(spec['source_root'])
        if spec['source_commit'] != '204698b45b3712590f06245fbfba32d3be539812':
            raise ValueError('Official source commit differs')
        for relative, digest in spec['source_files_sha256'].items():
            if sha(root / relative) != digest:
                raise ValueError('Official source SHA differs: ' + relative)
        sys.path.insert(0, str(root))
        module = importlib.import_module(spec['encoder_module'])
        backbones = importlib.import_module('src.hub.backbones')
        if Path(module.__file__).resolve() != (root / 'app/vjepa_2_1/models/vision_transformer.py').resolve() or Path(backbones.__file__).resolve() != (root / 'src/hub/backbones.py').resolve():
            raise ValueError('Official source namespace collision')
        model = getattr(module, spec['encoder_constructor'])(**spec['constructor_kwargs'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = backbones._clean_backbone_key(payload['ema_encoder'].copy())
    else:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, dynamic_img_size=True)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = {k: v for k, v in payload.items() if not k.startswith('proj.')}
    model.load_state_dict(state, strict=True)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError('Finite original FP32 weights required')
    return FrozenImageEncoder(model, spec).to(device).eval()


def encode_uint8(model, pixels):
    """Native convenience entrypoint: uint8 BHWC224 -> same backbone path."""
    pixels = torch.as_tensor(pixels)
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (224, 224, 3) or pixels.dtype != torch.uint8:
        raise ValueError('Expected uint8 BHWC224 RGB')
    return model.encode_uint8(pixels.permute(0, 3, 1, 2))
