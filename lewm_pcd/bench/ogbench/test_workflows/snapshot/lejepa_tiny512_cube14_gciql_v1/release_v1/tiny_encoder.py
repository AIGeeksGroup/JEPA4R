"""Tiny512 raw RGB boundary over each exact original load_lejepa implementation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

FAMILY = 'lejepa_tiny512'
SPEC_DIGESTS = {1: 'de42dc548ab0931df8334569bcf041bf9d0a868fc19da982661bc76732ed6351', 2: 'd077f72f934a2ff4f58b7242033ccf8fb8c784bd5df38315ddbce6c2a63319c6', 3: '17c7c3ef442edf72c2fa0632a792c9f79d2c461afe0bbbe2ff643b2790bfb17b', 4: '3a3cf8142de491a629f8e80ba03d8b51c93e66d14e65b5adfd80ae30dad8aa8d'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def validate_spec(spec,verify_files=False):
    if isinstance(spec,(str,Path)):
        spec=json.loads(Path(spec).read_text())
    assert isinstance(spec,dict)
    n=spec.get('cube_count')
    assert type(n) is int and n in SPEC_DIGESTS
    assert spec_digest(spec)==SPEC_DIGESTS[n],'Exact per-Cube Tiny spec differs'
    assert spec['family']==FAMILY and spec['output_dim']==512 and spec['feature']=='cls192_head512'
    assert spec['inference_precision']=='fp32' and not spec['feature_L2_normalization'] and not spec['training_projector_used']
    if verify_files:
        checkpoint=Path(spec['checkpoint'])
        assert checkpoint.name=='weights_epoch_10.pt' and checkpoint.parent==Path(spec['checkpoint_dir'])
        assert checkpoint.stat().st_size==spec['checkpoint_bytes'] and sha(checkpoint)==spec['checkpoint_sha256']
        for path,digest in [(spec['original_loader'],spec['original_loader_sha256']),
                            (spec['original_config'],spec['original_config_sha256']),
                            (spec['asset_manifest_path'],spec['asset_manifest_sha256'])]:
            assert sha(path)==digest,path
    return spec


def make_spec(cube_count):
    assert type(cube_count) is int and cube_count in SPEC_DIGESTS
    return validate_spec(Path(__file__).parent/'specs'/f'cube{cube_count}.json')


def _precision(torch):
    assert torch.get_default_dtype()==torch.float32,'Original Tiny default FP32 required'
    assert torch.get_float32_matmul_precision()=='highest'
    assert torch.backends.cuda.matmul.allow_tf32 is False


def load_encoder(spec,device='cuda'):
    spec=validate_spec(spec,verify_files=True)
    import torch
    from PIL import Image
    _precision(torch)
    location=Path(spec['original_loader'])
    loaded=importlib.util.spec_from_file_location('_tiny_original_loader_cube'+str(spec['cube_count']),location)
    original=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(original)
    assert Path(original.__file__).resolve()==location.resolve()
    model,dim,transforms,metadata=original.load_lejepa(spec['checkpoint_dir'],epoch=10)
    assert dim==512 and len(model.state_dict())==200
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert all(not v.is_floating_point() or (v.dtype==torch.float32 and bool(torch.isfinite(v).all())) for v in model.state_dict().values())

    class FrozenTiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=model;self.transforms=transforms;self.spec=spec
            self.output_dim=512;self.original_metadata=metadata;self.eval()
        def train(self,mode=True):
            return super().train(False)
        def preprocess_uint8(self,raw):
            assert isinstance(raw,torch.Tensor) and raw.device.type=='cpu'
            assert raw.dtype==torch.uint8 and raw.ndim==4 and tuple(raw.shape[1:])==(3,224,224)
            images=raw.permute(0,2,3,1).contiguous().numpy()
            # Execute the original returned PIL transform once per image, without a second normalizer.
            result=torch.stack([self.transforms(Image.fromarray(image)) for image in images])
            assert result.dtype==torch.float32 and tuple(result.shape)==(len(raw),3,224,224)
            assert bool(torch.isfinite(result).all())
            return result
        @torch.inference_mode()
        def encode_uint8(self,raw):
            _precision(torch)
            assert not self.training and not any(p.requires_grad for p in self.parameters())
            assert all(p.dtype==torch.float32 for p in self.parameters())
            pixels=self.preprocess_uint8(raw).to(next(self.encoder.parameters()).device,non_blocking=True)
            assert pixels.dtype==torch.float32
            with torch.autocast(device_type=pixels.device.type,enabled=False):
                features=self.encoder(pixels)
            assert features.dtype==torch.float32 and tuple(features.shape)==(len(raw),512)
            assert bool(torch.isfinite(features).all())
            return features
        def forward(self,raw):
            return SimpleNamespace(logits=self.encode_uint8(raw))
    return FrozenTiny().to(device).eval().requires_grad_(False)


def encode_uint8(model,pixels):
    """Existing Native callback: CPU uint8 BHWC224 -> exact Tiny BCHW raw path."""
    import torch
    pixels=torch.as_tensor(pixels)
    assert pixels.device.type=='cpu' and pixels.dtype==torch.uint8 and pixels.ndim==4
    assert tuple(pixels.shape[1:])==(224,224,3)
    return model.encode_uint8(pixels.permute(0,3,1,2).contiguous())
