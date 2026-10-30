"""The same accepted PLDM512 factory and preprocessing for cache and Native."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

FAMILY,DIM='pldm',512
ASSET_ID='cube3_pldm_impala512_epoch10_20261021_v1'
MANIFEST_SHA='96e2805301373e09cba689358e97673993af6b0a1ee4f3236b7f1ba0aa64d38a'
ADDENDUM_SHA='c15741c0c3969c4e29d0f2cffa3820ee5dc33bb6205b895ef7375fea133f52be'
CHECKPOINT_SHA='0ed6b328afcecf8858af3d46061bed1026a9c75ab7472a855496c907ba495b67'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_modules(root):
    root=Path(root).resolve();assert root.name==ASSET_ID
    assert sha(root/'MANIFEST.json')==MANIFEST_SHA and sha(root/'RUNTIME_ADDENDUM.json')==ADDENDUM_SHA
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert manifest['asset_id']==ASSET_ID and manifest['embedding_dim']==512 and manifest['epochs']==10
    assert manifest['members']['weights/weights_epoch_10.pt']['sha256']==CHECKPOINT_SHA
    sys.path.insert(0,str(root))
    for name in ('encoder','py312_config_compat','encoder_py312'):
        value=importlib.import_module(name)
        assert Path(value.__file__).resolve()==root/(name+'.py'),'Wrong encoder module: '+name
    return sys.modules['encoder_py312']


def load_frozen(checkpoint,device='cuda'):
    checkpoint=Path(checkpoint).resolve();root=checkpoint.parent.parent
    assert checkpoint==root/'weights/weights_epoch_10.pt' and sha(checkpoint)==CHECKPOINT_SHA
    model=asset_modules(root).FrozenPLDM(root).to(device).eval().requires_grad_(False)
    assert len(model.model.state_dict())==44
    model.obs_key='pixels'
    return model


def as_bchw(pixels):
    import torch
    if pixels.dtype!=torch.uint8 or pixels.ndim!=4:raise ValueError('uint8 batch of RGB224 frames required')
    if tuple(pixels.shape[1:])==(224,224,3):pixels=pixels.permute(0,3,1,2)
    if tuple(pixels.shape[1:])!=(3,224,224):raise ValueError('RGB224 full-FOV required')
    return pixels.contiguous()  # Native Impala uses view, so preserve standard NCHW layout.


def preprocess(batch,modality='2d'):
    if modality!='2d':raise ValueError('PLDM RGB only')
    module=sys.modules.get('encoder')
    if module is None:raise RuntimeError('Load the pinned PLDM factory first')
    result=dict(batch);result['pixels']=module.FrozenPLDM.preprocess(as_bchw(batch['pixels']))
    return result


def encode(model,inputs):
    import torch
    if inputs.dtype!=torch.float32 or tuple(inputs.shape[1:])!=(3,65,65):raise ValueError('FP32 PLDM RGB65 required')
    with torch.inference_mode():result=model.model.encode(inputs)
    if result.shape!=(len(inputs),512) or result.dtype!=torch.float32 or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid frozen PLDM512 feature output')
    return result


def encode_uint8(model,pixels,*,preprocess_device):
    if preprocess_device not in ('cpu','cuda'):raise ValueError('Use the cache-accepted preprocessing selection')
    pixels=as_bchw(pixels).to(preprocess_device)
    prepared=preprocess({'pixels':pixels})['pixels']
    return encode(model,prepared.to(next(model.parameters()).device))
