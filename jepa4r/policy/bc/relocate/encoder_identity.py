# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/encoder_identity.py (verbatim)
"""Read every registered encoder parameter/buffer, including nonpersistent ones."""
import hashlib
import json


def encoder_digest(model):
    import torch
    digest=hashlib.sha256()
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,value in sorted(items):
            header=json.dumps([kind,name,str(value.dtype),list(value.shape)],separators=(',',':')).encode()
            raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(header).to_bytes(8,'big'));digest.update(header)
            digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def assert_encoder_unchanged(model,expected):
    if encoder_digest(model)!=expected:raise ValueError('frozen encoder parameter or buffer changed')
