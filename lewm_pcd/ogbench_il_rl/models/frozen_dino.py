"""Original strict local DINOv2-small loader and frozen raw CLS384."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

SOURCE_FILES={'config.json':'1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
 'model.safetensors':'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',
 'preprocessor_config.json':'14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def validate_source(folder):
    folder=Path(folder).resolve()
    if any(sha(folder/name)!=digest for name,digest in SOURCE_FILES.items()):raise ValueError('Local DINO source files changed')
    config=json.loads((folder/'config.json').read_text())
    if config['hidden_size']!=384 or config['patch_size']!=14 or config['model_type']!='dinov2':raise ValueError('Real DINO-small CLS384 required')
    return folder


class FrozenDinoCLS(nn.Module):
    def __init__(self,model_folder,defer_cuda_checks=False,encoder_tf32=False):
        super().__init__()
        from transformers import AutoModel
        folder=validate_source(model_folder)
        self.model,info=AutoModel.from_pretrained(str(folder),local_files_only=True,output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):raise ValueError('DINO full pretrained state must load strictly')
        if self.model.config.hidden_size!=384:raise ValueError('Wrong local DINO hidden dimension')
        self.defer_cuda_checks=bool(defer_cuda_checks);self.encoder_tf32=bool(encoder_tf32);self.finite_ok=None
        self.requires_grad_(False);self.train(False)
    def train(self,mode=True):
        super().train(False);return self
    def check_finite(self,value):
        flag=torch.isfinite(value).all()
        if self.defer_cuda_checks:self.finite_ok=flag if self.finite_ok is None else self.finite_ok & flag
        elif not flag:raise ValueError('Nonfinite DINO input/CLS')
    def forward(self,inputs):
        if inputs.ndim!=4 or inputs.shape[1:]!=(3,224,224):raise ValueError('Actual normalized full-FOV224 BCHW input required')
        self.check_finite(inputs);prior=torch.backends.cuda.matmul.allow_tf32
        try:
            if self.encoder_tf32:torch.backends.cuda.matmul.allow_tf32=True
            with torch.no_grad():latent=self.model(pixel_values=inputs.float()).last_hidden_state[:,0,:]
        finally:
            if self.encoder_tf32:torch.backends.cuda.matmul.allow_tf32=prior
        if latent.shape!=(len(inputs),384):raise ValueError('True normalized DINO CLS384 required')
        self.check_finite(latent)
        return SimpleNamespace(logits=latent)
