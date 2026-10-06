"""Original frozen encoder/projector adapter, no action head."""
from types import SimpleNamespace
import torch
from torch import nn

class LeWMFrozenEncoder(nn.Module):
    """Expose encoder+projector output through GCRL's existing global-logits path.

    RGB and point-cloud inputs retain their native flattened batch shape. This
    adapter runs the real encoder and projector on every call; it caches no
    features and constructs no synthetic patch or CLS tokens.
    """

    def __init__(self, source_checkpoint=None, world_model=None, latent_dim=192):
        super().__init__()
        if (source_checkpoint is None) == (world_model is None):
            raise ValueError('Provide exactly one source_checkpoint or world_model')
        if world_model is None:
            from stable_worldmodel.wm.utils import load_pretrained
            world_model = load_pretrained(str(source_checkpoint), cache_dir='/')
        self.encoder = world_model.encoder
        self.projector = world_model.projector
        self.config = SimpleNamespace(hidden_size=int(latent_dim))
        self.num_patches = 1  # One actual global latent, not image patch tokens.
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # GCRL.train() must not reactivate dropout or BN running-stat updates.
        super().train(False)
        return self

    def forward(self, inputs):
        with torch.no_grad():
            latent = self.projector(self.encoder(inputs.float()))
        if latent.ndim != 2 or latent.shape != (inputs.shape[0], self.config.hidden_size):
            raise ValueError(f'Expected real global latent (batch,{self.config.hidden_size}), got {tuple(latent.shape)}')
        if not torch.isfinite(latent).all():
            raise ValueError('LeWM encoder/projector produced nonfinite global latents')
        return SimpleNamespace(logits=latent)
