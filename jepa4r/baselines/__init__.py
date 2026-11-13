"""2D baselines of Tab. 1 / Tab. 2 (supp. sec:supp-baselines). LeWM is not here: it is a ``jepa_wm`` method
trained with exactly the JEPA4R recipe (jepa4r/models/world_model.py).

trained by us on the paired images, 10 epochs, own recipes : pldm.py, lejepa.py
released weights, frozen                                    : visreg.py, vjepa21.py
never trained                                               : random_vit.py
Each module: load_frozen_encoder(cfg) -> FrozenEncoder (+ train(cfg, out_dir) for the trained ones).
The encoders are built as the integrated policy harnesses read them (jepa4r/policy/bc/jepa_baselines.py: CortexBench
and DQC; the same model, embedding dimension and image transforms).
"""


class FrozenImageEncoder:
    """FrozenEncoder (jepa4r/policy/frozen.py) around a (model, dim, transforms) triple of jepa_baselines: ``encode``
    takes frames already through ``transforms`` [B, 3, H, W] and returns the embedding [B, dim]."""

    obs = 'image'

    def __init__(self, model, latent_dim, transforms, device='cpu'):
        self.model, self.latent_dim, self.transforms, self.device = (
            model.to(device).eval(),
            int(latent_dim),
            transforms,
            device,
        )

    def encode(self, batch):
        import torch

        with torch.no_grad():
            return self.model(torch.as_tensor(batch).to(self.device).float())


def frozen(loaded, cfg):
    model, dim, transforms, _ = loaded
    return FrozenImageEncoder(model, dim, transforms, device=str(cfg.get('device', 'cpu')))
