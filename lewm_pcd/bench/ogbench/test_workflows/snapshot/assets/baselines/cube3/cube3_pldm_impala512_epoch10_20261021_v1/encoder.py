"""Frozen 3Cube PLDM512 encoder; no training or environment creation."""
import importlib.util
from pathlib import Path
import random
import numpy as np
import torch
from torchvision.transforms import v2


class FrozenPLDM(torch.nn.Module):
    embedding_dim = 512

    def __init__(self, asset_root):
        super().__init__()
        root = Path(asset_root).resolve()
        path = root / "source/methods/pldm/method.py"
        spec = importlib.util.spec_from_file_location("cube3_pldm_method", path)
        method = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(method)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            # Factory initialization is discarded after strict load. Preserve the
            # caller's CPU RNG; do not initialize or seed CUDA in this loader.
            with torch.random.fork_rng(devices=[]):
                self.model = method.build_model(root / "source/vendor/pldm/source",
                                                pretrained=False)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        state = torch.load(root / "weights/weights_epoch_10.pt",
                           map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.eval()

    @staticmethod
    def preprocess(pixels):
        if pixels.dtype != torch.uint8 or pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 224, 224):
            raise ValueError("uint8 RGB [B,3,224,224] required")
        # Same operations/order as the producer's PLDMPreprocess, applied to
        # individual images; no duplicated temporal frames or ImageNet norm.
        return v2.Resize((65, 65), antialias=True)(pixels.float() / 255.0)

    @torch.inference_mode()
    def forward(self, pixels):
        return self.model.encode(self.preprocess(pixels))
