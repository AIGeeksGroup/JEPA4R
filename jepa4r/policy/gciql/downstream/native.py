"""The frozen encoder of the Cube policy stages (encode and evaluate): the run's world-model encoder (LeWM / JEPA4R
and PLDM through their adapters, LeJEPA through the BC baseline loader) on a uint8 RGB frame or an XYZRGB point
cloud."""

import importlib.util
from pathlib import Path
import sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_adapter(binding):
    module = _load("_cube_encoder_adapter", Path(binding["adapter"]))
    if (getattr(module, "FAMILY", None), getattr(module, "DIM", None)) != (binding["family"], binding["latent_dim"]):
        raise ValueError("the encoder adapter's family / output width differ from the run's encoder")
    return module


class OnlineEncoder:
    def __init__(self, config):
        import torch

        self.config, self.binding = config, config["encoder_binding"]
        binding = self.binding
        checkpoint = Path(binding["checkpoint"])
        self.torch, self.calls = torch, 0
        self.device = binding.get("device", "cuda")
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the frozen encoder runs on CUDA")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        family = binding["family"]
        if family == "lejepa":
            from ...bc import jepa_baselines

            self.model, dim, self.transform, _ = jepa_baselines.load_lejepa(str(checkpoint.parent), binding["epoch"])
            if dim != binding["latent_dim"]:
                raise ValueError("the LeJEPA head's output width differs from the run's encoder")
            self.model = self.model.to(self.device).eval().requires_grad_(False)
        else:
            self.adapter = load_adapter(binding)
            self.model = self.adapter.load_frozen(checkpoint, device=self.device)
            if family == "jepa_wm":
                module = _load("_cube_wm_preprocess", Path(binding["preprocessing"]["source_path"]))
                self.img_preprocess = module.get_img_preprocessor("pixels", "pixels", 224)
                pre = binding["preprocessing"]
                self.pcd_preprocess = module.get_pcd_preprocessor(
                    center=pre["center"] if config["modality"] == "pcd" else (0.425, 0.0, 0.2),
                    scale=pre["scale"] if config["modality"] == "pcd" else 4.0,
                    use_rgb=True,
                )
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("the encoder must stay frozen in eval mode")

    def encode(self, value):
        import numpy as np

        binding, torch = self.binding, self.torch
        family = binding["family"]
        if self.config["modality"] == "rgb":
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError("the RGB input must be uint8 HWC 224x224")
            pixels = torch.from_numpy(np.ascontiguousarray(value[None]))
            if family == "pldm":
                features = self.adapter.encode_uint8(
                    self.model, pixels, preprocess_device=binding["preprocessing"]["device"]
                )
            elif family == "lejepa":
                from PIL import Image

                prepared = self.transform(Image.fromarray(value)).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    features = self.model(prepared)
            else:
                batch = self.img_preprocess({"pixels": pixels.permute(0, 3, 1, 2)})
                features = self.adapter.encode(self.model, batch["pixels"].to(self.device))
        else:
            if value.shape != (1200, 6) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("the point cloud must be finite FP32 XYZRGB [1200, 6]")
            xyz = torch.from_numpy(value[None, :, :3].copy())
            color = np.clip(np.rint(value[None, :, 3:]), 0, 255).astype(np.uint8)
            batch = self.pcd_preprocess({"pcd_xyz": xyz, "pcd_rgb": torch.from_numpy(color)})
            features = self.adapter.encode(self.model, batch["pcd"].to(self.device))
        latent = features[0].detach().cpu().numpy()
        if latent.shape != (binding["latent_dim"],) or latent.dtype != np.float32 or not np.isfinite(latent).all():
            raise ValueError("the encoder did not return a finite FP32 latent of its width")
        self.calls += 1
        return latent
