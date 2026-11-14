"""The encoder (Impala-512) of a trained Cube PLDM run, for the policy stages, built from the PLDM authors' modules
unchanged.

The JEPAConfig / objective factories are not imported: their predictor and optimizer weights are not needed to
encode single frames. No model code is rewritten (also not for Python 3.11+); the authors' Impala class runs the
whole forward pass."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

FAMILY, DIM = "pldm", 512
PREFIX = "core.backbone."
IMPORT_FILES = (
    "pldm/__init__.py",
    "pldm/configs.py",
    "pldm/models/__init__.py",
    "pldm/models/encoders/__init__.py",
    "pldm/models/encoders/enums.py",
    "pldm/models/encoders/base_class.py",
    "pldm/models/encoders/impala.py",
)


def encoder_source_bindings(source):
    source = Path(source).resolve()
    result = {}
    for relative in IMPORT_FILES:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError("Missing original PLDM source: " + str(path))
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def select_backbone_state(state, expected_shapes):
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError("A native bare PLDM state dictionary is required")
    selected = {key[len(PREFIX) :]: value for key, value in state.items() if key.startswith(PREFIX)}
    if not selected or selected.keys() != expected_shapes.keys():
        raise ValueError("PLDM original backbone keys differ; full Lightning/ViT checkpoints are not accepted")
    for key, value in selected.items():
        if tuple(getattr(value, "shape", ())) != tuple(expected_shapes[key]):
            raise ValueError("PLDM original backbone tensor shape differs: " + key)
    return selected


def load_frozen(checkpoint, device="cuda"):
    import torch

    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf8"))
    if (
        config.get("_target_") != "jepa4r.baselines.cube.pldm.method.build_model"
        or config.get("pretrained") is not False
    ):
        raise ValueError("An actual Cube Impala factory config is required")
    source = Path(config["official_source"]).resolve()
    encoder_source_bindings(source)
    sys.path.insert(0, str(source))
    package = importlib.import_module("pldm")
    if Path(package.__file__).resolve() != source / "pldm/__init__.py":
        raise ValueError("Conflicting PLDM package import")
    module = importlib.import_module("pldm.models.encoders.impala")
    if Path(module.__file__).resolve() != source / "pldm/models/encoders/impala.py":
        raise ValueError("Conflicting original Impala import")
    # Exact constructor used by ImagePLDM.core.backbone; caller CPU RNG is
    # preserved and CUDA is not initialized by discarded factory weights.
    with torch.random.fork_rng(devices=[]):
        model = module.ImpalaEncoder(input_channels=3, final_ln=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected = select_backbone_state(state, {key: tuple(value.shape) for key, value in model.state_dict().items()})
    if any(
        not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
        for v in selected.values()
    ):
        raise ValueError("Finite original FP32 Impala backbone weights required")
    model.load_state_dict(selected, strict=True)
    return model.to(device).eval().requires_grad_(False)


def encode(model, pixels):
    import torch

    if pixels.dtype != torch.float32 or tuple(pixels.shape[1:]) != (3, 65, 65):
        raise ValueError("Original Impala requires FP32 BCHW RGB65")
    with torch.inference_mode():
        result = model(pixels).encodings
    if result.shape != (len(pixels), 512) or result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise ValueError("Original Impala512 output shape/dtype/finite failed")
    return result


def encode_uint8(model, pixels, *, preprocess_device="cpu"):
    import torch
    from torchvision.transforms import v2

    if pixels.dtype != torch.uint8 or pixels.ndim != 4:
        raise ValueError("Real uint8 RGB224 frames required")
    if tuple(pixels.shape[1:]) == (224, 224, 3):
        pixels = pixels.permute(0, 3, 1, 2)
    if tuple(pixels.shape[1:]) != (3, 224, 224) or preprocess_device != "cpu":
        raise ValueError("Explicit native CPU preprocessing and full RGB224 required")
    # The same operations, in the same order, as PLDMPreprocess (training).
    pixels = pixels.contiguous().to("cpu")
    prepared = v2.Resize((65, 65), antialias=True)(pixels.float() / 255.0)
    return encode(model, prepared.to(next(model.parameters()).device))
