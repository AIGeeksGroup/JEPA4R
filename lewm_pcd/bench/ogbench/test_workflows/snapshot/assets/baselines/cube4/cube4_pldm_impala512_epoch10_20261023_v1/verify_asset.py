"""CPU-only strict restore and synthetic interface check, without data access."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, row in manifest["members"].items():
        path = root / name
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("payload hash differs: " + name)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(root))
    from encoder import FrozenPLDM
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    model = FrozenPLDM(root)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    assert not torch.cuda.is_initialized()
    bare = torch.load(root / "weights/weights_epoch_10.pt", map_location="cpu", weights_only=True)
    assert len(bare) == 44
    assert all(torch.equal(model.model.state_dict()[k], v) for k, v in bare.items())
    # Check frame adaptation against the unchanged training preprocessing.
    path = root / "source/shared_data.py"
    spec = importlib.util.spec_from_file_location("original_pldm_data", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    frames = torch.arange(16 * 3 * 224 * 224, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(1, 16, 3, 224, 224)
    expected = original.PLDMPreprocess()({"pixels": frames.clone(), "actions": torch.zeros(1, 15, 5, dtype=torch.float32)})["pixels"]
    actual = model.preprocess(frames.flatten(0, 1)).reshape_as(expected)
    assert torch.equal(expected, actual)
    pixels = frames[:, 0].contiguous()
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    first, second = model(pixels), model(pixels)
    assert tuple(first.shape) == (1, 512) and first.dtype == torch.float32
    assert bool(torch.isfinite(first).all()) and torch.equal(first, second)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(v, state_before[k]) for k, v in model.state_dict().items())
    assert not model.training and all(not x.requires_grad for x in model.parameters())
    assert not torch.cuda.is_initialized()
    report = dict(actual_exit_code=0, accepted=True, asset_id=manifest["asset_id"],
        payload_members_verified=len(manifest["members"]), strict_tensor_leaves=44,
        preprocessing_exact=True, caller_Python_NumPy_Torch_CPU_RNG_preserved=True,
        synthetic_encode_shape=list(first.shape), synthetic_encode_finite=True,
        repeated_encode_bitexact=True, model_unchanged=True, GPU_used=False,
        training_or_real_data_or_SR_performed=False,
        versions={n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "einops", "omegaconf", "timm")})
    with a.report.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
