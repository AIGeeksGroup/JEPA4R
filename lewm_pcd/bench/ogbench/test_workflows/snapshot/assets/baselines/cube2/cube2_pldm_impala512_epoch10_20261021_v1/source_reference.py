"""One CPU image encoding with the unchanged trained factory/preprocess."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(a.root))
    from methods.pldm.method import build_model
    from shared_data import PLDMPreprocess
    accepted = json.loads((a.root / "pldm_accept_migrate_20261021_v1/acceptance.json").read_text())
    bare = torch.load(accepted["epoch_exports"][-1]["path"], map_location="cpu", weights_only=True)
    model = build_model(a.root / "vendor/pldm/source", pretrained=False).eval().requires_grad_(False)
    model.load_state_dict(bare, strict=True)
    frames = torch.arange(16 * 3 * 224 * 224, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(1, 16, 3, 224, 224)
    pixels = PLDMPreprocess()({"pixels": frames})["pixels"][:, 0].contiguous()
    with torch.inference_mode():
        features = model.encode(pixels)
    assert list(features.shape) == [1, 512] and bool(torch.isfinite(features).all())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in bare.items())
    assert not torch.cuda.is_initialized()
    report = dict(actual_exit_code=0, GPU_used=False, model_unchanged=True,
        raw_dataset_opened=False, synthetic_input="arange %251 RGB224, first frame of original16-frame preprocessing",
        shape=list(features.shape), features=features.tolist(), torch_version=torch.__version__,
        Python_version=sys.version, checkpoint_sha256=accepted["epoch_exports"][-1]["sha256"])
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
    print("source original factory CPU reference accepted")


if __name__ == "__main__":
    main()
