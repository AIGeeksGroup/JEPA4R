"""Reuse the original asset tests with the scoped Python import adapter."""
import hashlib
import json
import os
from pathlib import Path
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    addon = json.loads((root / "RUNTIME_ADDENDUM.json").read_text())
    for name, h in addon["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h
    import encoder
    import encoder_py312
    import verify_asset
    old = encoder.FrozenPLDM
    encoder.FrozenPLDM = encoder_py312.FrozenPLDM
    try:
        asset_report = root / "asset_verification.tmp.json"
        assert not asset_report.exists()
        old_argv = sys.argv
        sys.argv = [str(root / "verify_asset.py"), "--root", str(root), "--report", str(asset_report)]
        try:
            verify_asset.main()
        finally:
            sys.argv = old_argv
    finally:
        encoder.FrozenPLDM = old
    import torch
    torch.set_num_threads(1)
    model = encoder_py312.FrozenPLDM(root)
    pixels = torch.arange(3 * 224 * 224, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(1, 3, 224, 224)
    result = model(pixels)
    expected = json.loads((root / "source_cpu_reference.json").read_text())
    reference = torch.tensor(expected["features"], dtype=torch.float32)
    difference = (result - reference).abs().max().item()
    if not torch.equal(result, reference):
        raise ValueError(f"source/target CPU encoding differs: {difference}")
    from pldm.models.jepa import JEPAConfig
    first, second = JEPAConfig(), JEPAConfig()
    assert first.backbone is second.backbone and first.predictor is second.predictor
    report = dict(actual_exit_code=0, source_target_CPU_encoding_bitexact=True,
        max_abs_difference=difference, Python310_shared_default_identity_preserved=True,
        original_producer_sources_unchanged=True, GPU_used=False)
    asset_result = json.loads(asset_report.read_text())
    asset_report.unlink()
    report["asset_verification"] = asset_result
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
