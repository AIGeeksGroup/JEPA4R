"""CPU audit v2: exact model keys plus the trainer's three SIGReg buffers."""
import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
EXPECTED = {
    "2d": ("995a99f4904d87e71424b2de7910e7e0cf79556de8c2cebd08ac582d70880190",
           "c11a7bbb27574ab03dd708b9c4c30b00e52b472d9168d2ca06a0c50bc433a6e4"),
    "3d": ("f391f047aa73fa08b2b4534e6018fde6a219cc3ae6ba4464b907e2b9f9138a46",
           "f0b911701f567879666f6bfdfb1479290117b92a5878493b2eb42f3ab3a8a365"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_tensors(state, torch):
    require(isinstance(state, Mapping) and bool(state), "state_dict must be nonempty mapping")
    for key, tensor in state.items():
        require(isinstance(key, str) and isinstance(tensor, torch.Tensor), f"non-tensor: {key}")
        require(tensor.device.type == "cpu" and tensor.layout == torch.strided, f"layout: {key}")
        require(not tensor.is_quantized, f"quantized tensor unsupported: {key}")
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), f"nonfinite: {key}")


def check_pair(root, mode, entries, torch):
    base = f"cube2/{mode}_epoch10/"
    members = [base + "weights_epoch_10.pt", base + "full/epoch=9-step=375780.ckpt"]
    files = []
    for member, expected in zip(members, EXPECTED[mode]):
        path = (root / member).resolve(strict=True)
        require(path.is_relative_to(root), "checkpoint escapes root")
        stat = path.stat()
        require(entries[member]["sha256"] == expected, "manifest differs from frozen source SHA")
        require(stat.st_size == entries[member]["bytes"] and digest(path) == expected, member)
        files.append({"member": member, "bytes": stat.st_size, "sha256": expected,
                      "mtime_ns_before": stat.st_mtime_ns})
    bare = torch.load(root / members[0], map_location="cpu", weights_only=True)
    full = torch.load(root / members[1], map_location="cpu", weights_only=False)
    check_tensors(bare, torch)
    require(isinstance(full, Mapping), "full checkpoint must be mapping")
    require(full["epoch"] == 9 and full["global_step"] == 375780, "epoch/global_step mismatch")
    state = full["state_dict"]
    check_tensors(state, torch)
    prefix = "model."
    auxiliary_keys = {"sigreg.t", "sigreg.phi", "sigreg.weights"}
    expected_keys = {prefix + key for key in bare} | auxiliary_keys
    require(set(state) == expected_keys, "trainer complete keyset differs from model plus exact SIGReg buffers")
    auxiliary = {}
    for key in sorted(auxiliary_keys):
        tensor = state[key]
        require(tensor.dtype == torch.float32 and tuple(tensor.shape) == (17,), f"SIGReg schema: {key}")
        auxiliary[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": True}
    for key, tensor in bare.items():
        other = state[prefix + key]
        require(tensor.shape == other.shape and tensor.dtype == other.dtype, f"shape/dtype: {key}")
        left = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        right = other.detach().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(left, right), f"tensor byte mismatch: {key}")
    optimizers = full["optimizer_states"]
    require(isinstance(optimizers, list) and bool(optimizers), "missing optimizer states")
    counts = []
    for optimizer in optimizers:
        states = optimizer["state"]
        require(bool(states), "empty Adam state")
        for value in states.values():
            require({"step", "exp_avg", "exp_avg_sq"} <= set(value), "not complete Adam state")
            step = value["step"]
            if isinstance(step, torch.Tensor):
                require(step.device.type == "cpu" and step.numel() == 1, "invalid Adam step tensor")
                step = step.item()
            require(not isinstance(step, bool) and step == 375780, "Adam step mismatch")
        counts.append(len(states))
    for info in files:
        stat = (root / info["member"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (info["bytes"], info["mtime_ns_before"]), "file changed")
    return {"files": files, "tensor_count": len(bare), "elements": sum(t.numel() for t in bare.values()),
            "dtypes": sorted({str(t.dtype) for t in bare.values()}), "all_tensors_finite": True,
            "full_state_prefix": prefix, "complete_trainer_keyset_exact": True,
            "model_keyset_and_tensor_bytes_exact": True, "SIGReg_buffers": auxiliary,
            "full_epoch_zero_based": 9, "global_step": 375780, "Adam_state_counts": counts,
            "all_Adam_state_steps": 375780, "CPU_tensor_integrity_passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing cube2/origin_manifest.json")
    parser.add_argument("--output", required=True, help="New JSON receipt; existing file is rejected")
    args = parser.parse_args()
    with Path(args.output).open("x", encoding="utf-8") as output:
        report = {"started": time.time(), "actual_exit_code": 1, "modes": {},
                  "GPU_or_model_forward_performed": False, "optimizer_updates_performed": False,
                  "load_policy": {"bare_weights_only": True, "full_weights_only": False,
                                  "full_trust_basis": "own source checkpoints with four frozen SHA256 values"},
                  "old_2d_GPU_strict_reload_failure": "preserved; CPU audit does not replace GPU reload"}
        try:
            import torch
            torch.set_num_threads(1)
            root = Path(args.root).resolve(strict=True)
            manifest_path = root / "cube2/origin_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(manifest["cube_count"] == 2 and manifest["source_host"] == "101632", "wrong provenance")
            entries = {item["member"]: item for item in manifest["source_file_manifest"]}
            report.update(root=str(root), torch_version=torch.__version__, manifest_sha256=digest(manifest_path))
            for mode in ("2d", "3d"):
                report["modes"][mode] = check_pair(root, mode, entries, torch)
                gc.collect()
            report.update(actual_exit_code=0, stage="CPU_checkpoint_integrity_verified")
        except Exception:
            report["error"] = traceback.format_exc()
        report["finished"] = time.time()
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return report["actual_exit_code"]


if __name__ == "__main__":
    sys.exit(main())
