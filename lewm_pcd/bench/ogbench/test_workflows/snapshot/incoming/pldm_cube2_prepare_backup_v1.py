"""Create a source-bound Cube2 PLDM result archive without cache vectors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile
import time


CASE = Path("/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/pldm_cube2_reaudit_20261029_v1")
GROUP = "P2_cube2_pldm_reaudit_rgb"
OUT = CASE / "closed_backup_v1"
ARCHIVE = OUT / "pldm_cube2_reaudit_results.tar.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    run = CASE / "runs" / GROUP
    selected: set[Path] = set()

    for path in CASE.glob("*.json"):
        selected.add(path)
    for root in (
        CASE / "functional_v1",
        CASE / "acceptance",
        CASE / "gates" / GROUP,
        CASE / "native_sr_v1",
        CASE / "control/run/runtime",
        CASE / "control/run/logs",
    ):
        if root.exists():
            selected.update(path for path in root.rglob("*") if path.is_file())

    for name in (
        "PREPARED.json",
        "producer_plan.json",
        "encoder_adapter.py",
        "producer_bridge.py",
        "audit_bridge.py",
        "prepare_plan.py",
        "source_overlay.py",
    ):
        selected.add(CASE / "cache_v1" / name)
    for rel in (
        "cache/cache_manifest.json",
        "cache/cache_seal.json",
        "cache/protocol.json",
        "cache/status.json",
        "cache/preprocess_selection.json",
        "cache/batch1_alignment.json",
        "cache/pilot_receipt.json",
        "independent_audit/report.json",
        "independent_audit/actual_exit_code",
        "source_arrays/source_arrays.json",
        "source_arrays/data_acceptance.json",
        "source_arrays/stat-freeze.json",
    ):
        selected.add(CASE / "cache_v1" / rel)

    for rel in (
        "status.json",
        "resolved.json",
        "train.csv",
        "trainer_exit_code",
        "final_val_probe.npz",
        "final_state_fingerprint.json",
    ):
        selected.add(run / rel)
    selected.update(path for path in (run / "checkpoints").rglob("*") if path.is_file())

    files = sorted(path for path in selected if path.is_file())
    forbidden = {".pt", ".ckpt", ".h5", ".hdf5"}
    if any(path.suffix in forbidden for path in files):
        raise ValueError("unexpected model/data payload in selected backup")
    manifest = {
        "schema_version": 1,
        "actual_exit_code": 0,
        "created": time.time(),
        "scope": (
            "PLDM Cube2 independent rerun: source/config, 11 GCIQL checkpoints, "
            "CPU acceptance, cache metadata/audit, Native tiny/full step results, "
            "control exits/logs, and final records; excludes raw datasets and feature vectors"
        ),
        "case_root": str(CASE),
        "group_id": GROUP,
        "files": {
            str(path.relative_to(CASE)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    manifest_path = OUT / "MEMBER_MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")

    with tarfile.open(ARCHIVE, "x:gz", compresslevel=3) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(CASE)), recursive=False)
        archive.add(manifest_path, arcname=str(manifest_path.relative_to(CASE)))

    receipt = {
        "schema_version": 1,
        "actual_exit_code": 0,
        "completed": time.time(),
        "archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "member_manifest_sha256": sha256(manifest_path),
        "member_count": len(files) + 1,
        "source_bytes": sum(entry["bytes"] for entry in manifest["files"].values()),
    }
    receipt_path = OUT / "ARCHIVE_RECEIPT.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
