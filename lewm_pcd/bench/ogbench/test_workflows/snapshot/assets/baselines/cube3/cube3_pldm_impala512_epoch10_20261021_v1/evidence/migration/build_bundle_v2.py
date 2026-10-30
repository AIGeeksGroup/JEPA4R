"""Build one closed final-model asset; never open data or periodic weights."""
import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tarfile

ASSET_ID = "cube3_pldm_impala512_epoch10_20261021_v1"


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return stream_sha(stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    control = root / "pldm_accept_migrate_20261021_v1"
    out = control / "bundle_v2"
    out.mkdir(exist_ok=False)
    read = lambda path: json.loads(Path(path).read_text())
    accepted = read(control / "acceptance.json")
    assert accepted["accepted"] and accepted["actual_exit_code"] == 0
    assert (control / "actual_exit_code").read_text().strip() == "0"
    assert accepted["global_step"] == 457030 and len(accepted["export_lineage"]) == 10
    formal = root / "outputs/pldm_native_prefetch_selected_continuation_v1"
    own = Path(accepted["own_run"])
    training = formal / "formal/pldm"
    sources = {}

    def add(path, name):
        path = Path(path)
        if name in sources or path.is_symlink() or not path.is_file():
            raise ValueError("missing/duplicate/symlink member: " + name)
        if path.stat().st_size > 40_000_000:
            raise ValueError("unexpectedly large member: " + name)
        sources[name] = path

    full = Path(accepted["full_checkpoint"]["path"])
    bare = Path(accepted["epoch_exports"][-1]["path"])
    assert sha(full) == accepted["full_checkpoint"]["sha256"]
    assert sha(bare) == accepted["epoch_exports"][-1]["sha256"]
    add(full, "weights/last.ckpt")
    add(bare, "weights/weights_epoch_10.pt")
    native = root / "native_prefetch_continuation_v1"
    freeze = read(native / "source_freeze_v1.json")
    for base, rows in ((native, freeze["files"]), (root, freeze["reference_files"])):
        for name, h in rows.items():
            path = base / name
            if sha(path) != h:
                raise ValueError("frozen source changed: " + str(path))
            add(path, "source/" + path.relative_to(root).as_posix())
    add(native / "source_freeze_v1.json", "source/native_prefetch_continuation_v1/source_freeze_v1.json")
    for path in sorted((root / "vendor/pldm/source/pldm").rglob("*.py")):
        add(path, "source/" + path.relative_to(root).as_posix())
    for path in sorted((root / "vendor/pldm").glob("*.json")):
        add(path, "source/" + path.relative_to(root).as_posix())
    # This target source snapshot omits upstream top-level docs. Optional exact
    # author archive documents are supplied separately by the local controller.
    for path in sorted((control / "author_docs").glob("*")):
        add(path, "reference/author/" + path.name)
    for name in ("encoder.py", "verify_asset.py"):
        add(control / name, name)
    add(control / "ASSET_README.md", "README.md")
    for name in ("accept_existing.py", "test_own_run.py", "build_bundle_v2.py", "helper_freeze.json"):
        add(control / name, "evidence/migration/" + name)
    for name in ("acceptance.json", "actual_argv.json", "actual_exit_code", "stdout.log", "started_at", "ended_at"):
        add(control / name, "evidence/" + name)
    for name in ("actual_exit_code", "actual_argv.json", "stdout.log", "resources.jsonl"):
        path = training / name
        if path.exists():
            add(path, "evidence/trainer/" + name)
    for name in ("actual_protocol.json", "resume_readback.json", "loader_resource_record.json", "status.json"):
        add(training / "result" / name, "evidence/trainer/result/" + name)
    for name in ("run_meta.json", "summary.json", "metrics.csv"):
        add(own / name, "evidence/native_run/" + name)
    old = formal / "acceptance_process/pldm"
    for name in ("actual_exit_code", "actual_argv.json", "stdout.log", "result/status.json"):
        path = old / name
        if path.exists():
            add(path, "evidence/original_failed_acceptance/" + name)
    add(root / "outputs/native_prefetch_boundary_epoch4_v1/preserved_epoch4/status.json", "evidence/parent_epoch4.json")
    add(root / "root_authorizations/native_prefetch_continuation_full_release_v1.json", "evidence/original_release.json")
    versions = {n: importlib.metadata.version(n) for n in (
        "torch", "torchvision", "numpy", "einops", "omegaconf", "hydra-core", "timm", "pytorch-lightning")}
    before = {name: dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                        sha256=sha(path), source=str(path)) for name, path in sources.items()}
    manifest = dict(asset_id=ASSET_ID, cube_count=3, modality="RGB", method="PLDM",
        backbone="Impala512", embedding_dim=512,
        epochs=10, global_step=457030, RNG_exact_continuation=False,
        upstream_training_data="custom TripleCube front600; 10000 train/1000 validation",
        source_versions=versions, members=before)
    # Match the actual frozen method's explicit source pin, not a generated tag.
    import ast
    tree = ast.parse((root / "methods/pldm/method.py").read_text())
    manifest["author_commit"] = next(ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AUTHOR_COMMIT" for t in n.targets))
    archive = out / (ASSET_ID + ".tar.gz")
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for name, path in sources.items():
            tar.add(path, arcname=name, recursive=False)
    for name, path in sources.items():
        row = before[name]
        if (path.stat().st_size, path.stat().st_mtime_ns, sha(path)) != (row["bytes"], row["mtime_ns"], row["sha256"]):
            raise ValueError("source changed during packaging: " + name)
    with gzip.open(archive, "rb") as f:
        while f.read(1024 * 1024):
            pass
    with tarfile.open(archive, "r:gz") as tar:
        assert len(tar.getmembers()) == len(sources) + 1
        for name, row in before.items():
            stream = tar.extractfile(name)
            if stream_sha(stream) != row["sha256"]:
                raise ValueError("archive member changed: " + name)
    receipt = dict(actual_exit_code=0, asset_id=ASSET_ID, archive=str(archive),
        bytes=archive.stat().st_size, sha256=sha(archive), members=len(sources) + 1,
        source_before_after_exact=True, all_member_SHA_verified=True, full_gzip_CRC_verified=True)
    (out / "MANIFEST.json").write_bytes(manifest_bytes)
    (out / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
