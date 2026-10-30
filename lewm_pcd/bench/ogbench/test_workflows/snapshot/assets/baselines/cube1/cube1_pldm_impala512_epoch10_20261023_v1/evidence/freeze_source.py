"""Write one deterministic source freeze for the thin SingleCube PLDM case."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "outputs",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--data-contract", required=True, type=Path)
    parser.add_argument("--reference-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    data_root = args.data_root.resolve()
    data_contract = args.data_contract.resolve()
    reference = args.reference_freeze.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing source freeze: {output}")

    files = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(source)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files[str(relative)] = sha256(path)

    expected_contract = json.loads(data_contract.read_text(encoding="utf8"))
    expected_manifest = data_root / "data_views" / "training_view_index.json"
    if sha256(expected_manifest) != expected_contract["manifest_sha256"]:
        raise ValueError("data manifest does not match sealed contract")

    payload = {
        "schema": "single_cube_pldm_rgb_source_freeze_v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case_root": str(source.parent),
        "source_root": str(source),
        "data_root": str(data_root),
        "reference_source_freeze": {
            "path": str(reference),
            "sha256": sha256(reference),
        },
        "data_contract": {
            "path": str(data_contract),
            "sha256": sha256(data_contract),
        },
        "files": files,
    }
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(output)
    print(
        json.dumps(
            {
                "actual_exit_code": 0,
                "path": str(output),
                "sha256": sha256(output),
                "files": len(files),
                "data_root": str(data_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
