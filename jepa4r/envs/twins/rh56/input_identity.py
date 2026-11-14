"""Checks before training: the BC source files, and the RH56 data split against the batches that produced it."""

import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_bc_source(eai_root):
    path = Path(eai_root).resolve() / "cortexbench/mujoco_vc/visual_imitation/train_loop.py"
    if not path.is_file():
        raise FileNotFoundError("EAI BC train loop missing: " + str(path))
    return dict(path=str(path), sha256=sha(path))


def _tuple(row, index):
    return (int(row["seed"]), int(index), row["split"], row["initial_state_sha256"], int(row["rows"]))


def _payload(report, batch, name):
    rows = [row for row in report["files"] if Path(row["path"]).name == name]
    path = batch / name
    if (
        len(rows) != 1
        or not path.is_file()
        or path.stat().st_size != rows[0]["bytes"]
        or sha(path) != rows[0]["sha256"]
    ):
        raise ValueError("producer payload/HDF binding differs: " + name)
    return path


def _batch(batch, hdf_name, expected_acceptance=None):
    receipt = batch / "batch_acceptance.json"
    digest = sha(receipt)
    if expected_acceptance is not None and digest != expected_acceptance:
        raise ValueError("independent producer acceptance SHA differs")
    report = json.loads(receipt.read_text())
    if report.get("status") != "PASS_BATCH" or report.get("deep_data_check") is not True:
        raise ValueError("independent producer requires deep accepted payloads")
    _payload(report, batch, hdf_name)
    producer = json.loads(_payload(report, batch, "manifest.json").read_text())
    config = json.loads(_payload(report, batch, "config.json").read_text())
    state = json.loads(_payload(report, batch, "state.json").read_text())
    if state.get("phase") != "complete" or state.get("exit_code") != 0:
        raise ValueError("producer collection did not complete")
    provenance = producer["provenance"]
    config_hash = hashlib.sha256(
        json.dumps(config["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        config["provenance"] != provenance
        or config_hash != provenance["config_sha256"]
        or report.get("plan_sha256") != provenance.get("plan_sha256")
    ):
        raise ValueError("producer frozen config/plan binding differs")
    records = producer["episodes"]
    accepted = [row for row in report["records"] if row["split"] in ("train", "val", "test")]
    planned = [row for row in config["config"]["episodes"] if row["split"] in ("train", "val", "test")]
    if [_tuple(r, r["episode_index"]) for r in accepted] != [_tuple(r, i) for i, r in enumerate(records)] or [
        (r["seed"], r["split"], r["requested_coverage"]) for r in records
    ] != [(r["seed"], r["split"], r["coverage"]) for r in planned]:
        raise ValueError("producer record tuple/config partition differs")
    return records, dict(acceptance_sha256=digest, config_sha256=config_hash, plan_sha256=report.get("plan_sha256"))


def validate_episode_producers(dataset, manifest):
    """Check the dataset's episode split against the records of the collection batches that produced each episode;
    split labels alone are not trusted."""
    path = Path(dataset.h5_path).resolve()
    dataset_sha = sha(path)
    proof = []
    records = []
    if "datasets" in manifest:
        receipts = manifest.get("source_receipts")
        if not receipts or manifest.get("source_acceptance_checked") is not True:
            raise ValueError("publication is missing independent producer receipts")
        selected = [
            row for row in manifest["datasets"] if Path(row["path"]).name == path.name and row["sha256"] == dataset_sha
        ]
        if len(selected) != 1:
            raise ValueError("published dataset/HDF header SHA binding differs")
        selected = selected[0]
        offset = 0
        for source in selected["sources"]:
            actual = (path.parent / source["path"]).resolve()
            batch = actual.parent
            acceptance = sha(batch / "batch_acceptance.json")
            expected = [r["acceptance_sha256"] for r in receipts if r["acceptance_sha256"] == acceptance]
            if len(expected) != 1:
                raise ValueError("publication producer acceptance binding differs")
            original, binding = _batch(batch, actual.name, expected[0])
            if source["offset"] != offset or source["rows"] != sum(r["rows"] for r in original):
                raise ValueError("producer HDF source offset/rows differ")
            offset += source["rows"]
            records.extend(original)
            proof.append(binding)
        if offset != selected["rows"] or offset != sum(map(int, dataset.lengths)):
            raise ValueError("producer dataset header total rows differ")
    elif "provenance" in manifest:
        records, binding = _batch(path.parent, path.name)
        proof.append(binding)
    else:
        raise ValueError("dataset is missing independent producer publication contract")
    current = sorted(manifest["episodes"], key=lambda r: r["episode_index"])
    if [_tuple(r, r["episode_index"]) for r in current] != [_tuple(r, i) for i, r in enumerate(records)]:
        raise ValueError("global partition differs from independent producer tuples")
    if any(
        r.get("source_episode_index", original["episode_index"]) != original["episode_index"]
        for r, original in zip(current, records)
    ):
        raise ValueError("global/source producer episode index differs")
    if len(records) != len(dataset.lengths) or [int(r["rows"]) for r in records] != list(map(int, dataset.lengths)):
        raise ValueError("producer episode/HDF lengths differ")
    # Seeds of the standard plan lie in the 61M (train), 62M (val) and 63M (test) ranges: an episode
    # labelled with another split is rejected. Plans with other seeds (e.g. 6000...) pass as they are.
    ranges = {"train": (61000000, 62000000), "val": (62000000, 63000000), "test": (63000000, 64000000)}
    canonical = any(61000000 <= int(r["seed"]) < 64000000 for r in records)
    if canonical and any(not ranges[r["split"]][0] <= int(r["seed"]) < ranges[r["split"]][1] for r in records):
        raise ValueError("canonical producer seed namespace partition differs")
    return dict(
        dataset_sha256=dataset_sha,
        producer_bindings=proof,
        tuples=[list(_tuple(r, i)) for i, r in enumerate(records)],
        canonical_seed_namespaces=canonical,
    )
