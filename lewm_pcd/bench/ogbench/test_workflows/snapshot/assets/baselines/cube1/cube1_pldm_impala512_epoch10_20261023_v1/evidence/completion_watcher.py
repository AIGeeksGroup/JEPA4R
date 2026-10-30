"""Wait for the main pipeline and preserve a completion/evidence archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    case = args.case_root.resolve()
    watch_root = case / "run_state" / "completion_watch"
    watch_root.mkdir(parents=True, exist_ok=False)
    status_path = watch_root / "status.json"
    status = {
        "schema": "single_cube_pldm_completion_watch_v1",
        "stage": "waiting_pipeline",
        "started_at": now(),
        "updated_at": now(),
        "pipeline_pid": args.pid,
    }
    atomic_json(status_path, status)

    while Path(f"/proc/{args.pid}").exists():
        status["updated_at"] = now()
        atomic_json(status_path, status)
        time.sleep(max(10, args.interval))

    pipeline_path = case / "run_state" / "pipeline" / "status.json"
    pipeline = (
        json.loads(pipeline_path.read_text(encoding="utf8"))
        if pipeline_path.exists()
        else {}
    )
    final_path = case / "FINAL_ACCEPT.json"
    final = (
        json.loads(final_path.read_text(encoding="utf8"))
        if final_path.exists()
        else {}
    )
    persistent = Path(
        "/autodl-fs/data/lewm/single_cube_pldm_rgb10ep_10225090_20261022_v1"
    )

    archive = case / "single_cube_pldm_evidence.tar.gz"
    if not archive.exists():
        names = [
            "source",
            "data_views",
            "run_state",
            "stages",
            "reference",
            "reference_originals",
            "FINAL_ACCEPT.json",
            "run_state_cpu_contract_status.json",
            "run_state_cpu_tests.log",
            "run_state_cpu_tests.exit",
        ]
        with tarfile.open(archive, "w:gz") as tar:
            for name in names:
                path = case / name
                if path.exists():
                    tar.add(path, arcname=name)

    result = {
        "schema": "single_cube_pldm_completion_watch_result_v1",
        "finished_at": now(),
        "pipeline_status": pipeline,
        "final_accept_exists": final_path.is_file(),
        "final_accept_exit_code": final.get("actual_exit_code"),
        "final_accept_sha256": digest(final_path) if final_path.is_file() else None,
        "persistent_root_exists": persistent.exists(),
        "persistent_root": str(persistent),
        "evidence_archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
        },
    }
    atomic_json(watch_root / "result.json", result)
    status.update(
        stage=(
            "training_and_accept_completed"
            if final.get("actual_exit_code") == 0
            else "pipeline_ended_without_final_accept"
        ),
        finished_at=result["finished_at"],
        updated_at=result["finished_at"],
    )
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
