"""Prepare OGBench's public play data; never rewrite an existing source."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    temporary.replace(path)


def build_plan(cube_count, output_root):
    if type(cube_count) is not int or cube_count not in (1, 2, 3, 4):
        raise ValueError("cube_count must be 1..4")
    sources = HERE.parents[2] / "configs" / "env" / "cube" / "official_sources.json"
    source = json.loads(sources.read_text(encoding="utf8"))[str(cube_count)]
    return dict(
        schema="cube_official_data_plan_v1",
        cube_count=cube_count,
        output_root=str(Path(output_root).resolve()),
        files=source["files"],
        source_config=source["source_config"],
        executed=False,
    )


def publish(partial, target):
    """Same-directory atomic no-replace publication of already verified bytes."""
    os.link(partial, target)
    Path(partial).unlink()


def download(item, directory):
    """Resume only a matching partial range and publish only a complete download."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = item["name"]
    if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
        raise ValueError("dataset name must be a simple file name")
    target = directory / name
    if target.exists():
        if target.stat().st_size != item["bytes"]:
            raise ValueError("existing source size differs: " + str(target))
        return dict(path=str(target.resolve()), sha256=sha256(target), reused=True)
    partial = directory / (name + ".part")
    metadata = directory / (name + ".download.json")
    status = directory / (name + ".status.json")
    if partial.exists():
        if not metadata.is_file() or json.loads(metadata.read_text()) != item:
            raise ValueError("partial download has another source binding")
    else:
        save(metadata, item)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["bytes"]:
        raise ValueError("partial download is larger than its source")
    save(status, dict(stage="downloading", completed_bytes=offset, total_bytes=item["bytes"], actual_exit_code=None))
    try:
        if offset < item["bytes"]:
            request = urllib.request.Request(
                item["url"], headers={"Range": "bytes=" + str(offset) + "-"} if offset else {}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith("bytes " + str(offset) + "-")
                ):
                    raise ValueError("server did not honor the partial range")
                with partial.open("ab" if partial.exists() else "xb") as output:
                    last_report = offset
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if offset - last_report >= 16 << 20:
                            output.flush()
                            save(
                                status,
                                dict(
                                    stage="downloading",
                                    completed_bytes=offset,
                                    total_bytes=item["bytes"],
                                    actual_exit_code=None,
                                ),
                            )
                            last_report = offset
        if partial.stat().st_size != item["bytes"]:
            raise ValueError("download size differs from the source")
        publish(partial, target)
        result = dict(path=str(target.resolve()), sha256=sha256(target), bytes=item["bytes"], reused=False)
        save(
            status,
            dict(
                stage="source_bytes_verified",
                completed_bytes=item["bytes"],
                total_bytes=item["bytes"],
                actual_exit_code=0,
            ),
        )
        return result
    except BaseException as error:
        save(
            status,
            dict(
                stage="download_failed",
                completed_bytes=partial.stat().st_size if partial.exists() else 0,
                total_bytes=item["bytes"],
                actual_exit_code=1,
                error=repr(error),
            ),
        )
        raise


def audit_npz(path, cube_count, *, expected_episodes):
    """Check the downloaded arrays and their terminal rows as they are; nothing is derived or replaced."""
    import numpy as np

    rows = expected_episodes * 1001
    arrays = {}
    with np.load(path, allow_pickle=False) as source:
        for key in ("observations", "actions", "terminals", "qpos", "qvel"):
            value = source[key]
            if len(value) != rows or not np.isfinite(value).all():
                raise ValueError("source row count/nonfinite data: " + key)
            arrays[key] = dict(shape=list(value.shape), dtype=str(value.dtype))
            if key == "observations" and value.shape != (rows, 19 + 9 * cube_count):
                raise ValueError("official observation dimensions differ")
            if key == "actions" and value.shape != (rows, 5):
                raise ValueError("official action dimensions differ")
        terminals = source["terminals"]
        if not np.isin(terminals, (0, 1)).all() or not np.array_equal(
            np.flatnonzero(terminals), np.arange(1000, rows, 1001)
        ):
            raise ValueError("official true terminal layout differs")
    return dict(
        raw_rows=rows,
        transition_rows=expected_episodes * 1000,
        episodes=expected_episodes,
        arrays=arrays,
        true_terminal_verified=True,
    )


def prepare(plan):
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "OFFICIAL_DATA.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["cube_count"] != plan["cube_count"]:
            raise ValueError("existing official-data receipt belongs to another Cube")
        for split in ("train", "val"):
            if sha256(previous[split]["path"]) != previous[split]["sha256"]:
                raise ValueError("existing official dataset checksum changed")
        return receipt
    result = dict(
        schema="cube_official_data_binding_v1", cube_count=plan["cube_count"], source_config=plan["source_config"]
    )
    for split, item in plan["files"].items():
        result[split] = download(item, output)
        result[split]["audit"] = audit_npz(
            result[split]["path"], plan["cube_count"], expected_episodes=item["episodes"]
        )
    result.update(
        actual_exit_code=0,
        stage="source_bytes_and_schema_verified",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save(receipt, result)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-count", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.cube_count, args.output)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        print(prepare(plan), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
