#!/usr/bin/env python3
"""Light copies of the official Relocate demonstrations (the 100 of relocate-v0.pickle, in their order) without their
images: <output>/demos/demo_<i>.pickle with every other field unchanged (the 1536-D cached observations and the 39-D
state_obs, the actions, rewards, initial states and env_infos), manifest.json listing them, and
rh56_official_task_init.json with each demonstration's object and target positions (the RH56 twin starts its BC
demonstrations from them). No simulator is imported.
usage: python jepa4r/data/demos/prepare_official_assets.py --source <relocate-v0.pickle> --output <new dir>"""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_assets(source, output):
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with source.open("rb") as stream:
        demos = pickle.load(stream)
    demo_dir = output / "demos"
    demo_dir.mkdir()
    records, initial_records = [], []
    for index, demo in enumerate(demos):
        path = demo_dir / f"demo_{index:04d}.pickle"
        with path.open("xb") as stream:
            stream.write(pickle.dumps({key: value for key, value in demo.items() if key != "images"}, protocol=4))
        records.append(dict(demo_index=index, relative_path=path.relative_to(output).as_posix()))
        initial = demo["init_state_dict"]
        initial_records.append(
            dict(
                demo_index=index,
                source_id=f"official-relocate-{index:03d}",
                **{key: np.asarray(initial[key], dtype=np.float64).tolist() for key in ("obj_pos", "target_pos")},
            )
        )
    manifest = dict(source=str(source), demo_count=len(demos), removed_fields=["images"], demo_files=records)
    write_json(output / "manifest.json", manifest)
    write_json(output / "rh56_official_task_init.json", dict(records=initial_records))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="a new output directory")
    args = parser.parse_args(argv)
    manifest = prepare_assets(args.source, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), demo_count=manifest["demo_count"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
