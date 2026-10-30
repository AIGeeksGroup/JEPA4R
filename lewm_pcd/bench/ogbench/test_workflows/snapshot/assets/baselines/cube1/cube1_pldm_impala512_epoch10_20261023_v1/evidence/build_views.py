"""Build sealed rows200 VDS views while preserving the source HDF5 unchanged."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


RAW_ROWS = 201
VIEW_ROWS = 200
KEYS = ("pixels", "action")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(path):
    value = Path(path).stat()
    return {
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inspect_source(path, episodes):
    with h5py.File(path, "r") as source:
        require(set(KEYS).issubset(source.keys()), f"missing RGB/action keys: {path}")
        require(len(source["ep_len"]) == episodes, f"episode count differs: {path}")
        require(
            np.array_equal(source["ep_len"][:], np.full(episodes, RAW_ROWS)),
            f"raw episode lengths differ: {path}",
        )
        require(
            np.array_equal(
                source["ep_offset"][:],
                np.arange(episodes, dtype=np.int64) * RAW_ROWS,
            ),
            f"raw episode offsets differ: {path}",
        )
        require(
            source["pixels"].shape
            == (episodes * RAW_ROWS, 224, 224, 3),
            f"pixel shape differs: {path}",
        )
        require(
            source["pixels"].dtype == np.dtype("uint8"),
            f"pixel dtype differs: {path}",
        )
        require(
            source["action"].shape == (episodes * RAW_ROWS, 5),
            f"action shape differs: {path}",
        )
        require(
            source["action"].dtype == np.dtype("float32"),
            f"action dtype differs: {path}",
        )

        actions = source["action"][:]
        terminal = np.arange(episodes, dtype=np.int64) * RAW_ROWS + VIEW_ROWS
        nonfinite = np.flatnonzero(~np.isfinite(actions).all(axis=1))
        require(
            len(nonfinite) == episodes,
            "expected exactly one nonfinite terminal action row per episode",
        )
        require(
            np.array_equal(nonfinite, terminal),
            "terminal NaN rows do not align to every episode boundary",
        )
        require(
            np.isfinite(np.delete(actions, terminal, axis=0)).all(),
            "nonterminal actions contain NaN or Inf",
        )


def build_view(source_path, output_path, episodes, source_sha, manifest_sha):
    with h5py.File(source_path, "r") as source:
        with h5py.File(output_path, "x", libver="latest") as view:
            for key in KEYS:
                dataset = source[key]
                layout = h5py.VirtualLayout(
                    shape=(episodes * VIEW_ROWS, *dataset.shape[1:]),
                    dtype=dataset.dtype,
                )
                virtual = h5py.VirtualSource(
                    str(source_path),
                    key,
                    shape=dataset.shape,
                )
                selector = h5py.MultiBlockSlice(
                    start=0,
                    stride=RAW_ROWS,
                    count=episodes,
                    block=VIEW_ROWS,
                )
                layout[:] = virtual[selector]
                view.create_virtual_dataset(key, layout)

            view.create_dataset(
                "ep_len",
                data=np.full(episodes, VIEW_ROWS, dtype=np.int32),
            )
            view.create_dataset(
                "ep_offset",
                data=np.arange(episodes, dtype=np.int64) * VIEW_ROWS,
            )
            view.attrs["view_recipe"] = "single_cube_H201_action_bearing_rows200_v1"
            view.attrs["canonical_index_sha256"] = manifest_sha
            view.attrs["source_train_sha256"] = source_sha["train"]
            view.attrs["source_validation_sha256"] = source_sha["validation"]
            view.attrs["excluded_terminal_action_row"] = VIEW_ROWS
            view.attrs["raw_rows_per_episode"] = RAW_ROWS


def verify_view(source_path, view_path, episodes, sample_episodes):
    with h5py.File(source_path, "r") as source, h5py.File(view_path, "r") as view:
        require(set(view) == {*KEYS, "ep_len", "ep_offset"}, "view fields differ")
        require(all(view[key].is_virtual for key in KEYS), "RGB/action must be virtual")
        require(len(view["ep_len"]) == episodes, "view episode count differs")
        require(
            np.array_equal(view["ep_len"][:], np.full(episodes, VIEW_ROWS)),
            "view episode lengths differ",
        )
        require(
            np.array_equal(
                view["ep_offset"][:],
                np.arange(episodes, dtype=np.int64) * VIEW_ROWS,
            ),
            "view episode offsets differ",
        )
        require(
            np.array_equal(view["action"][:], source["action"][:]
            .reshape(episodes, RAW_ROWS, 5)[:, :VIEW_ROWS]
            .reshape(episodes * VIEW_ROWS, 5)),
            "all action rows differ between source and view",
        )

        checks = []
        for episode in sample_episodes(episodes):
            for start in (0, 1, 92, 184):
                view_index = episode * VIEW_ROWS + start
                source_index = episode * RAW_ROWS + start
                require(
                    np.array_equal(
                        view["pixels"][view_index : view_index + 16],
                        source["pixels"][source_index : source_index + 16],
                    ),
                    f"RGB window differs: episode={episode} start={start}",
                )
                require(
                    np.array_equal(
                        view["action"][view_index : view_index + 15],
                        source["action"][source_index : source_index + 15],
                    ),
                    f"action window differs: episode={episode} start={start}",
                )
                checks.append(
                    {
                        "episode": int(episode),
                        "start": int(start),
                        "view_index": int(view_index),
                        "source_index": int(source_index),
                    }
                )
        return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract-output", required=True, type=Path)
    args = parser.parse_args()

    train_source = args.train_source.resolve()
    validation_source = args.validation_source.resolve()
    split_manifest = args.split_manifest.resolve()
    output = args.output.resolve()
    contract_output = args.contract_output.resolve()

    require(train_source.is_file(), f"missing train source: {train_source}")
    require(validation_source.is_file(), f"missing validation source: {validation_source}")
    require(split_manifest.is_file(), f"missing split manifest: {split_manifest}")
    require(not output.exists(), f"refusing existing output: {output}")
    require(not contract_output.exists(), f"refusing existing contract: {contract_output}")

    split = read_json(split_manifest)
    require(split.get("source_episodes") == 11000, "split manifest episode count differs")
    split_counts = {"train": 10000, "validation": 1000}
    for name, expected in split_counts.items():
        actual = sum(
            int(part["episodes"])
            for part in split["parts"]
            if part["split"] == name
        )
        require(actual == expected, f"split manifest {name} count differs")

    source_before = {
        "train": {**stat(train_source), "sha256": sha256(train_source)},
        "validation": {**stat(validation_source), "sha256": sha256(validation_source)},
        "split_manifest": {
            **stat(split_manifest),
            "sha256": sha256(split_manifest),
        },
    }
    require(
        source_before["train"]["sha256"]
        == "ed96945b068d7939e2f0cde9ea71abb3d3d038a302eb4dd0e32be00b5388d678",
        "train VDS SHA differs from readiness freeze",
    )
    require(
        source_before["validation"]["sha256"]
        == "f8f4a699c6e8ce9e072eb71596a5e36a1a5191dfcfc6f3c88234377d2545433c",
        "validation VDS SHA differs from readiness freeze",
    )

    inspect_source(train_source, split_counts["train"])
    inspect_source(validation_source, split_counts["validation"])

    output.mkdir(parents=True)
    train_view = output / "2d_train_action_rows200.h5"
    validation_view = output / "2d_validation_action_rows200.h5"
    sources_sha = {
        "train": source_before["train"]["sha256"],
        "validation": source_before["validation"]["sha256"],
    }

    manifest_path = output / "training_view_index.json"
    canonical_index_sha = source_before["split_manifest"]["sha256"]
    build_view(
        train_source,
        train_view,
        split_counts["train"],
        sources_sha,
        canonical_index_sha,
    )
    build_view(
        validation_source,
        validation_view,
        split_counts["validation"],
        sources_sha,
        canonical_index_sha,
    )

    def representative_episodes(episodes):
        required = {0, 1, min(2, episodes - 1), episodes - 2, episodes - 1}
        for episode in sorted(required):
            yield episode
        rng = np.random.default_rng(0)
        for episode in sorted(
            int(x) for x in rng.choice(episodes, size=min(8, episodes), replace=False)
        ):
            yield episode

    checks = {
        "train": verify_view(
            train_source,
            train_view,
            split_counts["train"],
            representative_episodes,
        ),
        "validation": verify_view(
            validation_source,
            validation_view,
            split_counts["validation"],
            representative_episodes,
        ),
    }
    source_after = {
        "train": {**stat(train_source), "sha256": sha256(train_source)},
        "validation": {
            **stat(validation_source),
            "sha256": sha256(validation_source),
        },
        "split_manifest": {
            **stat(split_manifest),
            "sha256": sha256(split_manifest),
        },
    }
    require(source_before == source_after, "source files changed while building views")

    views = []
    for name, path, episodes in (
        ("train", train_view, 10000),
        ("validation", validation_view, 1000),
    ):
        views.append(
            {
                "path": str(path),
                **stat(path),
                "sha256": sha256(path),
                "episodes": episodes,
                "rows": episodes * VIEW_ROWS,
                "clips": episodes * 185,
                "keys": list(KEYS),
            }
        )

    final_manifest = {
        "schema": "single_cube_pldm_rgb_rows200_index_v1",
        "stage": "sealed_rows200_views_complete",
        "actual_exit_code": 0,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(__file__),
        },
        "sources": source_before,
        "split_counts": split_counts,
        "raw_rows_per_episode": RAW_ROWS,
        "view_rows_per_episode": VIEW_ROWS,
        "window": {
            "clips_per_episode": 185,
            "frameskip": 1,
            "num_steps": 16,
            "span": 16,
            "terminal_row_excluded": True,
        },
        "views": views,
        "boundary_checks": checks,
        "raw_data_unchanged": source_before == source_after,
        "no_GT_model_input": True,
        "no_physical_visual_merge": True,
        "no_feature_cache": True,
        "canonical_index": {
            "path": str(split_manifest),
            "sha256": canonical_index_sha,
        },
    }
    write_json(manifest_path, final_manifest)
    final_sha = sha256(manifest_path)

    # Bind the final manifest to its own exact bytes through a detached contract.
    data_contract = {
        "schema": "single_cube_pldm_rgb_data_contract_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": final_sha,
        "data_root": str(output.parent),
        "sources": source_before,
        "split_counts": split_counts,
        "views": {
            item["path"].rsplit("/", 1)[-1]: {
                "name": item["path"].rsplit("/", 1)[-1],
                "path": item["path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "episodes": item["episodes"],
                "rows": item["rows"],
            }
            for item in views
        },
    }
    data_contract["views"] = {
        "train": next(
            x for x in data_contract["views"].values() if "train" in x["name"]
        ),
        "validation": next(
            x for x in data_contract["views"].values() if "validation" in x["name"]
        ),
    }
    contract_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(contract_output, data_contract)

    # The index cannot contain its own final hash without recursion. The source
    # adapter binds the final index bytes in data_contract.json.
    print(
        json.dumps(
            {
                "actual_exit_code": 0,
                "manifest_path": str(manifest_path),
                "manifest_sha256": final_sha,
                "contract_path": str(contract_output),
                "data_contract": data_contract,
                "views": views,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
