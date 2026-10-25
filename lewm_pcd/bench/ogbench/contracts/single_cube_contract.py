"""Narrow metadata validator for the sealed SingleCube RGB 200-row VDS."""
import json
from pathlib import Path

import numpy as np


TRAIN_VIEW = "2d_train_action_rows200.h5"
VAL_VIEW = "2d_validation_action_rows200.h5"
ROWS_PER_EPISODE = 200


def validate_view_attributes(file, *, expected_sha256=None):
    path = Path(file.filename).resolve()
    manifest_path = path.parent / "training_view_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    views = {
        Path(row["path"]).name: row
        for row in manifest["views"]
        if row.get("keys") == ["pixels", "action"]
    }
    row = views.get(path.name)
    if row is None or row["sha256"] != expected_sha256:
        raise ValueError("SingleCube view identity differs from manifest")

    if path.name == TRAIN_VIEW:
        expected_episodes = 10000
    elif path.name == VAL_VIEW:
        expected_episodes = 1000
    else:
        raise ValueError("unrecognized SingleCube RGB view")

    expected_rows = expected_episodes * ROWS_PER_EPISODE
    if (
        row["episodes"] != expected_episodes
        or row["rows"] != expected_rows
        or set(file) != {"pixels", "action", "ep_len", "ep_offset"}
        or file["pixels"].shape != (expected_rows, 224, 224, 3)
        or file["action"].shape != (expected_rows, 5)
        or file["pixels"].dtype != np.dtype("uint8")
        or file["action"].dtype != np.dtype("float32")
        or not file["pixels"].is_virtual
        or not file["action"].is_virtual
        or not np.array_equal(
            file["ep_len"][:], np.full(expected_episodes, ROWS_PER_EPISODE)
        )
        or not np.array_equal(
            file["ep_offset"][:],
            np.arange(expected_episodes) * ROWS_PER_EPISODE,
        )
    ):
        raise ValueError(
            "SingleCube RGB VDS schema, modality, or episode offsets differ"
        )

    attrs = file.attrs
    if (
        attrs.get("view_recipe")
        != "single_cube_H201_action_bearing_rows200_v1"
        or attrs.get("canonical_index_sha256")
        != manifest["canonical_index"]["sha256"]
        or attrs.get("source_train_sha256") != manifest["sources"]["train"]["sha256"]
        or attrs.get("source_validation_sha256")
        != manifest["sources"]["validation"]["sha256"]
        or attrs.get("excluded_terminal_action_row") != 200
    ):
        raise ValueError("SingleCube RGB provenance attributes differ")
    return row
