"""Narrow metadata validator for the sealed QuadrupleCube 600-row VDS."""
import json
from pathlib import Path

import numpy as np


def validate_view_attributes(file, *, expected_sha256=None):
    path = Path(file.filename).resolve()
    manifest_path = path.parent / "training_view_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    views = {Path(row["path"]).name: row for row in manifest["views"]
             if row.get("keys") == ["pixels", "action"]}
    row = views.get(path.name)
    if row is None or row["sha256"] != expected_sha256:
        raise ValueError("QuadrupleCube view identity differs from manifest")
    expected_episodes = 10000 if row["path"].endswith("2d_train_action_rows600.h5") else 1000
    if (row["episodes"] != expected_episodes or row["rows"] != expected_episodes * 600
            or set(file) != {"pixels", "action", "ep_len", "ep_offset"}
            or file["pixels"].shape != (expected_episodes * 600, 224, 224, 3)
            or file["action"].shape != (expected_episodes * 600, 5)
            or file["pixels"].dtype != np.dtype("uint8")
            or file["action"].dtype != np.dtype("float32")
            or not file["pixels"].is_virtual or not file["action"].is_virtual
            or not np.array_equal(file["ep_len"][:], np.full(expected_episodes, 600))
            or not np.array_equal(file["ep_offset"][:], np.arange(expected_episodes) * 600)):
        raise ValueError("QuadrupleCube VDS schema, modality, or episode offsets differ")
    attrs = file.attrs
    if (attrs.get("view_recipe") != "quadruple_H600_action_bearing_rows600_v1"
            or attrs.get("env_type") != "quadruple"
            or attrs.get("physical_cube_count") != 4
            or attrs.get("horizon_actions") != 600
            or attrs.get("canonical_index_sha256") != manifest["canonical_index"]["sha256"]
            or attrs.get("independent_seal_sha256") != manifest["independent_seal"]["sha256"]
            or attrs.get("raw_terminal_observation_row600") != "preserved_in_raw_archive"):
        raise ValueError("QuadrupleCube provenance attributes differ")
    return row
