"""Validated frame-aligned cube-to-robot contact labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContactLabels:
    by_key: dict[tuple[int, int], int]
    positive_count: int
    negative_count: int

    def for_frames(
        self, episode_id: int, frame_ids: Iterable[int]
    ) -> np.ndarray:
        values: list[int] = []
        for frame_id in frame_ids:
            key = (int(episode_id), int(frame_id))
            if key not in self.by_key:
                raise ValueError(
                    f"missing contact label for episode {key[0]} frame {key[1]}"
                )
            values.append(self.by_key[key])
        return np.asarray(values, dtype=np.float32)


def load_contact_sidecar(
    path: str | Path, *, alignment_tolerance_m: float = 1e-4
) -> ContactLabels:
    """Load exact simulator contact labels and reject misaligned sidecars."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing contact sidecar: {source}")
    if not np.isfinite(alignment_tolerance_m) or alignment_tolerance_m < 0:
        raise ValueError("alignment tolerance must be finite and non-negative")
    required = {
        "episode_id",
        "frame_id",
        "cube_robot_contact",
        "object_pos_error_m",
        "tcp_pos_error_m",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"contact sidecar is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    count = len(arrays["episode_id"])
    if count < 1 or any(value.ndim != 1 or len(value) != count for value in arrays.values()):
        raise ValueError("contact sidecar arrays must be non-empty same-length vectors")
    episode_id = np.asarray(arrays["episode_id"], dtype=np.int64)
    frame_id = np.asarray(arrays["frame_id"], dtype=np.int64)
    contact = np.asarray(arrays["cube_robot_contact"], dtype=np.int64)
    if (episode_id < 0).any() or (frame_id < 0).any():
        raise ValueError("contact episode and frame ids must be non-negative")
    if not np.isin(contact, (0, 1)).all():
        raise ValueError("cube_robot_contact must contain only 0 and 1")
    object_error = np.asarray(arrays["object_pos_error_m"], dtype=np.float64)
    tcp_error = np.asarray(arrays["tcp_pos_error_m"], dtype=np.float64)
    if not np.isfinite(object_error).all() or not np.isfinite(tcp_error).all():
        raise ValueError("contact alignment errors must be finite")
    maximum_error = float(max(object_error.max(), tcp_error.max()))
    if maximum_error > alignment_tolerance_m:
        raise ValueError(
            "contact alignment error exceeds tolerance "
            f"{alignment_tolerance_m:.3e} m: {maximum_error:.3e} m"
        )
    by_key: dict[tuple[int, int], int] = {}
    for episode, frame, label in zip(episode_id, frame_id, contact, strict=True):
        key = (int(episode), int(frame))
        if key in by_key:
            raise ValueError(f"duplicate contact label for episode {key[0]} frame {key[1]}")
        by_key[key] = int(label)
    positives = int(contact.sum())
    return ContactLabels(
        by_key=by_key,
        positive_count=positives,
        negative_count=int(count - positives),
    )
