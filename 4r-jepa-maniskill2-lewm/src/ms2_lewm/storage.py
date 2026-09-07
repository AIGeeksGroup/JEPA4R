"""Compact episode writers; collection, official readers and model inputs stay unchanged.

Only the point-cloud storage precision is reduced. Images use the pinned
official JPEG encoder and its default quality. No trajectory filtering or
subsampling happens here, and no full intermediate dataset is created.
"""

from __future__ import annotations

import io
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
from PIL import Image

from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.data.formats.lance import (
    LanceWriter, _DEFAULT_JPEG_QUALITY, _encode_frame,
)
from stable_worldmodel.data.formats.utils import is_image_column


GIB = 1024 ** 3
DEFAULT_RESERVE_BYTES = 10 * GIB
POINTCLOUD_BYTES_PER_FRAME = 1200 * 6 * np.dtype(np.float16).itemsize


def compact_episode(episode):
    """Drop exact aliases and quantize only the on-disk point-cloud copy."""
    result = dict(episode)
    if "pointcloud" not in result:
        raise ValueError("Collection requires a pointcloud column.")
    points = np.asarray(result["pointcloud"])
    if points.ndim != 3 or points.shape[1:] != (1200, 6):
        raise ValueError(f"Expected [steps,1200,6] pointcloud, got {points.shape}.")
    if not np.isfinite(points).all() or np.abs(points).max() > np.finfo(np.float16).max:
        raise ValueError("Pointcloud must be finite and representable as float16.")
    aliases = {"rl_xyz": points[..., :3], "rl_rgb": points[..., 3:]}
    if "observation" in result:
        aliases.update(proprio=result["observation"], rl_state=result["observation"])
    for name, canonical in aliases.items():
        if name in result:
            if not np.array_equal(np.asarray(result[name]), np.asarray(canonical), equal_nan=True):
                raise ValueError(f"Refusing to discard non-identical alias {name}.")
            del result[name]
    result["pointcloud"] = points.astype(np.float16)
    # The official writer's image classifier expects a list of step arrays.
    return {name: list(values) for name, values in result.items()}


class StorageBudget:
    """Check remaining output bytes against current free space every episode.

    Estimates use actual committed-file growth (JPEG included). A 20% margin is applied
    to the largest observed episode rate. This is an estimate, not a promise
    that future scenes compress identically; subsequent episodes recheck it.
    """

    def __init__(self, path, *, expected_episodes=None, horizon=200,
                 reserve_bytes=DEFAULT_RESERVE_BYTES, report_path=None):
        self.path = Path(path)
        if reserve_bytes < 0 or horizon < 1 or (expected_episodes is not None and expected_episodes < 1):
            raise ValueError("Invalid disk budget arguments.")
        self.expected_steps = None if expected_episodes is None else int(expected_episodes) * (int(horizon) + 1)
        self.horizon = int(horizon)
        self.reserve_bytes = int(reserve_bytes)
        self.report_path = Path(report_path) if report_path else None
        self.max_bytes_per_step = 0.0
        self.report = None

    def check(self, *, completed_steps=0, episode_bytes=None, episode_steps=None, nominal=False):
        if episode_bytes is not None:
            self.max_bytes_per_step = max(self.max_bytes_per_step, episode_bytes / episode_steps)
        rate = POINTCLOUD_BYTES_PER_FRAME if nominal else self.max_bytes_per_step
        remaining = max(0, (self.expected_steps or completed_steps) - completed_steps)
        # Before data exists, admit only a small probe episode plus reserve.
        # Raw fp16 bytes are NOT a lower bound for a compressed full dataset.
        projected = math.ceil(rate * (min(remaining, self.horizon + 1) if nominal else remaining)
                              * (1.0 if nominal else 1.2))
        ancestor = self.path.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free
        required = projected + self.reserve_bytes
        self.report = {
            "path": str(self.path.resolve()), "stage": "nominal_pointcloud_only" if nominal else "measured_episode",
            "expected_steps_including_reset_allowance": self.expected_steps,
            "completed_steps": int(completed_steps), "remaining_steps": int(remaining),
            "maximum_observed_bytes_per_step": self.max_bytes_per_step,
            "estimated_remaining_bytes": projected, "reserve_bytes": self.reserve_bytes,
            "required_free_bytes": required, "available_free_bytes": free, "passed": free >= required,
            "note": "Nominal check budgets a probe episode only. Measured remaining-file growth adds 20% margin; completed data is already deducted from disk free space.",
        }
        if self.report_path:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(self.report, indent=2) + "\n")
        if free < required:
            raise RuntimeError(
                f"Insufficient dataset disk space: {free / GIB:.2f} GiB free, "
                f"need estimated {projected / GIB:.2f} GiB remaining + {self.reserve_bytes / GIB:.2f} GiB reserve. "
                "No sampling recipe is reduced; choose a larger output volume."
            )
        return self.report


class CompactLanceWriter(LanceWriter):
    """Official streaming Lance writer with fp16 pointcloud and exact alias removal."""

    def __init__(self, path, *, mode="error", expected_episodes=None, horizon=200,
                 reserve_bytes=DEFAULT_RESERVE_BYTES, report_path=None):
        if mode != "error":
            raise ValueError("Compact collection requires a new dataset (mode='error').")
        self.budget = StorageBudget(path, expected_episodes=expected_episodes, horizon=horizon,
                                    reserve_bytes=reserve_bytes, report_path=report_path)
        self.budget.check(nominal=True)
        self._previous_file_bytes = 0
        super().__init__(path, mode=mode)

    def _init_schema(self, sample_ep):
        super()._init_schema(sample_ep)
        index = self._schema.get_field_index("pointcloud")
        self._schema = self._schema.set(index, pa.field("pointcloud", pa.list_(pa.float16(), self._dims["pointcloud"])))

    def write_episodes(self, episodes):
        # Commit one probe episode, then bounded groups of ten. This gives
        # real compressed-file measurements before a full collection while
        # preserving the official streaming iterator and transition order.
        iterator = (compact_episode(episode) for episode in episodes)
        while True:
            try:
                first = next(iterator)
            except StopIteration:
                return
            before = self._global_ptr
            self.budget.check(completed_steps=before)
            count = 1 if before == 0 else 10
            super().write_episodes(itertools.chain([first], itertools.islice(iterator, count - 1)))
            self._check_committed_size(self._global_ptr - before)

    def write_episode(self, episode):
        before = self._global_ptr
        self.budget.check(completed_steps=before)
        super().write_episode(compact_episode(episode))
        self._check_committed_size(self._global_ptr - before)

    def _check_committed_size(self, steps):
        total = sum(file.stat().st_size for file in self.budget.path.rglob("*") if file.is_file())
        growth = max(0, total - self._previous_file_bytes)
        self._previous_file_bytes = total
        self.budget.check(completed_steps=self._global_ptr, episode_bytes=growth, episode_steps=steps)


class CompactHDF5Writer(HDF5Writer):
    """Direct streaming HDF5, gzip compressed, read by the untouched official reader.

    RGB follows the previous Lance-to-HDF5 path's JPEG round-trip, using the
    official encoder/quality. Only one final file is kept, with exact official
    episode offsets plus explicit indices used by the evaluation script.
    """

    def __init__(self, path, *, mode="error", expected_episodes=None, horizon=200,
                 reserve_bytes=DEFAULT_RESERVE_BYTES, report_path=None):
        if mode != "error":
            raise ValueError("Compact collection requires a new dataset (mode='error').")
        self.budget = StorageBudget(path, expected_episodes=expected_episodes, horizon=horizon,
                                    reserve_bytes=reserve_bytes, report_path=report_path)
        self.budget.check(nominal=True)
        self._previous_file_bytes = 0
        super().__init__(path, mode=mode)

    def _init_schema(self, sample_ep):
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            chunk_steps = max(1, min(32, (1024 * 1024) // max(1, sample.nbytes)))
            self._f.create_dataset(col, shape=(0, *sample.shape), maxshape=(None, *sample.shape),
                                   dtype=sample.dtype, chunks=(chunk_steps, *sample.shape),
                                   compression="gzip", compression_opts=1, shuffle=True)
        self._f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
        self._f.attrs["pointcloud_storage_dtype"] = "float16"
        self._f.attrs["jpeg_quality"] = _DEFAULT_JPEG_QUALITY

    def write_episode(self, episode):
        episode = compact_episode(episode)
        length = len(episode["pointcloud"])
        for name, values in list(episode.items()):
            if name == "pixels" or name.startswith("pixels_") or is_image_column(values):
                episode[name] = np.stack([
                    np.asarray(Image.open(io.BytesIO(_encode_frame(np.asarray(value), _DEFAULT_JPEG_QUALITY))).convert("RGB"))
                    for value in values
                ])
        episode["episode_idx"] = np.full(length, self._ep_written, dtype=np.int32)
        episode["step_idx"] = np.arange(length, dtype=np.int32)
        # Admission before each episode prevents writing into the reserve.
        self.budget.check(completed_steps=self._global_ptr)
        super().write_episode(episode)
        self._f.flush()
        total_bytes = self.path.stat().st_size
        growth = max(0, total_bytes - self._previous_file_bytes)
        self._previous_file_bytes = total_bytes
        self.budget.check(completed_steps=self._global_ptr,
                          episode_bytes=growth, episode_steps=length)


def collection_writer(path, **kwargs):
    suffix = Path(path).suffix.lower()
    if suffix == ".lance":
        return CompactLanceWriter(path, **kwargs)
    if suffix in (".h5", ".hdf5"):
        return CompactHDF5Writer(path, **kwargs)
    raise ValueError("Dataset output must end in .lance, .h5 or .hdf5.")
