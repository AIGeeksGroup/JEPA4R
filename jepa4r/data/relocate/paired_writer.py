"""The paired HDF writer shared by the Relocate collectors (Adroit and the RH56 twin)."""

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np


class AdroitPairedWriter:
    PIXEL_KEYS = frozenset(("pixels",))
    PCD_KEYS = frozenset(("pcd_xyz", "pcd_rgb"))

    def __init__(self, pixel_path: str | Path, pcd_path: str | Path):
        self.pixel_path = Path(pixel_path)
        self.pcd_path = Path(pcd_path)
        self._files = None

    @staticmethod
    def split_episode(episode: dict[str, np.ndarray]):
        pixel = {k: v for k, v in episode.items() if k not in AdroitPairedWriter.PCD_KEYS}
        pcd = {k: v for k, v in episode.items() if k not in AdroitPairedWriter.PIXEL_KEYS}
        return pixel, pcd

    def __enter__(self):
        if self.pixel_path.exists() or self.pcd_path.exists():
            raise FileExistsError("paired smoke output already exists; use a new output directory")
        self.pixel_path.parent.mkdir(parents=True, exist_ok=True)
        self.pcd_path.parent.mkdir(parents=True, exist_ok=True)
        self._files = (h5py.File(self.pixel_path, "w"), h5py.File(self.pcd_path, "w"))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._files is not None:
            for f in self._files:
                f.close()
            self._files = None

    @staticmethod
    def _init_file(f, episode):
        for key, values in episode.items():
            sample = np.asarray(values)
            if sample.ndim == 0:
                raise ValueError(f"column {key} must have a time axis")
            shape = sample.shape[1:]
            chunk_rows = max(1, min(16, sample.shape[0]))
            compression = "gzip" if key in ("pixels", "pcd_rgb") else None
            f.create_dataset(
                key,
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=sample.dtype,
                chunks=(chunk_rows, *shape),
                compression=compression,
                compression_opts=4 if compression else None,
            )
        f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

    @staticmethod
    def _append(f, episode):
        if "ep_len" not in f:
            AdroitPairedWriter._init_file(f, episode)
        keys = {k for k in f.keys() if k not in ("ep_len", "ep_offset")}
        if keys != set(episode):
            raise ValueError(f"paired schema mismatch: file={sorted(keys)} incoming={sorted(episode)}")
        length = len(next(iter(episode.values())))
        offset = int(f["ep_len"][:].sum())
        for key, values in episode.items():
            arr = np.asarray(values)
            if len(arr) != length or arr.shape[1:] != f[key].shape[1:]:
                raise ValueError(f"column shape mismatch: {key} {arr.shape} vs {f[key].shape}")
            f[key].resize(offset + length, axis=0)
            f[key][offset : offset + length] = arr
        n = len(f["ep_len"])
        f["ep_len"].resize(n + 1, axis=0)
        f["ep_len"][n] = length
        f["ep_offset"].resize(n + 1, axis=0)
        f["ep_offset"][n] = offset

    def write_episode(self, episode):
        if self._files is None:
            raise RuntimeError("writer is not open")
        pixel, pcd = self.split_episode(episode)
        self._append(self._files[0], pixel)
        self._append(self._files[1], pcd)
