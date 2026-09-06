"""Compact ManiSkill2 trajectory windows for the 4D JEPA WorldModel."""

from __future__ import annotations

import heapq
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


ACTION_DIM = 7
DEFAULT_POINT_COUNT = 1024
_SEED = 3072


def _episode_ids(root: Path, split: str) -> list[int]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train = [int(x) for x in manifest["train_episode_ids"]]
    val = [int(x) for x in manifest["val_episode_ids"]]
    if set(train).intersection(val):
        raise ValueError("train and validation episode ids must be disjoint")
    return train if split == "train" else val


def _validate_episode(episode: dict[str, np.ndarray], path: Path) -> None:
    xyz, action = episode["xyz"], episode["action"]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"{path}: xyz must have shape [frames, points, 3], got {xyz.shape}")
    if action.ndim != 2 or action.shape != (len(xyz) - 1, ACTION_DIM):
        raise ValueError(f"{path}: action must have shape [{len(xyz) - 1}, {ACTION_DIM}], got {action.shape}")
    if not np.isfinite(xyz).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: xyz and action must be finite")


def _sample_points(xyz: np.ndarray, point_count: int, episode_id: int, frame_id: int) -> np.ndarray:
    if xyz.shape[0] < point_count:
        raise ValueError(f"episode {episode_id} frame {frame_id} has {xyz.shape[0]} points; {point_count} required")
    if xyz.shape[0] == point_count:
        return np.ascontiguousarray(xyz, dtype=np.float32)
    rng = np.random.default_rng(_SEED + 1_000_003 * int(episode_id) + int(frame_id))
    indices = np.sort(rng.choice(xyz.shape[0], size=point_count, replace=False))
    return np.ascontiguousarray(xyz[indices], dtype=np.float32)


def action_stats(root: str | Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Compute action normalization statistics from complete episodes."""
    root = Path(root)
    chunks: list[np.ndarray] = []
    for episode_id in _episode_ids(root, split):
        path = root / f"episode_{episode_id:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: np.asarray(archive[key]) for key in archive.files}
        _validate_episode(episode, path)
        chunks.append(episode["action"].astype(np.float64, copy=False))
    if not chunks:
        raise ValueError(f"no episodes found under {root} for split={split!r}")
    actions = np.concatenate(chunks, axis=0)
    mean = actions.mean(axis=0).astype(np.float32)
    std = actions.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class CrossEpisodeBatchSampler(Sampler[list[int]]):
    """Yield batches whose windows come from distinct episodes."""

    def __init__(
        self,
        dataset: "ManiSkillSequenceDataset",
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        groups: OrderedDict[int, list[int]] = OrderedDict()
        for index, (episode_id, _) in enumerate(dataset.index):
            groups.setdefault(int(episode_id), []).append(index)
        if batch_size > len(groups):
            raise ValueError(
                f"batch_size={batch_size} requires at least that many distinct episodes; got {len(groups)}"
            )
        self.groups = groups
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        size = sum(len(indices) for indices in self.groups.values())
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            self.epoch += 1
        episode_ids = list(self.groups)
        if self.shuffle:
            order = torch.randperm(len(episode_ids), generator=generator).tolist()
            episode_ids = [episode_ids[index] for index in order]
        pools: dict[int, list[int]] = {}
        for episode_id in episode_ids:
            indices = list(self.groups[episode_id])
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in order]
            pools[episode_id] = indices

        tie_order = list(range(len(episode_ids)))
        if self.shuffle:
            tie_order = torch.randperm(len(episode_ids), generator=generator).tolist()
        heap = [
            (-len(pools[episode_id]), tie_order[index], episode_id)
            for index, episode_id in enumerate(episode_ids)
        ]
        heapq.heapify(heap)
        while heap:
            selected = [heapq.heappop(heap) for _ in range(min(self.batch_size, len(heap)))]
            if len(selected) < self.batch_size and self.drop_last:
                break
            batch = []
            for _, tie, episode_id in selected:
                batch.append(pools[episode_id].pop())
                if pools[episode_id]:
                    heapq.heappush(heap, (-len(pools[episode_id]), tie, episode_id))
            yield batch


class ManiSkillSequenceDataset(Dataset):
    """Read compact episodes and emit WorldModel-compatible sequence windows.

    Each item is ``(points, actions)`` with shapes ``[T+1,K,3]`` and
    ``[T*skip_frame,7]``.
    Episode splits come from ``manifest.json`` and windows never cross episodes.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        T: int = 3,
        point_count: int = DEFAULT_POINT_COUNT,
        single_frame_context: bool = False,
        skip_frame: int = 1,
        cache_size: int = 2,
    ):
        if T < 1:
            raise ValueError("T must be >= 1")
        if point_count < 1:
            raise ValueError("point_count must be >= 1")
        if skip_frame < 1:
            raise ValueError("skip_frame must be >= 1")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.task = manifest.get("task", self.root.name)
        self.T = int(T)
        self.point_count = int(point_count)
        self.single_frame_context = bool(single_frame_context)
        self.skip_frame = int(skip_frame)
        self.cache_size = max(1, int(cache_size))
        self.episode_ids = _episode_ids(self.root, split)
        self.index: list[tuple[int, int]] = []
        for episode_id in self.episode_ids:
            path = self.root / f"episode_{episode_id:04d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                episode = {key: np.asarray(archive[key]) for key in archive.files}
            _validate_episode(episode, path)
            if episode["xyz"].shape[1] < self.point_count:
                raise ValueError(f"{path}: has only {episode['xyz'].shape[1]} points per frame; {self.point_count} required")
            required_span = self.T * self.skip_frame
            self.index.extend((episode_id, start) for start in range(len(episode["xyz"]) - required_span))
        if not self.index:
            raise ValueError(f"no complete T={self.T} windows found under {self.root}")
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        if episode_id in self._cache:
            self._cache.move_to_end(episode_id)
            return self._cache[episode_id]
        path = self.root / f"episode_{episode_id:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: np.asarray(archive[key]) for key in archive.files}
        _validate_episode(episode, path)
        self._cache[episode_id] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_id, start = self.index[index]
        episode = self._load_episode(episode_id)
        frame_ids = start + np.arange(self.T + 1) * self.skip_frame
        raw_points = episode["xyz"][frame_ids]
        if self.single_frame_context:
            context_frame = int(frame_ids[self.T - 1])
            target_frame = int(frame_ids[self.T])
            context = _sample_points(raw_points[self.T - 1], self.point_count, episode_id, context_frame)
            target = _sample_points(raw_points[self.T], self.point_count, episode_id, target_frame)
            points = np.concatenate([np.repeat(context[None], self.T, axis=0), target[None]], axis=0)
        else:
            points = np.stack([
                _sample_points(frame, self.point_count, episode_id, int(frame_id))
                for frame_id, frame in zip(frame_ids, raw_points)
            ])
        actions = np.ascontiguousarray(
            episode["action"][start : start + self.T * self.skip_frame], dtype=np.float32
        )
        return torch.from_numpy(points), torch.from_numpy(actions)
