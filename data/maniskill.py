"""Compact ManiSkill2 trajectory windows for the 4D JEPA WorldModel."""

from __future__ import annotations

import heapq
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.contact import ContactLabels, load_contact_sidecar


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


def _validate_episode(
    episode: dict[str, np.ndarray],
    path: Path,
    *,
    require_rgb: bool = False,
    require_proprio: bool = False,
) -> None:
    xyz, action = episode["xyz"], episode["action"]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"{path}: xyz must have shape [frames, points, 3], got {xyz.shape}")
    if action.ndim != 2 or action.shape != (len(xyz) - 1, ACTION_DIM):
        raise ValueError(f"{path}: action must have shape [{len(xyz) - 1}, {ACTION_DIM}], got {action.shape}")
    if not np.isfinite(xyz).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: xyz and action must be finite")
    if require_rgb:
        if "rgb" not in episode:
            raise ValueError(f"{path}: rgb is required when include_rgb=True")
        rgb = episode["rgb"]
        if rgb.shape != xyz.shape or rgb.dtype != np.uint8:
            raise ValueError(
                f"{path}: rgb must be uint8 with shape {xyz.shape}, "
                f"got {rgb.shape} {rgb.dtype}"
            )
    if require_proprio:
        expected = {
            "qpos": (len(xyz), 9),
            "qvel": (len(xyz), 9),
            "tcp_pose": (len(xyz), 7),
            "object_pos": (len(xyz), 3),
        }
        for name, shape in expected.items():
            if name not in episode:
                raise ValueError(
                    f"{path}: {name} is required when include_proprio=True"
                )
            value = episode[name]
            if value.shape != shape:
                raise ValueError(
                    f"{path}: {name} must have shape {shape}, got {value.shape}"
                )
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{path}: {name} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {name} must be finite")


def _sample_indices(
    count: int, point_count: int, episode_id: int, frame_id: int
) -> np.ndarray:
    if count < point_count:
        raise ValueError(
            f"episode {episode_id} frame {frame_id} has {count} points; "
            f"{point_count} required"
        )
    if count == point_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_SEED + 1_000_003 * int(episode_id) + int(frame_id))
    return np.sort(rng.choice(count, size=point_count, replace=False))


def _sample_points(
    xyz: np.ndarray, point_count: int, episode_id: int, frame_id: int
) -> np.ndarray:
    indices = _sample_indices(len(xyz), point_count, episode_id, frame_id)
    return np.ascontiguousarray(xyz[indices], dtype=np.float32)


def _sample_features(
    episode: dict[str, np.ndarray],
    frame_id: int,
    point_count: int,
    episode_id: int,
    include_rgb: bool,
) -> np.ndarray:
    indices = _sample_indices(
        episode["xyz"].shape[1], point_count, episode_id, frame_id
    )
    xyz = np.asarray(episode["xyz"][frame_id, indices], dtype=np.float32)
    if not include_rgb:
        return np.ascontiguousarray(xyz)
    rgb = np.asarray(episode["rgb"][frame_id, indices], dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.concatenate([xyz, rgb], axis=-1))


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


def load_proprio_stats(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the manifest's train-only 25-D normalization."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    normalization = manifest.get("proprio_normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"{root}: manifest is missing proprio_normalization")
    train_ids = [int(value) for value in manifest.get("train_episode_ids", [])]
    stats_ids = [int(value) for value in normalization.get("episode_ids", [])]
    if normalization.get("source_split") != "train" or stats_ids != train_ids:
        raise ValueError(
            f"{root}: proprio normalization must use the ordered training split"
        )
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,):
        raise ValueError(
            f"{root}: proprio mean/std must both have shape [25], "
            f"got {mean.shape} and {std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(f"{root}: proprio mean/std must be finite")
    if not (std > 0).all():
        raise ValueError(f"{root}: proprio std must be positive")
    return mean, std


def _proprio(episode: dict[str, np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            [episode["qpos"], episode["qvel"], episode["tcp_pose"]], axis=-1
        ),
        dtype=np.float32,
    )


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
        sample_repeats: list[int] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if sample_repeats is None:
            sample_repeats = getattr(dataset, "sample_repeats", None)
        if sample_repeats is None:
            sample_repeats = [1] * len(dataset.index)
        if len(sample_repeats) != len(dataset.index) or any(
            not isinstance(value, (int, np.integer)) or int(value) < 1
            for value in sample_repeats
        ):
            raise ValueError("sample_repeats must contain one positive integer per window")
        groups: OrderedDict[int, list[int]] = OrderedDict()
        for index, (episode_id, _) in enumerate(dataset.index):
            groups.setdefault(int(episode_id), []).extend(
                [index] * int(sample_repeats[index])
            )
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

    By default each item is ``(points, actions)`` with shapes
    ``[T+1,K,3 or 6]`` and ``[T*skip_frame,7]``. With proprioception enabled,
    each item is a mapping that also contains aligned ``proprio`` and
    ``object_pos`` at the sampled observation frames.
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
        include_rgb: bool = False,
        include_proprio: bool = False,
        contact_sidecar: str | Path | None = None,
        contact_alignment_tolerance_m: float = 1e-4,
        phase_repeat: int = 1,
    ):
        if T < 1:
            raise ValueError("T must be >= 1")
        if point_count < 1:
            raise ValueError("point_count must be >= 1")
        if skip_frame < 1:
            raise ValueError("skip_frame must be >= 1")
        if phase_repeat < 1:
            raise ValueError("phase_repeat must be >= 1")
        if contact_sidecar is not None and not include_proprio:
            raise ValueError("contact labels require include_proprio=True")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.task = manifest.get("task", self.root.name)
        self.T = int(T)
        self.point_count = int(point_count)
        self.single_frame_context = bool(single_frame_context)
        self.skip_frame = int(skip_frame)
        self.include_rgb = bool(include_rgb)
        self.include_proprio = bool(include_proprio)
        self.contact_labels: ContactLabels | None = (
            None
            if contact_sidecar is None
            else load_contact_sidecar(
                contact_sidecar,
                alignment_tolerance_m=contact_alignment_tolerance_m,
            )
        )
        self.phase_repeat = int(phase_repeat)
        self.cache_size = max(1, int(cache_size))
        self.episode_ids = _episode_ids(self.root, split)
        self.index: list[tuple[int, int]] = []
        self.sample_repeats: list[int] = []
        for episode_id in self.episode_ids:
            path = self.root / f"episode_{episode_id:04d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                episode = {key: np.asarray(archive[key]) for key in archive.files}
            _validate_episode(
                episode,
                path,
                require_rgb=self.include_rgb,
                require_proprio=self.include_proprio,
            )
            if episode["xyz"].shape[1] < self.point_count:
                raise ValueError(f"{path}: has only {episode['xyz'].shape[1]} points per frame; {self.point_count} required")
            required_span = self.T * self.skip_frame
            for start in range(len(episode["xyz"]) - required_span):
                self.index.append((episode_id, start))
                repeat = 1
                if self.contact_labels is not None:
                    frame_ids = start + np.arange(self.T + 1) * self.skip_frame
                    contact = self.contact_labels.for_frames(episode_id, frame_ids)
                    targets = contact[1:]
                    transition = (contact[:-1] == 0) & (contact[1:] == 1)
                    if (targets == 0).any() or transition.any():
                        repeat = self.phase_repeat
                self.sample_repeats.append(repeat)
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
        _validate_episode(
            episode,
            path,
            require_rgb=self.include_rgb,
            require_proprio=self.include_proprio,
        )
        self._cache[episode_id] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        episode_id, start = self.index[index]
        episode = self._load_episode(episode_id)
        frame_ids = start + np.arange(self.T + 1) * self.skip_frame
        if self.single_frame_context:
            context_frame = int(frame_ids[self.T - 1])
            target_frame = int(frame_ids[self.T])
            context = _sample_features(
                episode,
                context_frame,
                self.point_count,
                episode_id,
                self.include_rgb,
            )
            target = _sample_features(
                episode,
                target_frame,
                self.point_count,
                episode_id,
                self.include_rgb,
            )
            points = np.concatenate([np.repeat(context[None], self.T, axis=0), target[None]], axis=0)
        else:
            points = np.stack([
                _sample_features(
                    episode,
                    int(frame_id),
                    self.point_count,
                    episode_id,
                    self.include_rgb,
                )
                for frame_id in frame_ids
            ])
        actions = np.ascontiguousarray(
            episode["action"][start : start + self.T * self.skip_frame], dtype=np.float32
        )
        if not self.include_proprio:
            return torch.from_numpy(points), torch.from_numpy(actions)
        proprio = _proprio(episode)[frame_ids]
        object_pos = np.ascontiguousarray(
            episode["object_pos"][frame_ids], dtype=np.float32
        )
        result = {
            "points": torch.from_numpy(points),
            "actions": torch.from_numpy(actions),
            "proprio": torch.from_numpy(proprio),
            "object_pos": torch.from_numpy(object_pos),
        }
        if self.contact_labels is not None:
            result["cube_robot_contact"] = torch.from_numpy(
                self.contact_labels.for_frames(episode_id, frame_ids)
            )
        return result
