from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from .pcd import AdroitPCDConfig, depth_to_world, sample_adroit_pcd


@dataclass(frozen=True)
class EpisodePlan:
    sample_type: int
    noise_std: float
    branch_t: int
    branch_len: int


def draw_episode_plan(
    rng: np.random.Generator,
    episode_len: int,
    mix=(0.5, 0.3, 0.2),
    branch_tail_prob: float = 0.5,
    burst_len=(10, 30),
) -> EpisodePlan:
    sample_type = int(rng.choice(3, p=np.asarray(mix, dtype=np.float64) / np.sum(mix)))
    noise_std = float(rng.uniform(0.0, 0.2)) if sample_type < 2 else 0.0
    if sample_type != 1:
        return EpisodePlan(sample_type, noise_std, -1, 0)
    branch_t = int(rng.integers(15, max(16, episode_len - 39)))
    if rng.uniform() < branch_tail_prob:
        branch_len = int(episode_len)
    else:
        branch_len = int(rng.integers(burst_len[0], burst_len[1] + 1))
    return EpisodePlan(sample_type, noise_std, branch_t, branch_len)


def normalized_action(raw) -> np.ndarray:
    return np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0).astype(np.float32, copy=False)


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
            f.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape), dtype=sample.dtype,
                             chunks=(chunk_rows, *shape), compression=compression, compression_opts=4 if compression else None)
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
            f[key][offset:offset + length] = arr
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


def _descendants(model, root_body: int) -> set[int]:
    parent = np.asarray(model.body_parentid, dtype=np.int64)
    out = {int(root_body)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent):
            if int(parent_id) in out and body_id not in out:
                out.add(body_id)
                changed = True
    return out


def adroit_geom_masks(model):
    robot = _descendants(model, int(model.body_name2id("forearm")))
    obj = _descendants(model, int(model.body_name2id("Object")))
    try:
        target = _descendants(model, int(model.body_name2id("target")))
    except Exception:
        target = set()
    body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
    foreground = np.isin(body_ids, list(robot | obj | target))
    object_geom = np.isin(body_ids, list(obj))
    return foreground, object_geom


def _metric_depth(sim, depth):
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return near / (1.0 - depth.astype(np.float64) * (1.0 - near / far))


def capture_adroit_frame(env, rng, config=AdroitPCDConfig()):
    sim = env.sim
    size = int(config.image_size)
    sim.model.vis.quality.offsamples = 0
    rgb, depth = sim.render(width=size, height=size, depth=True, camera_name=config.camera, device_id=0)
    seg = sim.render(width=size, height=size, segmentation=True, camera_name=config.camera, device_id=0)
    rgb, depth, seg = np.asarray(rgb)[::-1], np.asarray(depth)[::-1], np.asarray(seg)[::-1]
    geom = np.where(seg[..., 0].astype(np.int64) == 5, seg[..., 1].astype(np.int64), -1)
    metric = _metric_depth(sim, depth)
    cam_id = int(sim.model.camera_name2id(config.camera))
    xyz = depth_to_world(metric, sim.data.cam_xpos[cam_id], np.asarray(sim.data.cam_xmat[cam_id]).reshape(3, 3), float(sim.model.cam_fovy[cam_id]))
    foreground, object_geom = adroit_geom_masks(sim.model)
    pcd_xyz, pcd_rgb, stats = sample_adroit_pcd(xyz.reshape(-1, 3), rgb.reshape(-1, 3), geom.reshape(-1), foreground, object_geom, rng, config)
    stats["foreground_pixel_fraction"] = float(((geom >= 0) & foreground[np.clip(geom, 0, len(foreground) - 1)]).mean())
    return rgb.astype(np.uint8), pcd_xyz, pcd_rgb, stats


def pen_oracle_action(env, home_qpos: np.ndarray) -> np.ndarray:
    """Bounded deterministic stabilizer used only for the plumbing smoke.

    It is deliberately not promoted as a task-solving expert controller.
    """
    current = np.asarray(env.data.qpos[:24], dtype=np.float64)
    raw = np.tanh(4.0 * (home_qpos - current)).astype(np.float32)
    raw[:3] = np.clip(raw[:3] + 0.25 * np.asarray(env.model.body_quat[env.target_obj_bid][:3], dtype=np.float32), -1.0, 1.0)
    return normalized_action(raw)


def branch_action(plan: EpisodePlan, step: int, rng, env, home_qpos):
    random_now = plan.sample_type == 2 or (plan.sample_type == 1 and plan.branch_t <= step < plan.branch_t + plan.branch_len)
    if random_now:
        return normalized_action(rng.uniform(-1.0, 1.0, env.action_space.shape)), True
    raw = pen_oracle_action(env, home_qpos)
    if plan.noise_std:
        raw = normalized_action(raw + rng.normal(0.0, plan.noise_std, env.action_space.shape))
    return raw, False


def dump_json(path: str | Path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
