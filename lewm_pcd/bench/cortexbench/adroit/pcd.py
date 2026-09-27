from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdroitPCDConfig:
    camera: str = "vil_camera"
    image_size: int = 256
    num_points: int = 1200
    voxel_size: float = 0.005
    crop_min: tuple[float, float, float] = (-0.60, -0.60, -0.05)
    crop_max: tuple[float, float, float] = (0.60, 0.60, 1.10)


def depth_to_world(depth: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, fovy: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[0] != depth.shape[1]:
        raise ValueError(f"depth must be square HxW, got {depth.shape}")
    size = depth.shape[0]
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    center = (size - 1.0) / 2.0
    focal = 0.5 * size / np.tan(np.deg2rad(float(fovy)) / 2.0)
    camera_points = np.stack(
        [(u - center) * depth / focal, -(v - center) * depth / focal, -depth], axis=-1
    )
    return camera_points @ np.asarray(camera_rotation, dtype=np.float64).T + np.asarray(camera_position, dtype=np.float64)


def sample_adroit_pcd(
    xyz: np.ndarray,
    rgb: np.ndarray,
    geom_ids: np.ndarray,
    foreground_geom: np.ndarray,
    object_geom: np.ndarray,
    rng: np.random.Generator,
    config: AdroitPCDConfig = AdroitPCDConfig(),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    geom_ids = np.asarray(geom_ids, dtype=np.int64).reshape(-1)
    if not (len(xyz) == len(rgb) == len(geom_ids)):
        raise ValueError("xyz, rgb, and geom_ids must have the same flattened length")
    foreground_geom = np.asarray(foreground_geom, dtype=bool)
    object_geom = np.asarray(object_geom, dtype=bool)
    valid = (geom_ids >= 0) & (geom_ids < len(foreground_geom))
    valid &= foreground_geom[np.clip(geom_ids, 0, len(foreground_geom) - 1)]
    lo = np.asarray(config.crop_min, dtype=np.float32)
    hi = np.asarray(config.crop_max, dtype=np.float32)
    valid &= np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb, geom_ids = xyz[valid], rgb[valid], geom_ids[valid]
    is_object = object_geom[geom_ids] if len(geom_ids) else np.zeros(0, dtype=bool)
    if len(xyz) and config.voxel_size > 0:
        q = np.floor(xyz / config.voxel_size).astype(np.int64)
        _, keep = np.unique(q, axis=0, return_index=True)
        xyz, rgb, is_object = xyz[keep], rgb[keep], is_object[keep]
    if len(xyz) == 0:
        raise ValueError("no finite foreground points remain after crop and voxel filtering")
    obj_idx = np.flatnonzero(is_object)
    non_obj_idx = np.flatnonzero(~is_object)
    target_obj = min(int(round(config.num_points * 0.25)), len(obj_idx))
    if len(non_obj_idx) == 0:
        selected = rng.choice(obj_idx, config.num_points, replace=len(obj_idx) < config.num_points)
    else:
        selected = np.concatenate(
            [
                rng.choice(obj_idx, target_obj, replace=False) if target_obj else np.empty(0, dtype=np.int64),
                rng.choice(non_obj_idx, config.num_points - target_obj, replace=len(non_obj_idx) < config.num_points - target_obj),
            ]
        )
        selected = rng.permutation(selected)
    out_xyz = xyz[selected].astype(np.float32, copy=False)
    out_rgb = rgb[selected].astype(np.uint8, copy=False)
    stats = {
        "input_points": float(len(valid)),
        "foreground_points": float(len(xyz)),
        "object_fraction": float(is_object[selected].mean()),
    }
    if not np.isfinite(out_xyz).all():
        raise ValueError("sampled point cloud contains non-finite xyz")
    return out_xyz, out_rgb, stats
