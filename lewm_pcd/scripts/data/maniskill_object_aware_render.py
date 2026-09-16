"""Six-view RGB-D/segmentation rendering for ManiSkill PickCube replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    uid: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int
    height: int
    fov: float = np.deg2rad(90.0)
    near: float = 0.01
    far: float = 100.0


def default_camera_specs(extra_resolution: int = 256) -> tuple[CameraSpec, ...]:
    """Return the fixed camera protocol used by replay and CEM."""
    if extra_resolution <= 0:
        raise ValueError("extra_resolution must be positive")
    target = (-0.1, 0.0, 0.1)
    return (
        CameraSpec("base_camera", (0.30, 0.00, 0.60), target, 128, 128),
        CameraSpec("front_left", (0.25, -0.35, 0.45), target, extra_resolution, extra_resolution),
        CameraSpec("front_right", (0.25, 0.35, 0.45), target, extra_resolution, extra_resolution),
        CameraSpec("rear_left", (-0.35, -0.35, 0.45), target, extra_resolution, extra_resolution),
        CameraSpec("rear_right", (-0.35, 0.35, 0.45), target, extra_resolution, extra_resolution),
        CameraSpec("top", (0.00, 0.00, 0.80), target, extra_resolution, extra_resolution),
    )


def camera_observation_to_world(
    position_mm: np.ndarray,
    rgb: np.ndarray,
    segmentation: np.ndarray,
    cam2world_gl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one batch-one camera observation to valid world-frame points."""
    position_mm = np.asarray(position_mm)
    rgb = np.asarray(rgb)
    segmentation = np.asarray(segmentation)
    cam2world_gl = np.asarray(cam2world_gl)
    if position_mm.ndim != 4 or position_mm.shape[-1] != 3:
        raise ValueError(f"position must have shape [1,H,W,3], got {position_mm.shape}")
    if rgb.shape != position_mm.shape:
        raise ValueError(f"rgb shape {rgb.shape} does not match position shape {position_mm.shape}")
    if segmentation.ndim != 4 or segmentation.shape[:3] != position_mm.shape[:3]:
        raise ValueError(
            f"segmentation shape {segmentation.shape} does not match position shape {position_mm.shape}"
        )
    if cam2world_gl.shape not in {(4, 4), (1, 4, 4)}:
        raise ValueError(f"cam2world_gl must have shape [4,4] or [1,4,4], got {cam2world_gl.shape}")

    position = position_mm[0].astype(np.float32) / 1000.0
    colors = rgb[0, ..., :3].astype(np.uint8)
    semantic = segmentation[0, ..., 0].astype(np.int32)
    valid = (semantic != 0) & np.isfinite(position).all(axis=-1)
    position = position[valid]
    colors = colors[valid]
    semantic = semantic[valid]
    homogeneous = np.concatenate(
        [position, np.ones((len(position), 1), dtype=np.float32)], axis=1
    )
    transform = cam2world_gl[0] if cam2world_gl.ndim == 3 else cam2world_gl
    world = homogeneous @ transform.astype(np.float32).T
    return world[:, :3].astype(np.float32), colors, semantic


class MultiViewPickCubeRenderer:
    """Attach five temporary cameras and merge their semantic point clouds."""

    def __init__(self, env, extra_resolution: int = 256):
        from mani_skill.sensors.camera import Camera, CameraConfig
        from mani_skill.utils import sapien_utils

        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.specs = default_camera_specs(extra_resolution)
        self.cameras = {"base_camera": self.env._sensors["base_camera"]}
        for spec in self.specs[1:]:
            config = CameraConfig(
                uid=spec.uid,
                pose=sapien_utils.look_at(spec.eye, spec.target),
                width=spec.width,
                height=spec.height,
                fov=spec.fov,
                near=spec.near,
                far=spec.far,
            )
            self.cameras[spec.uid] = Camera(config, self.env.scene)

    def capture(self, observation: dict | None = None) -> dict[str, np.ndarray]:
        """Render all views and return merged XYZ/RGB/semantic arrays."""
        import torch

        if observation is None:
            observation = self.env.get_obs()
        self.env.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        xyz_parts, rgb_parts, semantic_parts = [], [], []
        for spec in self.specs:
            if spec.uid == "base_camera" and "sensor_data" in observation:
                images = observation["sensor_data"][spec.uid]
                params = observation["sensor_param"][spec.uid]
            else:
                camera = self.cameras[spec.uid]
                # Cameras created outside BaseEnv._sensors are not included in
                # ``scene.update_render(update_sensors=True)``; take their
                # pictures explicitly before reading the textures.
                camera.camera.take_picture()
                images = camera.get_obs(rgb=True, position=True, segmentation=True)
                params = camera.get_params()
            to_numpy = lambda value: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            xyz, colors, semantic = camera_observation_to_world(
                to_numpy(images["position"]),
                to_numpy(images["rgb"]),
                to_numpy(images["segmentation"]),
                to_numpy(params["cam2world_gl"]),
            )
            xyz_parts.append(xyz)
            rgb_parts.append(colors)
            semantic_parts.append(semantic)
        return {
            "xyz": np.concatenate(xyz_parts, axis=0).astype(np.float32),
            "rgb": np.concatenate(rgb_parts, axis=0).astype(np.uint8),
            "semantic_id": np.concatenate(semantic_parts, axis=0).astype(np.int32),
        }
