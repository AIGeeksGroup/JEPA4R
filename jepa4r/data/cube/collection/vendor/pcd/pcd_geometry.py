"""Renders the frames (RGB, depth, segmentation) that PointCloudWrapper back-projects; the extraction stays in
PointCloudWrapper (no copy of it here)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PCDRenderer:
    def __init__(self, renderer_root, reference_config, reference_status, main_core_path):
        sys.path.insert(0, str(renderer_root))
        self.main_core_path = Path(main_core_path)
        self.core_sha = sha(self.main_core_path)
        spec = importlib.util.spec_from_file_location("frozen_main_pcd_core", self.main_core_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from cube_rgb_renderer import RGBOnlyRenderer
        import gymnasium as gym

        self.base = RGBOnlyRenderer(renderer_root, reference_config, reference_status)
        self.u = self.base.u
        self.renderer = self.base.unused_geometry_renderer
        u = self.u

        class Frame(gym.Env):
            @property
            def unwrapped(self):
                return self

            @property
            def _model(self):
                return u._model

            @property
            def _data(self):
                return u._data

        self.core = module.PointCloudWrapper(Frame())
        self.rgb = None
        self.last_seg = None
        self.metadata = {
            **self.base.metadata,
            "pcd_produced": True,
            "renderer_kind": "main_pointcloud_wrapper",
            "main_core_path": str(self.main_core_path),
            "main_core_sha256": self.core_sha,
            "renderer_adapter_sha256": sha(__file__),
            "rgb_render_calls": 0,
            "geometry_offsamples": 0,
            "PCD_color": "read-only canonical AA4 same-row owner",
            "view_evaluation": "not_evaluated",
            "encoder_features_cached": False,
        }

    def provide(self, camera=None, depth=False, segmentation=False):
        if camera != "front_pixels" or depth and segmentation:
            raise ValueError("Unexpected main render request")
        if not depth and not segmentation:
            return self.rgb.copy()
        r = self.renderer
        if depth:
            r.disable_segmentation_rendering()
            r.enable_depth_rendering()
        else:
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
        r.update_scene(self.u._data, camera="front_pixels")
        value = r.render().copy()
        if segmentation:
            self.last_seg = value
        return value

    def render(self, qpos, qvel, canonical_rgb, seed):
        if (
            qpos.shape != (21,)
            or qvel.shape != (20,)
            or qpos.dtype != np.float32
            or qvel.dtype != np.float32
            or canonical_rgb.shape != (224, 224, 3)
            or canonical_rgb.dtype != np.uint8
        ):
            raise ValueError("Official source/owner contract differs")
        u = self.u
        u.set_state(qpos, qvel)
        if not np.array_equal(u._data.qpos, qpos.astype(float)) or not np.array_equal(u._data.qvel, qvel.astype(float)):
            raise ValueError("Exact source restore failed")
        before = {
            k: np.asarray(getattr(u._data, k)).copy()
            for k in ["qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat", "qacc_warmstart", "eq_active"]
            if hasattr(u._data, k)
        }
        time = float(u._data.time)
        self.rgb = canonical_rgb
        self.core._rng = np.random.default_rng(seed)
        rng_before = rng_sha(self.core._rng.bit_generator.state)
        cloud = self.core._point_cloud(render_fn=self.provide, return_trace=True, empty_crop_fallback=True)
        trace = cloud["pcd_trace"]
        rng_after = rng_sha(self.core._rng.bit_generator.state)
        if time != u._data.time or any(not np.array_equal(v, getattr(u._data, k)) for k, v in before.items()):
            raise ValueError("Main frame extraction changed scene state")
        xyz, color = cloud["pcd_xyz"], cloud["pcd_rgb"]
        uv = trace["source_pixel_uv"]
        geom = trace["source_geom_id"]
        if (
            xyz.shape != (1200, 3)
            or xyz.dtype != np.float32
            or not np.isfinite(xyz).all()
            or not np.any(xyz)
            or color.shape != (1200, 3)
            or color.dtype != np.uint8
        ):
            raise ValueError("Main cloud finite/shape/nonzero failed")
        if (
            np.any((uv < 0) | (uv >= 224))
            or not np.array_equal(color, canonical_rgb[uv[:, 1], uv[:, 0]])
            or not np.array_equal(geom, self.last_seg[uv[:, 1], uv[:, 0], 0])
        ):
            raise ValueError("Main trace UV/owner/geom identity differs")
        valid = self.last_seg[..., 0] >= 0
        cube_visible = int(np.count_nonzero(self.core._geom_cube[np.where(valid, self.last_seg[..., 0], 0)] & valid))
        values = {
            "pcd_xyz": xyz,
            "pcd_rgb": color,
            "source_pixel_uv": uv.astype(np.int16),
            "source_geom_id": geom.astype(np.int32),
            "camera_K": trace["camera_K"],
            "world_from_cv": trace["world_from_cv"],
            "fallback_used": np.array(trace["fallback_used"], bool),
            "raw_foreground_pool": np.int32(trace["raw_foreground_pool"]),
            "cropped_pool": np.int32(trace["foreground_cropped_pool"]),
            "voxel_unique_pool": np.int32(trace["voxel_unique_pool"]),
            "sampled_unique_points": np.int32(trace["actual_selected_unique_pool_indices"]),
            "repeated_samples": np.int32(trace["repeated_samples"]),
            "sampled_cube_points": np.int32(trace["sampled_cube_points"]),
            "sampling_seed": np.int64(seed),
            "source_rgb_row_sha256": np.array(
                hashlib.sha256(canonical_rgb.tobytes()).hexdigest().encode(), dtype="S64"
            ),
            "view_branch_code": np.uint8(0),
            "cube_visible_pixels": np.int32(cube_visible),
        }
        core_trace = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in trace.items()
            if k not in ["source_pixel_uv", "source_geom_id"]
        }
        sampling = {
            k: v for k, v in core_trace.items() if k not in ["camera_K", "world_from_cv", "camera", "height", "width"]
        }
        sampling["sampling_seed"] = seed
        receipt = {
            "core_source_sha256": self.core_sha,
            "renderer_source_sha256": sha(__file__),
            "sampling": sampling,
            "core_trace": core_trace,
            "rng_before_sha256": rng_before,
            "rng_after_sha256": rng_after,
            "view_evaluation": "not_evaluated",
            "cube_visible_pixels": cube_visible,
            "RGB_source": "immutable canonical AA4 owner same row",
            "encoder_features_cached": False,
            "camera_K": core_trace["camera_K"],
            "world_from_cv": core_trace["world_from_cv"],
        }
        return values, receipt

    def close(self):
        errors = []
        for r in [self.base.unused_geometry_renderer, getattr(self.u, "_renderer", None)]:
            if r is not None:
                try:
                    r.close()
                except BaseException as e:
                    errors.append(repr(e))
        self.base.unused_geometry_renderer = None
        self.u._renderer = None
        try:
            self.base.env.close()
        except BaseException as e:
            errors.append(repr(e))
        self.base.env = None
        if errors:
            raise RuntimeError("Owned main frame cleanup failed:" + str(errors))
