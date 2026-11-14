"""Point clouds of the current and goal frames, rendered online, through PointCloudWrapper's own extraction
(envs/pcd_wrapper.py, loaded from its file; no copy of it here)."""

import importlib.util

import gymnasium as gym
import numpy as np


class NativePCDFrame:
    def __init__(self, core_path, adapter_path, seed, *, core_factory=None, renderer_factory=None):
        spec = importlib.util.spec_from_file_location("_pcd_wrapper_core", core_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        spec = importlib.util.spec_from_file_location("_pcd_render_provider", adapter_path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)

        class Frame(gym.Env):
            @property
            def unwrapped(self):
                return self

            @property
            def _model(self):
                return self.model

            @property
            def _data(self):
                return self.data

        self.frame = Frame()
        self.cores = {phase: (core_factory or module.PointCloudWrapper)(self.frame) for phase in ["current", "goal"]}
        self.seed = int(seed)
        for core in self.cores.values():
            core._rng = np.random.default_rng(self.seed)
        self.provider = object.__new__(adapter.PCDRenderer)
        self.provider.u = self.frame
        self.renderer_factory = renderer_factory
        self.model = None
        self.renderer = None
        self.provide = adapter.PCDRenderer.provide.__get__(self.provider)

    def capture(self, u, data, rgb, phase):
        import mujoco

        if phase not in self.cores or rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError("True online AA4 frame/phase required")
        model = u._model
        if int(model.vis.quality.offsamples) != 4:
            raise ValueError("AA4 current/goal RGB domain changed")
        if model is not self.model:
            self.close()
            previous = int(model.vis.quality.offsamples)
            try:
                model.vis.quality.offsamples = 0
                self.renderer = (self.renderer_factory or mujoco.Renderer)(model, height=224, width=224)
            finally:
                model.vis.quality.offsamples = previous
            self.model = model
            self.provider.renderer = self.renderer
        self.frame.model, self.frame.data = model, data
        self.provider.rgb = rgb
        self.provider.last_seg = None
        core = self.cores[phase]
        cloud = core._point_cloud(render_fn=self.provide, return_trace=True, empty_crop_fallback=True)
        trace = cloud["pcd_trace"]
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
            raise ValueError("True main foreground cloud contract failed")
        if (
            np.any((uv < 0) | (uv >= 224))
            or not np.array_equal(color, rgb[uv[:, 1], uv[:, 0]])
            or not np.array_equal(geom, self.provider.last_seg[uv[:, 1], uv[:, 0], 0])
        ):
            raise ValueError("Online main UV/color/geom source mismatch")
        result = np.concatenate([xyz, color.astype(np.float32)], axis=1)
        return result

    def close(self):
        renderer = self.renderer
        self.renderer = None
        self.model = None
        if renderer is not None:
            renderer.close()
