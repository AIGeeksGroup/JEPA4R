"""Minimal wrapper of the already passed RGB-only restoration/render flow."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np

BASE_SHA = 'd5e8a56cd90d1425be3e302be880bd110838d8efdceb877cf34fcdf865a6b5bd'
PASSED_RGB_SCRIPT_SHA = 'cc4951b090efb3aeec86b98854de56bf163bbe88d6e32676ee47a51b74120c77'


class RGBOnlyRenderer:
    def __init__(self, reference_config, reference_status):
        from . import scene as base
        from ..common import sha_file
        from ogbench.manipspace.envs.cube_env import CubeEnv
        from ogbench.manipspace import mjcf_utils
        reference_config = json.loads(Path(reference_config).read_text())
        reference_status = json.loads(Path(reference_status).read_text())
        if sys.version != reference_config['python'] or sha_file(inspect.getfile(CubeEnv)) != reference_config['env_source_sha256']:
            raise ValueError('Pinned simulator runtime source changed')
        if any(importlib.metadata.version(key) != version for key, version in reference_config['packages'].items()):
            raise ValueError('Pinned simulator dependency versions changed')
        self.env, self.u, unused, _, _, _, _ = base.setup()
        # Preserve the passed lifecycle exactly until worker/owner equality
        # has been measured. This context produces no new depth or PCD.
        self.unused_geometry_renderer = unused
        xml_sha = hashlib.sha256(mjcf_utils.to_string(self.u._mjcf_model).encode()).hexdigest()
        assets = {key: hashlib.sha256(value).hexdigest() for key, value in mjcf_utils.get_assets(self.u._mjcf_model).items()}
        if xml_sha != reference_status['model_xml_sha256'] or assets != reference_status['asset_sha256']:
            self.close(); raise ValueError('Pinned scene XML/assets changed')
        self.metadata = {'base_sha256': BASE_SHA, 'passed_rgb_script_sha256': PASSED_RGB_SCRIPT_SHA,
                         'helper_sha256': sha_file(__file__), 'model_xml_sha256': xml_sha, 'asset_sha256': assets,
                         'packages': reference_config['packages'], 'camera': 'front_pixels', 'rgb_offsamples': 4,
                         'width': 224, 'height': 224, 'no_env_steps': True, 'pcd_produced': False,
                         'hidden_state': 'bootstrap only; no archived ctrl/mocap or physics rollout equivalence claim'}

    def render(self, qpos, qvel):
        if qpos.shape != (21,) or qvel.shape != (20,) or qpos.dtype != np.float32 or qvel.dtype != np.float32:
            raise ValueError('Expected original Cube float32 state row')
        self.u.set_state(qpos, qvel)
        if not np.array_equal(self.u._data.qpos, qpos.astype(np.float64)) or not np.array_equal(self.u._data.qvel, qvel.astype(np.float64)):
            raise ValueError('State restore is not exact')
        before = {key: np.asarray(getattr(self.u._data, key)).copy() for key in ('qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat')}
        rgb = self.u.render(camera='front_pixels').copy()
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8 or not all(np.array_equal(value, np.asarray(getattr(self.u._data, key))) for key, value in before.items()):
            raise ValueError('Render image contract or state nonmutation failed')
        return rgb

    def close(self):
        if getattr(self, 'unused_geometry_renderer', None) is not None:
            self.unused_geometry_renderer.close(); self.unused_geometry_renderer = None
        if getattr(self, 'env', None) is not None:
            self.env.close(); self.env = None

    def diagnostics(self):
        """Actual NoAA geom IDs/context for a failed RGB-byte equality gate only."""
        names = ('qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'cam_xpos', 'cam_xmat', 'geom_xpos', 'geom_xmat')
        context = {key: np.asarray(getattr(self.u._data, key)).copy() for key in names}
        context['time'] = np.asarray(self.u._data.time).copy()
        context['mat_rgba'] = np.asarray(self.u._model.mat_rgba).copy()
        context['geom_rgba'] = np.asarray(self.u._model.geom_rgba).copy()
        renderer = self.unused_geometry_renderer
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(self.u._data, camera='front_pixels')
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
        if not all(np.array_equal(value, np.asarray(getattr(self.u._data, key))) for key, value in context.items() if hasattr(self.u._data, key)):
            raise ValueError('Diagnostic segmentation mutated physical state')
        return segmentation, context
