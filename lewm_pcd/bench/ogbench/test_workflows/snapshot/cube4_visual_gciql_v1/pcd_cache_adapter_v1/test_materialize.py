"""Pure CPU source/artifact tests; no renderer, Torch, JAX, or policy imports."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get('ORIGINAL_3D_CACHE_WORKSPACE', str(HERE / 'readback_sources_v1')))


class ThinQuadAdapter(unittest.TestCase):
    def adapter(self):
        path = HERE / 'materialize.py'
        self.assertTrue(path.is_file(), 'Quad3D thin materializer has not been implemented')
        spec = importlib.util.spec_from_file_location('_quad3d_adapter_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_guard_delta_preserves_inference_and_GPU_LUT(self):
        adapter = self.adapter()
        sources, _ = adapter.build_sources(BASE, '2' * 64)
        name = 'official_state_3d_cache_v1/produce.py'
        before, after = (ast.parse((BASE / name).read_bytes()), ast.parse(sources[name]))
        inference = lambda tree: next(ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                                      if isinstance(n, ast.FunctionDef) and n.name == 'infer')
        self.assertEqual(inference(before), inference(after))
        self.assertIn(b'completed == 5500 and rows == 5505500', sources[name])
        self.assertIn(b'qpos_width=42, qvel_width=38', sources['official_state_3d_cache_v1/common.py'])
        self.assertIn(b'quadruple_official_pcd_epoch10_v1', sources['official_state_3d_cache_v1/common.py'])
        gpu = 'official_state_3d_cache_v1/acceleration_v1/v4/gpu_preprocess.py'
        self.assertEqual(sources[gpu], (BASE / gpu).read_bytes())

    def test_four_cube_pose_guard_and_capture_use_real42_38(self):
        adapter = self.adapter()
        sources, _ = adapter.build_sources(BASE, '2' * 64)
        tree = ast.parse(sources['official_state_3d_cache_v1/pcd_renderer.py'])
        capture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_capture')
        data = type('Data', (), {})()
        class Base:
            def set_state(self, qpos, qvel):
                data.qpos, data.qvel = qpos.copy(), qvel.copy()
            def render(self, camera):
                self.last_camera = camera
                return np.zeros((224, 224, 3), np.uint8)
        class Frame:
            def capture(self, base, actual_data, rgb, phase):
                return np.zeros((1200, 6), np.float32)
        state = dict(base=Base(), data=data, arrays={'train': {
            'qpos': np.zeros((1, 42), np.float32), 'qvel': np.zeros((1, 38), np.float32)}})
        namespace = dict(np=np, STATE=state)
        exec(compile(ast.Module(body=[capture], type_ignores=[]), '<actual_capture>', 'exec'), namespace)
        cloud, image = namespace['_capture'](Frame(), 'train', 0)
        self.assertEqual((cloud.shape, image.shape), ((1200, 6), (224, 224, 3)))
        state['arrays']['train']['qpos'] = np.zeros((1, 28), np.float32)
        with self.assertRaises(AssertionError): namespace['_capture'](Frame(), 'train', 0)

    def test_config_membership_handles_resolved_external_checkpoint(self):
        adapter = self.adapter()
        sources, _ = adapter.build_sources(BASE, '2' * 64)
        tree = ast.parse(sources['official_state_3d_cache_v1/audit_cache.py'])
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_config_key')
        class Resolved:
            def __init__(self, value): self.value = value
            def resolve(self): return self.value
        class Workspace:
            def __truediv__(self, name):
                assert name == 'checkpoints/cube4/3d_epoch10/config.json'
                return Resolved('/actual/checkpoints/cube4/3d_epoch10/config.json')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual_membership>', 'exec'), namespace)
        key = 'checkpoints/cube4/3d_epoch10/config.json'
        self.assertEqual(namespace['checkpoint_config_key'](Workspace(), Resolved('/actual/' + key), {key: '2' * 64}), key)
        with self.assertRaises(AssertionError):
            namespace['checkpoint_config_key'](Workspace(), Resolved('/actual/checkpoints/cube2/3d_epoch10/config.json'), {key: '2' * 64})

    def test_resource_profile_requires_new_Quad_gate_identity(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); plan = directory / 'plan.json'; gate = directory / 'gate.json'
            plan.write_text(json.dumps(dict(checkpoint_sha256=adapter.CHECKPOINT_SHA,
                workspace=str(directory), gate_dir=str(directory),
                source_pins={adapter.MODULE + '/produce.py': 'a' * 64})))
            samples = directory / 'real_PCD32.npz'; samples.write_bytes(b'fake fixture identity only')
            good = dict(actual_exit_code=0, accepted=True, prepared_bitexact=True, latent_bitexact=True,
                        candidate_median_seconds=.009, baseline_median_seconds=.01,
                        candidate_source_sha256=adapter.EXPECTED['official_state_3d_cache_v1/acceleration_v1/v4/gpu_preprocess.py'],
                        plan_sha256=adapter.sha(plan), checkpoint_sha256=adapter.CHECKPOINT_SHA,
                        samples_path=str(samples), samples_sha256=adapter.sha(samples), source_samples_count=32,
                        real_raw_ids=list(range(29)) + [999, 1000, 1001])
            gate.write_text(json.dumps(good))
            profile = adapter.resource_profile(plan, gate)
            self.assertEqual((profile['renderer_workers'], profile['prefetch_jobs']), (16, 32))
            good['checkpoint_sha256'] = 'f' * 64; gate.write_text(json.dumps(good))
            with self.assertRaises(AssertionError): adapter.resource_profile(plan, gate)


if __name__ == '__main__':
    unittest.main()
