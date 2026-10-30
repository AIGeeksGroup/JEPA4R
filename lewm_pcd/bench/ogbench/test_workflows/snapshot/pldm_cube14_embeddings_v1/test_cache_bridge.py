"""Bounded CPU/source tests; no model construction, rendering or data writes."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'inventory/reference_original'
BRIDGE = HERE / 'cache_bridge.py'
SPEC = Path('metadata-only-binding-spec.json')


def plan(n=1, dim=512):
    return dict(cube_count=n, latent_dim=dim, encoder_family='pldm',
        renderer_workers=2, prefetch=2, encoder_batch=32, torch_cpu_threads=1,
        pixel_preprocess_device='shared_cpu', encoder_preprocessing_device='cpu',
        host_tensor_dtype='uint8', inference_precision='fp32', storage_dtype='float32',
        training_updates=0, SR_episodes=0, encoder_spec=str(SPEC),
        checkpoint='/pinned/weights_epoch_10.pt', checkpoint_sha256='a'*64)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        if not BRIDGE.exists():
            self.bridge = None
            return
        spec = importlib.util.spec_from_file_location('_pldm14_test_bridge', BRIDGE)
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        self.bridge.ROOT = REFERENCE
        self.encoder_spec = dict(cube_count=1, checkpoint='/pinned/weights_epoch_10.pt', checkpoint_sha256='a'*64)
        read_text = Path.read_text
        def metadata_read(path, *args, **kwargs):
            return json.dumps(self.encoder_spec) if path == SPEC else read_text(path, *args, **kwargs)
        self.metadata_patch = patch.object(Path, 'read_text', autospec=True, side_effect=metadata_read)
        self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)

    def available(self):
        self.assertIsNotNone(self.bridge, 'The explicit PLDM512 shared-CPU bridge is missing')
        return self.bridge

    def test_bridge_available(self):
        self.available()

    def test_existing_family_effective_sources_are_exact(self):
        bridge = self.available()
        for mode in ('producer', 'audit'):
            original = bridge.foundation_api(mode)
            source = original.effective_source if mode == 'producer' else original.source
            for n in (1, 4):
                for dim in (768, 1024):
                    p = dict(cube_count=n, latent_dim=dim)
                    self.assertEqual(bridge.effective_source(p, mode), source(p))

    def test_new_cpu_sources_keep_original_loop_and_alignment(self):
        bridge = self.available()
        for n in (1, 4):
            p = plan(n)
            source = bridge.effective_source(p, 'producer')
            ast.parse(source)
            self.assertIn("assert policy == 'shared_cpu'", source)
            self.assertIn("if policy == 'shared_cpu':\n        activate('cpu', None)", source)
            self.assertIn("assert selected_device == 'cpu'", source)
            self.assertIn('ProcessPoolExecutor(max_workers=2,', source)
            self.assertIn('for _ in range(min(2, len(pending))):', source)
            original = bridge.foundation_api('producer').effective_source(dict(cube_count=n, latent_dim=768))
            proof = lambda value: value[value.index('    alignment_checked = False'):value.index('\n    def infer(images):')]
            self.assertEqual(proof(source), proof(original))
            self.assertEqual(source[source.index('        pieces, padded = [], 0'):source.index("    records, pending, boundary")],
                original[original.index('        pieces, padded = [], 0'):original.index("    records, pending, boundary")].replace('len(images), 768', 'len(images), 512'))

    def test_audit_cpu_policy_and_real_identity(self):
        bridge = self.available()
        for n in (1, 4):
            source = bridge.effective_source(plan(n), 'audit')
            ast.parse(source)
            self.assertIn("assert policy == 'shared_cpu'", source)
            self.assertIn("selected_device = 'cpu'", source)
            self.assertIn("if policy not in ('cpu', 'shared_cpu'):", source)
            self.assertIn('ROWS, DIM = 1001, 512', source)
            self.encoder_spec['cube_count'] = n
            module = bridge.load(plan(n), 'audit')
            self.assertEqual(module.PRODUCER, 'pldm_cube14_embeddings_v1/cache_bridge.py')
            self.assertEqual(module.__file__, str(BRIDGE))
            self.assertIn('encoder_preprocess', module.PROTOCOL_KEYS)

    def test_producer_reuses_renderer_and_array_reader_without_model(self):
        bridge = self.available()
        module = bridge.load(plan(), 'producer')
        original = bridge.foundation_api('producer')
        self.assertEqual(module.HERE, REFERENCE/'foundation_visual_gciql_v1/cache')
        self.assertIs(module.source_arrays, original.arrays_existing)
        self.assertEqual(module.__file__, str(BRIDGE))
        self.assertIn('prefetch', module.CONTRACT_KEYS)
        self.assertIn('storage_dtype', module.CONTRACT_KEYS)

    def test_load_rejects_wrong_cube_checkpoint_and_weight_identity(self):
        bridge = self.available()
        for key, wrong in [('cube_count',4), ('checkpoint','/other/weights_epoch_10.pt'), ('checkpoint_sha256','b'*64)]:
            self.encoder_spec = dict(cube_count=1, checkpoint='/pinned/weights_epoch_10.pt', checkpoint_sha256='a'*64)
            self.encoder_spec[key] = wrong
            with self.subTest(key=key), patch.object(bridge, 'foundation_api', side_effect=AssertionError('Imported before binding check')):
                with self.assertRaisesRegex(AssertionError, 'Encoder spec/plan binding mismatch: ' + key):
                    bridge.load(plan(), 'producer')

    def test_rejects_protocol_changes(self):
        bridge = self.available()
        for key, value in [('cube_count',2), ('renderer_workers',8), ('prefetch',8),
                           ('encoder_batch',64), ('pixel_preprocess_device','shared_gpu'),
                           ('encoder_preprocessing_device','cuda'), ('inference_precision','fp16'),
                           ('storage_dtype','float16'), ('training_updates',1), ('SR_episodes',1)]:
            changed = dict(plan(), **{key:value})
            with self.assertRaises(AssertionError, msg=key):
                bridge.effective_source(changed, 'producer')

    def test_counted_replacement_rejects_missing_or_duplicate_anchor(self):
        bridge = self.available()
        self.assertEqual(bridge.replace_once('before', 'before', 'after'), 'after')
        for value in ('other', 'before before'):
            with self.assertRaises(AssertionError):
                bridge.replace_once(value, 'before', 'after')

    def test_cli_delegates_existing_main_with_original_flags(self):
        bridge = self.available()
        for mode, extra in [('producer',['--execute','--pilot']),
                            ('audit',['--cache-dir','cache','--output-dir','audit'])]:
            observed = []
            module = types.SimpleNamespace(main=lambda: observed.append(sys.argv[1:]))
            with patch.object(Path, 'read_text', return_value=json.dumps(plan())), patch.object(bridge,'load',return_value=module), patch.object(sys,'argv',['test']):
                bridge.main(['--mode',mode,'--plan','plan.json',*extra])
            self.assertEqual(observed, [['--plan','plan.json',*extra]])


if __name__ == '__main__':
    unittest.main()
