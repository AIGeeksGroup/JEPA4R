"""Source-only checks: no torch import, model construction, data read, or launch."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


CASE = Path(__file__).resolve().parent
CORE_SHA = '64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51'
IO_SHA = '320bff92d774c8499e9a6f69a622a1d6b64a5e43bd038d7124246f4f50cec8b9'


class SourceContract(unittest.TestCase):
    def test_exact_reference_entry_and_wrapper(self):
        for name, expected in [('source/train_lejepa_mw.py', CORE_SHA),
                               ('worker_prefetch_v1/native_io.py', IO_SHA)]:
            source = (CASE / name).read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected)
            compile(source, name, 'exec')
            ast.parse(source)

    def test_launcher_contract_and_guard(self):
        path = CASE / 'launch.py'
        self.assertTrue(path.is_file(), 'The source-only launch guard is not implemented yet')
        spec = importlib.util.spec_from_file_location('prepared_launch', path)
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        plan = launch.checked_plan(CASE, local_only=True)
        tail = plan['argv'][plan['argv'].index('--') + 1:]
        self.assertEqual(tail, [
            '--task', 'ogb_double_front500', '--dataset', plan['data']['train_vds'],
            '--name', 'lejepa_ogb_double_front500_seed0_w24pf2_10ep', '--workers', '24'])
        self.assertEqual(plan['budget']['batches_per_epoch'], 19531)
        self.assertEqual(plan['budget']['nominal_loop_updates'], 195310)
        self.assertEqual(plan['budget']['dropped_frames_per_epoch'], 64)
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'existing'
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                launch.reserve_output(existing)

    def test_incomplete_rgb_sources_refuse_metadata_ready(self):
        path = CASE / 'data_preflight.py'
        self.assertTrue(path.is_file(), 'The metadata source guard is not implemented yet')
        spec = importlib.util.spec_from_file_location('prepared_data_preflight', path)
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for i in range(125):
                split = 'train' if i < 100 else 'validation'
                item = root / f'fake_{split}_part{i}_pix.h5'
                item.write_bytes(b'x')
                sources[str(item)] = {'bytes': 1}
            index = root / 'index.json'
            index.write_text(json.dumps({'raw_data_files': sources}))
            index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
            self.assertTrue(guard.inspect_sources(index, index_sha)['training_metadata_ready'])
            (root / 'fake_validation_part100_pix.h5').unlink()
            validation_missing = guard.inspect_sources(index, index_sha)
            self.assertTrue(validation_missing['training_metadata_ready'])
            self.assertFalse(validation_missing['validation_metadata_ready'])
            (root / 'fake_train_part0_pix.h5').unlink()
            (root / 'fake_train_part1_pix.h5').write_bytes(b'wrong_size')
            result = guard.inspect_sources(index, index_sha)
            self.assertFalse(result['training_metadata_ready'])
            self.assertEqual(result['split_counts']['train']['missing'], 1)
            self.assertEqual(result['split_counts']['train']['size_mismatch'], 1)


if __name__ == '__main__':
    unittest.main()
