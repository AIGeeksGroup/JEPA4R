"""Shared scalar reader, immutable compact indices, and unsafe inputs."""
import importlib
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np


class ReaderTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'reader.py'
        self.assertTrue(path.is_file(), 'shared reader implementation is missing')
        return importlib.import_module('bench.ogbench.reader')

    def test_compact_index_preserves_scalar_slice_pickle_and_bounds(self):
        module = self.module()
        index = module.ReadOnlyClipIndex([(0, 0), (0, 184), (1, 0)])
        self.assertEqual(index[0], (0, 0))
        self.assertEqual(index[np.int64(-1)], (1, 0))
        self.assertEqual(index[1:], [(0, 184), (1, 0)])
        self.assertEqual(index[::-1], [(1, 0), (0, 184), (0, 0)])
        self.assertEqual(pickle.loads(pickle.dumps(index))[:], index[:])
        self.assertFalse(index._pairs.flags.writeable)
        with self.assertRaises(IndexError):
            index[3]
        with self.assertRaises(TypeError):
            index[1.2]

    def test_reader_rejects_unsupported_window_and_remote_before_open(self):
        module = self.module()
        for steps in (0, -1, 4, 17, True, 16.0):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                module.make_reader('missing.h5', 'double500', 'train', steps)
        with self.assertRaisesRegex(ValueError, 'local|remote'):
            module.make_reader('https://example.com/2d_train_action_rows500.h5', 'double500', 'train', 16)

    def test_shared_reader_uses_upstream_scalar_slice_and_item(self):
        module = self.module()
        try:
            import h5py
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        cls = module.reader_class()
        self.assertNotIn('_load_slice', cls.__dict__)
        self.assertNotIn('__getitem__', cls.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scalar.h5'
            with h5py.File(path, 'w', libver='latest') as handle:
                pixels = np.arange(12 * 2 * 2 * 3, dtype=np.uint8).reshape(12, 2, 2, 3)
                actions = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
                handle['pixels'] = pixels
                handle['action'] = actions
            # Construct only the upstream scalar path in this unit test. The
            # production factory is tested separately and has no fixture flag.
            reader = cls.__new__(cls)
            reader.h5_file = h5py.File(path, 'r', swmr=True)
            import os
            reader._owner_pid = os.getpid()
            reader.h5_path = path
            reader._cache = {'action': actions}
            reader._keys = ['pixels', 'action']
            reader.offsets = np.array([0, 6])
            reader.lengths = np.array([6, 6])
            reader.frameskip = 1
            reader.num_steps = 4
            reader.span = 4
            reader.transform = None
            reader.clip_indices = module.ReadOnlyClipIndex([(0, 0), (0, 2), (1, 0), (1, 2)])
            try:
                for index, start in ((0, 0), (1, 2), (2, 6), (-1, 8)):
                    sample = reader[index]
                    self.assertTrue(torch.equal(sample['pixels'], torch.from_numpy(pixels[start:start + 4]).permute(0, 3, 1, 2)))
                    self.assertTrue(torch.equal(sample['action'], torch.from_numpy(actions[start:start + 4])))
                self.assertEqual(reader.clip_indices[1], (0, 2))
                self.assertEqual(reader.clip_indices[2], (1, 0))
            finally:
                reader.close()

    def test_lazy_reader_class_can_unpickle_in_fresh_spawn_process(self):
        module = self.module()
        import subprocess
        import sys
        try:
            import h5py
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        cls = module.reader_class()
        reader = cls.__new__(cls)
        reader.h5_file = None
        reader.clip_indices = module.ReadOnlyClipIndex([(0, 0), (1, 0)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reader.pkl'
            path.write_bytes(pickle.dumps(reader))
            script = ('import pickle,sys; from pathlib import Path; '
                      'reader=pickle.loads(Path(sys.argv[1]).read_bytes()); '
                      'assert reader.h5_file is None; '
                      'assert reader.clip_indices[-1] == (1,0)')
            result = subprocess.run([sys.executable, '-c', script, str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_lazy_reader_module_has_fresh_process_class_resolver(self):
        module = self.module()
        # Pickle resolves classes using getattr(module, class_name), including
        # in a worker that has never called make_reader/reader_class.
        self.assertTrue(callable(getattr(module, '__getattr__', None)))
        with self.assertRaises(AttributeError):
            module.__getattr__('unregistered_reader')


if __name__ == '__main__':
    unittest.main()
