"""HDF lifecycle and native sampling preservation; no learning runtime required."""
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'scripts/baselines/pen/data.py'


class PenDataTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('_pen_baseline_data_test', HELPER)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def dataset(self, steps=1, skip=1):
        lengths = [7, 20, 21, 25, 101]
        span = steps * skip
        return SimpleNamespace(h5_path=ROOT / 'fake-pen.h5', lengths=lengths,
            offsets=[0, 7, 27, 48, 73], num_steps=steps, frameskip=skip, span=span,
            clip_indices=[(ep, start) for ep, size in enumerate(lengths)
                          for start in range(max(0, size - span + 1))], h5_file=None,
            _cache={'action': object()})

    def test_hook_preserves_native_indices_arrays_and_rng(self):
        for steps, skip, count in ((1, 1, 174), (4, 5, 91)):
            with self.subTest(steps=steps, skip=skip):
                ds = self.dataset(steps, skip)
                indices, lengths, offsets, cache = ds.clip_indices, ds.lengths, ds.offsets, ds._cache
                before = list(indices)
                rng = random.getstate()
                original = lambda *args, **kwargs: ds
                swm = SimpleNamespace(data=SimpleNamespace(load_dataset=original))
                seen = []
                with self.mod.dataset_handle_hook(swm, ds.h5_path, seen.append):
                    self.assertIs(swm.data.load_dataset(ds.h5_path), ds)
                self.assertIs(swm.data.load_dataset, original)
                self.assertIs(ds.clip_indices, indices)
                self.assertEqual(indices, before)
                self.assertEqual(len(indices), count)
                self.assertIs(ds.lengths, lengths)
                self.assertIs(ds.offsets, offsets)
                self.assertIs(ds._cache, cache)
                self.assertEqual(random.getstate(), rng)
                self.assertEqual(seen[0]['native_clip_count'], count)
                self.assertTrue(seen[0]['native_indices_preserved'])
                self.assertFalse(seen[0]['source_arrays_changed'])

    def test_valid_terminal_image_and_native_tail_window_remain(self):
        for ds in (self.dataset(), self.dataset(4, 5)):
            swm = SimpleNamespace(data=SimpleNamespace(load_dataset=lambda *a, **k: ds))
            with self.mod.dataset_handle_hook(swm, ds.h5_path):
                swm.data.load_dataset(ds.h5_path)
            for ep, size in enumerate(ds.lengths):
                if size >= ds.span:
                    self.assertIn((ep, size - ds.span), ds.clip_indices)

    def test_wrong_path_untouched_and_loader_restored_after_error(self):
        ds, other = self.dataset(), self.dataset()
        other.h5_path = ROOT / 'another.h5'
        def original(name, **kwargs):
            return ds if Path(name).resolve() == Path(ds.h5_path).resolve() else other
        swm = SimpleNamespace(data=SimpleNamespace(load_dataset=original))
        seen = []
        with self.assertRaises(RuntimeError):
            with self.mod.dataset_handle_hook(swm, ds.h5_path, seen.append):
                self.assertIs(swm.data.load_dataset(ds.h5_path), ds)
                self.assertIs(swm.data.load_dataset(other.h5_path), other)
                raise RuntimeError('body failure')
        self.assertIs(swm.data.load_dataset, original)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(other.clip_indices), 174)

    def test_closes_wrapped_handle_once_without_touching_cache(self):
        class Handle:
            calls = 0
            def close(self):
                self.calls += 1
        ds = self.dataset()
        handle, cache = Handle(), ds._cache
        ds.h5_file = handle
        wrapper = SimpleNamespace(dataset=ds, ds=ds)
        self.assertEqual(self.mod.close_parent_hdf(wrapper), 1)
        self.assertEqual(handle.calls, 1)
        self.assertIsNone(ds.h5_file)
        self.assertIs(ds._cache, cache)
        self.assertEqual(self.mod.close_parent_hdf(wrapper), 0)


if __name__ == '__main__':
    unittest.main()
