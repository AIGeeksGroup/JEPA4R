"""Only the data and prefetch seams surround the original Tiny loop."""
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "lewm_pcd"))


class TinyTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("bench.ogbench.tiny")
        except ModuleNotFoundError:
            self.fail("Tiny adapter has not been implemented")

    def test_canonical_source_unchanged(self):
        self.module()
        path = ROOT / "lewm_pcd/scripts/baselines/train_lejepa_mw.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51")

    def test_prefetch_changes_only_requested_loader_option(self):
        tiny = self.module()
        class Loader:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
        canonical = SimpleNamespace(DataLoader=Loader, swm=object())
        with patch.object(tiny, "_load_canonical", return_value=canonical), \
             patch.object(tiny, "validate_manifest"), \
             patch.object(tiny, "validate_view"):
            module = tiny.native_module("single200", Path("/sealed"), prefetch=2)
        loader = module.DataLoader("dataset", batch_size=256, persistent_workers=True, prefetch_factor=4)
        self.assertEqual(loader.args, ("dataset",))
        self.assertEqual(loader.kwargs, dict(batch_size=256, persistent_workers=True, prefetch_factor=2))

    def test_native_swm_loader_is_preserved_after_train_view_guard(self):
        tiny = self.module()
        class Loader:
            pass
        swm = object()
        canonical = SimpleNamespace(DataLoader=Loader, swm=swm)
        with patch.object(tiny, "_load_canonical", return_value=canonical), \
             patch.object(tiny, "validate_manifest") as validate, \
             patch.object(tiny, "validate_view", create=True) as view_guard:
            module = tiny.native_module("double500", Path("/sealed"))
            self.assertIs(module.DataLoader, Loader)
            self.assertIs(module.swm, swm)
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[1], "double500")
        self.assertEqual(view_guard.call_args.args[1], "double500")
        self.assertEqual(view_guard.call_args.args[2], "train")
        self.assertEqual(view_guard.call_args.args[0],
                         Path("/sealed/data_views/2d_train_action_rows500.h5"))


if __name__ == "__main__":
    unittest.main()
