"""Narrow CPU-only source/contract tests, without encoder construction."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'reference_source/official_state_2d_cache_v1'
if not REFERENCE.is_dir():REFERENCE=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1')


def load(name):
    path=HERE/(name+'.py')
    assert path.is_file(),'new PLDM cache adapter is not implemented'
    spec=importlib.util.spec_from_file_location('_tested_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class CacheBridgeTests(unittest.TestCase):
    def test_counted_producer_overlay_preserves_render_sampling_and_writer(self):
        bridge=load('producer_bridge');raw=(REFERENCE/'produce.py').read_bytes()
        updated,edits=bridge.effective_source(raw)
        restored=updated
        for before,after,count in reversed(edits):
            self.assertEqual(restored.count(after),count);restored=restored.replace(after,before)
        self.assertEqual(restored,raw.decode())
        original={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
        effective={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(updated).body if isinstance(n,ast.FunctionDef)}
        for name in ['canonical_ids','writer_helpers','raw_stat','source_arrays','source_observation_rows']:
            self.assertEqual(original[name],effective[name])
        self.assertIn('latent_dim=512',updated);self.assertIn('(32, 3, 65, 65)',updated)
        self.assertIn("ProcessPoolExecutor(max_workers=2",updated)
        with self.assertRaises(AssertionError):bridge.effective_source(raw+b'\n')
        auditor=load('audit_bridge')
        audit_source=HERE/'reference_readonly/official_state_2d_cache_v1/audit_cache.py'
        if not audit_source.is_file():audit_source=REFERENCE/'audit_cache.py'
        old=audit_source.read_bytes()
        effective,changes=auditor.effective_source(old);restored=effective
        for before,after,count in reversed(changes):
            self.assertEqual(restored.count(after),count);restored=restored.replace(after,before)
        self.assertEqual(restored,old.decode());self.assertEqual(len(changes),4)
        ast.parse(effective)

    def test_uint8_layout_and_shared_native_preprocessing_selection(self):
        adapter=load('encoder_adapter');u8=object()
        class Tensor:
            def __init__(self,shape):self.shape=shape;self.ndim=len(shape);self.dtype=u8;self.contiguous_called=False
            def permute(self,*order):return Tensor(tuple(self.shape[i] for i in order))
            def contiguous(self):self.contiguous_called=True;return self
        with patch.dict('sys.modules',{'torch':types.SimpleNamespace(uint8=u8)}):
            tensor=adapter.as_bchw(Tensor((32,224,224,3)))
            self.assertEqual(tensor.shape,(32,3,224,224));self.assertTrue(tensor.contiguous_called)
            with self.assertRaises(ValueError):adapter.as_bchw(Tensor((32,3,65,65)))
        self.assertEqual((adapter.FAMILY,adapter.DIM),('pldm',512))
        # An absent choice must not silently use a different online preprocessor.
        self.assertEqual(__import__('inspect').signature(adapter.encode_uint8).parameters['preprocess_device'].default,__import__('inspect').Parameter.empty)


if __name__=='__main__':unittest.main()
