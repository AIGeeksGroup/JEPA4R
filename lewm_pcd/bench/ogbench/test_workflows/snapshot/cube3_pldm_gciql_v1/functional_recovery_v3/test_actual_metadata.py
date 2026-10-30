"""Use the exact closed-cache JSON bytes; no invented audit fields or payload IO."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOCAL_CASE = HERE.parent
SNAPSHOT = Path(os.environ.get('P3_ACTUAL_METADATA_ROOT',str(HERE/'actual_metadata')))
REMOTE_CASE = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube3_pldm_gciql_v1')

import prepare as recovery


def fixture():
    receipt = json.loads((HERE/'actual_metadata_receipt.json').read_bytes())
    assert receipt['actual_exit_code'] == 0
    for relative, expected in receipt['files'].items():
        blob = (SNAPSHOT/relative).read_bytes()
        assert len(blob) == expected['bytes'] and hashlib.sha256(blob).hexdigest() == expected['sha256']
    module = recovery.load_original(REMOTE_CASE, LOCAL_CASE/'functional_v1')
    def mapped(path):
        return SNAPSHOT/Path(path).relative_to(REMOTE_CASE)
    def read(path, records, digest=None):
        path = Path(path)
        actual = mapped(path)
        before = recovery.file_stat(actual)
        blob = actual.read_bytes()
        assert recovery.file_stat(actual) == before
        value = hashlib.sha256(blob).hexdigest()
        assert digest is None or value == digest
        records[str(path)] = dict(sha256=value,stat=before)
        return json.loads(blob)
    module.read = read
    module.sha = lambda path: recovery.sha(mapped(path))
    original_read_text = Path.read_text
    def read_text(path, *args, **kwargs):
        if Path(path).is_relative_to(REMOTE_CASE) and mapped(path).is_file():
            path = mapped(path)
        return original_read_text(path,*args,**kwargs)
    return module, read_text


class ActualMetadataTests(unittest.TestCase):
    def test_exact_real_schema_reproduces_old_KeyError_then_all_cache_metadata_pass(self):
        module, read_text = fixture()
        report_path = SNAPSHOT/'cache_v1/independent_audit/report.json'
        before = recovery.sha(report_path)
        audit = json.loads(report_path.read_bytes())
        self.assertNotIn('checkpoint_sha256',audit)
        with patch.object(Path,'read_text',read_text):
            with self.assertRaisesRegex(KeyError,'checkpoint_sha256'):
                module.closed_cache(REMOTE_CASE,{})
            proof = recovery.install_checkpoint_binding(module,REMOTE_CASE)
            facts = module.closed_cache(REMOTE_CASE,{})
        self.assertTrue(proof['inverse_function_source_exact'])
        self.assertEqual(facts['selected_device'],'cpu')
        template = json.loads((LOCAL_CASE/'functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE_v2.json').read_bytes())
        cfg = module.config_from_cache(template,facts,REMOTE_CASE)
        self.assertEqual((cfg['input_contract']['current_dim'],cfg['input_contract']['goal_dim']),(531,521))
        self.assertEqual((cfg['batch_size'],cfg['total_updates'],len(cfg['checkpoint_steps'])),(1024,1000000,11))
        self.assertEqual(cfg['encoder_binding']['selected_preprocess_device'],'cpu')
        self.assertEqual(recovery.sha(report_path),before)
        self.assertNotIn('checkpoint_sha256',json.loads(report_path.read_bytes()))

    def test_wrong_real_plan_manifest_or_checkpoint_identity_is_rejected(self):
        module, read_text = fixture()
        recovery.install_checkpoint_binding(module,REMOTE_CASE)
        original_read = module.read
        for target,key in [('report.json','plan_sha256'),('report.json','manifest_sha256'),
                           ('report.json','producer_seal_sha256'),('producer_plan.json','checkpoint_sha256')]:
            def tampered(path,records,digest=None):
                value=original_read(path,records,digest)
                if Path(path).name==target:
                    value=copy.deepcopy(value);value[key]='0'*64
                return value
            module.read=tampered
            with patch.object(Path,'read_text',read_text),self.subTest(field=key),self.assertRaises(AssertionError):
                module.closed_cache(REMOTE_CASE,{})
        module.read=original_read


if __name__=='__main__':
    unittest.main()
