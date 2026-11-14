# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_matrix_source.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
"""CPU contracts for an explicit shared-encoder M2 source config."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from matrix_source import select_source_assets


class MatrixSourceTests(unittest.TestCase):
    def runtime(self,cubes=1):
        return dict(model='lewm_pcd',mode='smoke',cube_count=cubes,camera={'cube_count':cubes},
                    official_data={'train':{'path':'real-cube'+str(cubes)}},
                    encoder_binding=dict(family='lewm',latent_dim=192,checkpoint='actual-source.pt',
                         checkpoint_sha256='a'*64,adapter={'path':'adapter','sha256':'b'*64},
                         preprocessing={'center':[.425,0,.2],'scale':4.,'use_rgb':True}),
                    encoder_receipt={'path':'source-receipt','sha256':'c'*64})

    def pin(self,root,value):
        path=Path(root)/'source.json';path.write_text(json.dumps(value),encoding='utf-8')
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_default_path_same_object_original_loader_only_no_pin_IO(self):
        target=self.runtime(3);calls=[];result_object=object()
        with patch.object(Path,'read_bytes',side_effect=AssertionError('unexpected pin I/O')):
            result=select_source_assets({},target,assets_loader=lambda value:(calls.append(value),result_object)[1])
        self.assertIs(calls[0],target);self.assertIs(result['source_runtime'],target)
        self.assertIs(result['assets'],result_object);self.assertFalse(result['provenance']['shared_encoder'])

    def test_default_resolver_is_existing_downstream_assets(self):
        target=self.runtime(2);result_object=object()
        with patch('jepa4r.policy.gciql.workflow_downstream._assets',return_value=result_object) as loader:
            result=select_source_assets({},target)
        loader.assert_called_once_with(target)
        self.assertIs(loader.call_args.args[0],target)
        self.assertIs(result['assets'],result_object)

    def test_pin_validates_real_source_and_preserves_target_fields(self):
        source,target=self.runtime(1),self.runtime(4);before=copy.deepcopy(target);calls=[]
        with tempfile.TemporaryDirectory()as root:
            pin=self.pin(root,source)
            result = select_source_assets(
                {'source_encoder_config': pin},
                target,
                assets_loader=lambda value: (calls.append(value), {'source-strict': True})[1],
            )
        self.assertEqual(calls,[source]);self.assertEqual(target,before)
        self.assertEqual(result['source_runtime'],source);self.assertEqual(result['provenance']['source_cube_count'],1)
        self.assertEqual(result['provenance']['target_cube_count'], 4)
        self.assertTrue(result['provenance']['shared_encoder'])

    def test_bad_source_pin_rejects_before_loader(self):
        with tempfile.TemporaryDirectory()as root:
            pin=self.pin(root,self.runtime());pin['sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'source config'):
                select_source_assets(
                    {'source_encoder_config': pin},
                    self.runtime(2),
                    assets_loader=lambda value: self.fail('loader called'),
                )
        with self.assertRaises(ValueError):
            select_source_assets(
                {'source_encoder_config': {}}, self.runtime(), assets_loader=lambda value: self.fail('loader called')
            )

    def test_wrong_family_mode_weight_preprocess_or_full_binding_rejects(self):
        changes = [
            ('model', 'lewm_rgb'),
            ('mode', 'formal'),
            ('family', 'pldm'),
            ('latent_dim', 512),
            ('checkpoint', 'other.pt'),
            ('checkpoint_sha256', 'd' * 64),
            ('preprocessing', {'scale': 8.0}),
            ('adapter', {'sha256': 'd' * 64}),
        ]
        for key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory()as root:
                source,target=self.runtime(),self.runtime(2)
                if key in ('model','mode'):target[key]=value
                else:target['encoder_binding'][key]=value
                with self.assertRaises(ValueError):
                    select_source_assets(
                        {'source_encoder_config': self.pin(root, source)}, target, assets_loader=lambda value: {}
                    )

    def test_target_receipt_or_import_cannot_be_relabelled(self):
        for field in ['encoder_receipt','encoder_import']:
            with self.subTest(field=field),tempfile.TemporaryDirectory()as root:
                source,target=self.runtime(),self.runtime(2);target[field]={'path':'made-up-target','sha256':'f'*64}
                with self.assertRaisesRegex(ValueError, field):
                    select_source_assets(
                        {'source_encoder_config': self.pin(root, source)}, target, assets_loader=lambda value: {}
                    )

    def test_original_source_assets_failure_propagates(self):
        with tempfile.TemporaryDirectory()as root:

            def rejected(value):
                self.assertEqual(value['cube_count'], 1)
                raise ValueError('original source producer/data/horizon invalid')
            with self.assertRaisesRegex(ValueError,'producer/data/horizon'):
                select_source_assets(
                    {'source_encoder_config': self.pin(root, self.runtime())}, self.runtime(3), assets_loader=rejected
                )


if __name__=='__main__':unittest.main()
