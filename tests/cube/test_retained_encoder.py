# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_retained_encoder.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns, unused imports removed)
"""Retained export relocation is a descriptor, never a rewritten WM PASS."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa4r.models.cube import retained_encoder as imports
from jepa4r.policy.gciql import workflow_downstream as workflow


class RetainedEncoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root / 'baseline'
        self.baseline.mkdir()
        (self.baseline/'lewm_pcd').mkdir()
        self.files = {}
        code = Path(workflow.__file__).resolve().parents[3]   # this repository; the moved files under their old names
        moved = {'utils.py': code/'jepa4r/data/clips.py',
                 'bench/ogbench/workflow_lewm_encoder.py': code/'jepa4r/models/cube/workflow_lewm_encoder.py'}
        for name in ('utils.py', 'bench/ogbench/workflow_lewm_encoder.py'):
            path = self.baseline/'lewm_pcd'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(moved[name].read_bytes())
            self.files['lewm_pcd/'+name] = workflow.sha(path)
        manifest = self.baseline/'SOURCE_MANIFEST.json'
        manifest.write_text(json.dumps(dict(schema='cube_paper_release_sources_v1', files=self.files)))
        model = self.root/'retained'
        model.mkdir()
        checkpoint = model/'weights_epoch_1.pt'
        checkpoint.write_bytes(b'synthetic weight identity, not a model test')
        cfg = model/'config.json'
        cfg.write_text(
            json.dumps(
                dict(
                    _target_='jepa.JEPA',
                    encoder=dict(_target_='encoders.DexArtPointNet', in_dim=6, out_dim=192),
                    action_encoder=dict(input_dim=25),
                )
            )
        )
        self.old = dict(schema='cube_fresh_wm_acceptance_v1', model='lewm_pcd', method='lewm_pcd',
                        cube_count=1, mode='smoke', actual_exit_code=0, actual_epochs=1, strict_reload_verified=True,
                        checkpoint=dict(path='/old/weights_epoch_1.pt', sha256=workflow.sha(checkpoint)),
                        config=dict(path='/old/config.json', sha256=workflow.sha(cfg)),
                        data_manifest=dict(path='/old/data_manifest.json', sha256='a'*64),
                        source_binding={'/old/utils.py': self.files['lewm_pcd/utils.py']},
                        producer_binding={'original': 'preserved'})
        receipt = self.root/'original_WM_ACCEPTANCE.json'
        receipt.write_text(json.dumps(self.old))
        self.original_bytes = receipt.read_bytes()
        evidence = self.root/'fixed_case.json'
        evidence.write_text(json.dumps(dict(actual_exit_code=0,case=dict(model='JEPA4R',cube_count=1),
                            comparison=dict(passed=True),identities=dict(checkpoint_sha256=workflow.sha(checkpoint)))))
        self.descriptor = imports.create_descriptor(self.root/'import.json', original_receipt=receipt,
            checkpoint=checkpoint, factory_config=cfg, baseline_root=self.baseline,
            compatibility_evidence=evidence, adapter=moved['bench/ogbench/workflow_lewm_encoder.py'],
            preprocessing_source=moved['utils.py'])
        self.config = dict(
            model='lewm_pcd',
            cube_count=1,
            mode='smoke',
            encoder_receipt=dict(path=str(receipt), sha256=workflow.sha(receipt)),
            encoder_binding=dict(
                family='lewm',
                latent_dim=192,
                epoch=1,
                checkpoint=str(checkpoint),
                checkpoint_sha256=workflow.sha(checkpoint),
                factory_config=dict(path=str(cfg), sha256=workflow.sha(cfg)),
                adapter=dict(
                    path=str(moved['bench/ogbench/workflow_lewm_encoder.py']),
                    sha256=self.files['lewm_pcd/bench/ogbench/workflow_lewm_encoder.py'],
                ),
                preprocessing=dict(
                    source_path=str(moved['utils.py']),
                    source_sha256=self.files['lewm_pcd/utils.py'],
                    device='cpu',
                    input_dtype='float32',
                    layout='XYZRGB1200',
                    center=[0.425, 0.0, 0.2],
                    scale=4.0,
                    use_rgb=True,
                ),
            ),
        )

    def test_import_preserves_original_receipt_and_checks_resolved_assets(self):
        result = imports.validate_import(self.descriptor, self.config)
        self.assertEqual(result, self.old)
        self.assertEqual(Path(self.config['encoder_receipt']['path']).read_bytes(), self.original_bytes)

    def test_wrong_weight_or_factory_or_case_is_rejected(self):
        for field in ('checkpoint_sha256','latent_dim'):
            wrong = copy.deepcopy(self.config)
            wrong['encoder_binding'][field] = 'b'*64 if field.endswith('sha256') else 512
            with self.assertRaises(ValueError):
                imports.validate_import(self.descriptor, wrong)
        wrong = copy.deepcopy(self.config)
        wrong['mode'] = 'formal'
        with self.assertRaises(ValueError):
            imports.validate_import(self.descriptor, wrong)

    def test_original_receipt_or_baseline_code_mutation_is_rejected(self):
        path = self.baseline/'lewm_pcd/utils.py'
        path.write_text('# changed\n')
        with self.assertRaises(ValueError):
            imports.validate_import(self.descriptor, self.config)

    def test_explicit_prepare_recipe_import_works_and_default_keeps_original_guard(self):
        official = {}
        for split in ('train', 'val'):
            path = self.root/(split+'.npz')
            path.write_bytes(b'synthetic identity-only source')
            official[split] = dict(path=str(path.resolve()),sha256=workflow.sha(path))
        camera = dict(width=224,height=224,aa_samples=4,position=[1.,0.,1.],source_position=[1.,0.,1.],
                      source_quaternion=[1.,0.,0.,0.],source_name='front_pixels',render_name='front_pixels',
                      fov=45.,local_x_up_degrees=0.,pullback_m=0.,pointcloud_crop_box=[[.1,-.5,-.02],[.75,.5,.6]])
        run = dict(model='lewm_pcd',cube_count=1,mode='smoke',official_data=official,camera=camera,
                   downstream_output_root=str((self.root/'downstream').resolve()),encoder_import=self.descriptor)
        receipt = self.config['encoder_receipt']['path']
        config = workflow.prepare_recipe(run, receipt, self.root/'imported_recipe.json')
        self.assertEqual(config['encoder_import'],self.descriptor)
        self.assertEqual(Path(receipt).read_bytes(),self.original_bytes)
        del run['encoder_import']
        with self.assertRaises(FileNotFoundError):
            workflow.prepare_recipe(run, receipt, self.root/'default_recipe.json')


if __name__ == '__main__':
    unittest.main()
