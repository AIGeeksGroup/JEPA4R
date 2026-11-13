# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_rh56_workflow.py (verbatim except: module names / paths
# of the moved code adapted; lines wrapped at 120 columns)
"""Dependency-free contracts for the final RH56 workflow dispatcher."""
import importlib.util
import hashlib
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'jepa4r/policy/bc/relocate/rh56_workflow.py'


class RH56WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(FILE.is_file(), 'RH56 workflow dispatcher is not implemented')
        spec = importlib.util.spec_from_file_location('rh56_workflow_tested', FILE)
        self.workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.workflow)

    def args(self, out, **changes):
        values = dict(env='rh56-relocate', model='jepa4r', mode='smoke', output=Path(out),
                      wm_python='wm-python', sim_python='sim-python', seed=42)
        values.update(changes)
        return SimpleNamespace(**values)

    def assets(self):
        return dict(rh56_expert_root='/external/expert', eai_vc_root='/external/eai',
                    rh56_official_task_init='/external/official.json')

    def test_plan_is_pure_and_uses_cuda_and_final_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new-output'
            stages = self.workflow.plan(self.args(out), self.assets())
            self.assertFalse(out.exists())
            self.assertEqual(len({s['name'] for s in stages}), len(stages))
            collect = next(s for s in stages if s['name'].startswith('rh56_collect_'))
            self.assertEqual(collect['command'][0], 'sim-python')
            self.assertEqual(collect['command'][collect['command'].index('--camera') + 1], 'opposite')
            wm = next(s for s in stages if s['name'] == 'rh56_wm')
            self.assertIn('cuda', wm['command'])
            self.assertIn('--batch-size', wm['command'])
            bc = next(s for s in stages if s['name'] == 'rh56_bc_100')
            self.assertEqual(bc['command'][bc['command'].index('--mode') + 1], 'profile')
            self.assertNotIn('--uint8-gpu-preprocess', wm['command'])
            for stage in stages:
                self.assertIsInstance(stage['command'], list)
                self.assertTrue(all(isinstance(v, str) for v in stage['command']))
                self.assertIsInstance(stage['environment'], dict)
                self.assertTrue(Path(stage['cwd']).is_dir())

    def test_model_switch_changes_modality_without_changing_protocol(self):
        pcd = self.workflow.plan(self.args('/output'), self.assets())
        rgb = self.workflow.plan(self.args('/output', model='lewm'), self.assets())
        self.assertEqual([s['name'] for s in pcd], [s['name'] for s in rgb])
        for stages, route in ((pcd, 'pcd'), (rgb, 'rgb')):
            for s in stages:
                if '--route' in s['command']:
                    self.assertEqual(s['command'][s['command'].index('--route') + 1], route)
        with self.assertRaises(ValueError):
            self.workflow.plan(self.args('/output', model='pldm'), self.assets())
        with self.assertRaises(ValueError):
            self.workflow.plan(self.args('/output', mode='typo'), self.assets())

    def test_full_has_bounded_batches_original_budget_and_all_head_seeds(self):
        stages = self.workflow.plan(self.args('/output', mode='full'), self.assets())
        self.assertEqual(len([s for s in stages if s['name'].startswith('rh56_collect_')]), 40)
        self.assertEqual([s['name'] for s in stages if s['name'].startswith('rh56_bc_')],
                         ['rh56_bc_100', 'rh56_bc_200', 'rh56_bc_300'])
        wm = next(s for s in stages if s['name'] == 'rh56_wm')
        self.assertEqual(wm['command'][wm['command'].index('--mode') + 1], 'formal')
        self.assertEqual(wm['command'][wm['command'].index('--seed') + 1], '3072')

    def test_validation_does_not_accept_files_without_numerical_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            stages = self.workflow.plan(args, self.assets())
            for stage in stages:
                self.assertFalse(self.workflow.verify_stage(stage, args, self.assets())['passed'])
            p = Path(tmp) / 'rh56/wm'
            p.mkdir(parents=True)
            (p / 'state.json').write_text(json.dumps({'phase': 'complete', 'exit_code': 0}))
            (p / 'validation.json').write_text(json.dumps({'status': 'passed'}))
            wm = next(s for s in stages if s['name'] == 'rh56_wm')
            self.assertFalse(self.workflow.verify_stage(wm, args, self.assets())['passed'])

    def test_unknown_stage_is_rejected(self):
        report = self.workflow.verify_stage({'name': 'unrecognised'}, self.args('/output'), self.assets())
        self.assertFalse(report['passed'])

    def test_prepare_preserves_all_official_sources_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'source-manifest.json'
            manifest.write_text('{"source":"official100"}')
            source = root / 'initials.json'
            payload = dict(
                schema='official-task-init-v1',
                source_manifest='/old/location.json',
                source_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                records=[
                    dict(
                        demo_index=i,
                        source_id=str(i),
                        original_init_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                        obj_pos=[0.0, 0.0, 0.1],
                        target_pos=[0.0, 0.1, 0.2],
                    )
                    for i in range(100)
                ],
            )
            semantic = [
                {k: r[k] for k in ('demo_index', 'original_init_sha256', 'obj_pos', 'target_pos')}
                for r in payload['records']
            ]
            # Synthetic positives patch only this test module's trusted constant.
            self.workflow.OFFICIAL_TASK_CONTENT_SHA256 = hashlib.sha256(
                json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()
            ).hexdigest()
            source.write_text(json.dumps(payload))
            original = source.read_bytes()
            assets = dict(self.assets(), rh56_official_task_init=str(source),
                          rh56_official_source_manifest=str(manifest))
            args = self.args(root / 'output')
            self.workflow.prepare(dict(mode='smoke', output=str(args.output / 'rh56'), assets=assets))
            self.assertEqual(source.read_bytes(), original)
            inputs = args.output / 'rh56/inputs'
            copied = json.loads((inputs / 'official_task_init.json').read_text())
            self.assertEqual(copied['records'], payload['records'])
            self.assertEqual(copied['source_manifest'], str(manifest.resolve()))
            rows = json.loads((inputs / 'plan_000.json').read_text())['episodes']
            self.assertEqual([r['split'] for r in rows], ['train', 'train', 'val'])
            self.assertEqual(len({r['seed'] for r in rows}), 3)
            stage = self.workflow.plan(args, assets)[0]
            self.assertTrue(self.workflow.verify_stage(stage, args, assets)['passed'])
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = self.workflow.main(['official-demo', '--inputs', str(inputs),
                    'official-init', '--expert-root', assets['rh56_expert_root'], '--dry-run',
                    '--init-source', str(inputs / 'official_task_init.json'), '--count', '1',
                    '--output', str(args.output / 'unused-demo')])
            self.assertEqual(result, 0)
            forwarded = json.loads(captured.getvalue())['argv']
            actual_sha = hashlib.sha256((inputs / 'official_task_init.json').read_bytes()).hexdigest()
            self.assertEqual(forwarded[forwarded.index('--init-source-sha256') + 1], actual_sha)
            self.assertFalse((args.output / 'unused-demo').exists())
            # A forged replacement receipt cannot turn invented positions into
            # the canonical official task content, even with fresh file hashes.
            copied['records'][0]['obj_pos'] = [99.,99.,99.]
            (inputs / 'official_task_init.json').write_text(json.dumps(copied))
            report = json.loads((inputs / 'input_receipt.json').read_text())
            for item in report['files']:
                if Path(item['path']).name == 'official_task_init.json':
                    item['sha256'] = hashlib.sha256((inputs / 'official_task_init.json').read_bytes()).hexdigest()
            (inputs / 'input_receipt.json').write_text(json.dumps(report))
            verification = self.workflow.verify_stage(stage, args, assets)
            self.assertFalse(verification['passed'])
            self.assertIn('canonical', verification['details'])
            (inputs / 'official_task_init.json').write_text('{}')
            self.assertFalse(self.workflow.verify_stage(stage, args, assets)['passed'])
            with self.assertRaises(FileExistsError):
                self.workflow.prepare(dict(mode='smoke', output=str(args.output / 'rh56'), assets=assets))

    def test_repeated_initialization_hashes_are_rejected_even_with_test_trust_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source = root / 'manifest.json', root / 'source.json'
            manifest.write_text('{}')
            records=[dict(demo_index=i,source_id=str(i),original_init_sha256='a'*64,
                          obj_pos=[0.,0.,.1],target_pos=[0.,.1,.2]) for i in range(100)]
            semantic=[{k:r[k] for k in ('demo_index','original_init_sha256','obj_pos','target_pos')} for r in records]
            self.workflow.OFFICIAL_TASK_CONTENT_SHA256 = hashlib.sha256(
                json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()
            ).hexdigest()
            source.write_text(json.dumps(dict(schema='official-task-init-v1',source_manifest=str(manifest),
                              source_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(),records=records)))
            with self.assertRaisesRegex(ValueError,'duplicate.*hash'):
                self.workflow.prepare(dict(mode='smoke',output=str(root/'out'),
                    assets=dict(self.assets(),rh56_official_task_init=str(source))))
            self.assertFalse((root/'out').exists())

    def test_partial_official_sources_are_rejected_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source = root / 'manifest.json', root / 'source.json'
            manifest.write_text('{}')
            source.write_text(json.dumps(dict(schema='official-task-init-v1', source_manifest=str(manifest),
                 source_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(), records=[])))
            assets = dict(self.assets(), rh56_official_task_init=str(source))
            with self.assertRaisesRegex(ValueError, 'all100'):
                self.workflow.prepare(dict(mode='smoke', output=str(root / 'output'), assets=assets))
            self.assertFalse((root / 'output').exists())

    def test_default_trust_anchor_rejects_invented_official100_and_asset_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source = root / 'manifest.json', root / 'source.json'
            manifest.write_text('{"source":"official100"}')
            records = [dict(demo_index=i, source_id=str(i),
                           original_init_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                           obj_pos=[99., 99., 99.], target_pos=[99., 99., 99.]) for i in range(100)]
            source.write_text(json.dumps(dict(schema='official-task-init-v1', source_manifest=str(manifest),
                source_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(), records=records)))
            semantic = [{k:r[k] for k in ('demo_index','original_init_sha256','obj_pos','target_pos')} for r in records]
            assets = dict(
                self.assets(),
                rh56_official_task_init=str(source),
                expected_official_digest=hashlib.sha256(
                    json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, 'canonical|official.*content'):
                self.workflow.prepare(dict(mode='smoke', output=str(root / 'output'), assets=assets))
            self.assertFalse((root / 'output').exists())

    def test_simulator_flag_overrides_sim_asset_and_explicitly_selected_bc_python(self):
        assets = dict(self.assets(), rh56_sim_python='asset-sim', rh56_bc_python='asset-bc')
        default = self.workflow.plan(self.args('/output'), assets)
        self.assertEqual(default[0]['command'][0], 'sim-python')
        self.assertEqual(next(s for s in default if s['name']=='rh56_bc_100')['command'][0], 'asset-bc')
        explicit = self.workflow.plan(
            self.args('/output', sim_python='explicit-python', sim_python_override=True), assets
        )
        for stage in explicit:
            if stage['name'] != 'rh56_wm':
                self.assertEqual(stage['command'][0], 'explicit-python')

    def test_wm_receipt_requires_real_updates_finite_losses_and_bound_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            base = Path(tmp) / 'rh56'
            directory = base / 'wm'
            directory.mkdir(parents=True)
            (base / 'data').mkdir()
            for path in [base / 'data/manifest.json', base / 'data/paired_pcd.h5', directory / 'weights.pt']:
                path.write_bytes(b'receipt contract fixture')
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (directory / 'state.json').write_text(json.dumps(dict(phase='complete', exit_code=0)))
            (directory / 'run_identity.json').write_text(json.dumps(dict(device='cuda')))
            report = dict(status='passed', route='pcd', evidence=dict(optimizer_steps=2,
                          changed_parameter_tensors=1, gradient_checks=2, train_loss=.1,
                          epoch_validation_losses=[.2], epochs=[1], train_batches=2, val_batches=1),
                          checkpoint_epoch=1, strict_reload_maxabs={'emb': 0., 'loss': 0.},
                          actual_adam_step_values=[2], checkpoint=str(directory / 'weights.pt'),
                          checkpoint_sha256=sha(directory / 'weights.pt'),
                          manifest_sha256=sha(base / 'data/manifest.json'),
                          dataset_sha256=sha(base / 'data/paired_pcd.h5'))
            receipt = directory / 'validation.json'
            receipt.write_text(json.dumps(report))
            stage = next(s for s in self.workflow.plan(args, self.assets()) if s['name'] == 'rh56_wm')
            self.assertTrue(self.workflow.verify_stage(stage, args, self.assets())['passed'])
            report['evidence']['train_loss'] = float('nan')
            receipt.write_text(json.dumps(report))
            self.assertFalse(self.workflow.verify_stage(stage, args, self.assets())['passed'])
            report['evidence']['train_loss'] = .1
            receipt.write_text(json.dumps(report))
            (directory / 'weights.pt').write_bytes(b'changed')
            self.assertFalse(self.workflow.verify_stage(stage, args, self.assets())['passed'])

    def test_bc_gate_requires_cuda_job_and_actual_rollouts_not_only_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            base = Path(tmp) / 'rh56'
            directory = base / 'bc_seed100'
            paths = [directory / 'policy/iterations/best_policy.pickle',
                     directory / 'policy/iterations/final_policy.pickle',
                     base / 'demos/bc_demos_manifest.json', base / 'checkpoint/weights_epoch_1.pt']
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'receipt fixture')
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (directory / 'state.json').write_text(json.dumps(dict(phase='complete', exit_code=0)))
            report = dict(status='PASS_MINIMAL_OFFICIAL_BC_CHAIN', route='pcd',
                          run_spec=dict(camera='opposite', epochs=1, num_demos=1), train_seed=100,
                          input_dim=589, action_dim=12, encoder_frozen_and_unchanged=True,
                          head_updated=True, finite_head_gradients=True,
                          optimizer_evidence=dict(steps=1, finite_gradient_checks=1),
                          actual_online_episodes=1, actual_train_initial_episodes=1, online_steps=200,
                          policy_reload_max_abs=0., loss=.1,
                          policy_sha256=sha(paths[0]), final_policy_sha256=sha(paths[1]),
                          demos_manifest_sha256=sha(paths[2]), checkpoint=str(base / 'checkpoint'),
                          checkpoint_epoch=1, checkpoint_sha256=sha(paths[3]))
            receipt = directory / 'validation.json'
            receipt.write_text(json.dumps(report))
            stage = next(s for s in self.workflow.plan(args, self.assets()) if s['name'] == 'rh56_bc_100')
            self.assertFalse(self.workflow.verify_stage(stage, args, self.assets())['passed'])
            (directory / 'job_config.json').write_text(json.dumps(dict(device='cuda', epochs=1, num_demos=1)))
            self.assertTrue(self.workflow.verify_stage(stage, args, self.assets())['passed'])
            report['actual_online_episodes'] = 0
            receipt.write_text(json.dumps(report))
            self.assertFalse(self.workflow.verify_stage(stage, args, self.assets())['passed'])


if __name__ == '__main__':
    unittest.main()
