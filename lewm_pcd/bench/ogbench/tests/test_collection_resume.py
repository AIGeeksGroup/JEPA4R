"""Shard scheduling/recovery contracts; simulator and HDF QC are isolated."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.ogbench import collection
from bench.ogbench import workflow


class CollectionResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'collection'
        self.plan = collection.build_plan(1, self.root, mode='smoke')
        self.calls = []
        self.failed = set()
        self.patches = [
            patch.object(collection, 'materialize', self.materialize),
            patch.object(collection.subprocess, 'run', self.child),
            patch.object(collection, 'validate_pair', self.qc),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def materialize(self, root):
        root.mkdir(parents=True)
        files = {}
        for name in ('collect_cube_pcd.py', 'collection_protocol.py'):
            p = root / 'code/lewm_pcd/scripts/data' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# frozen test producer\n')
            files[str(p)] = collection.sha(p)
        return dict(files=files, reused_files={}, source_manifest_sha256='fixture')

    def child(self, command, **kwargs):
        values = dict(word.split('=', 1) for word in command if '=' in word)
        directory = Path(values['cache_dir'])
        index = int(values['world_seed'])
        self.calls.append((index, directory))
        if index in self.failed:
            (directory / 'failed-evidence.txt').write_text('preserve me')
            return SimpleNamespace(returncode=7)
        code = Path(kwargs['cwd']) / 'scripts/data'
        config = dict(task_id=int(values['fixed_task_id']), strategy=values['strategy'],
                      split=values['split'], world_seed=index, policy_seed=int(values['seed']),
                      horizon_actions=int(values['world.max_episode_steps']),
                      requested_episodes=int(values['num_traj']),
                      collector_sha256=collection.sha(code / 'collect_cube_pcd.py'),
                      protocol_sha256=collection.sha(code / 'collection_protocol.py'))
        manifest = dict(stage='complete', exit_code=0, completed=1,
                        pix_episodes_written=1, pcd_episodes_written=1, config=config)
        for name in ('pair_pix.h5', 'pair_pcd.h5'):
            p = directory / 'datasets' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(index))
            Path(str(p) + '.manifest.json').write_text(json.dumps(manifest))
        return SimpleNamespace(returncode=0)

    def qc(self, rgb, pcd, *, episodes, horizon, cube_count, world_seed):
        self.assertEqual(Path(rgb).read_text(), Path(pcd).read_text())
        return dict(episodes=episodes, rows=episodes * (horizon + 1),
                    reset_seeds=[world_seed], initial_state_sha256=[str(world_seed)],
                    actual_exit_code=0)

    def test_resume_only_missing_parts_keeps_completed_bytes(self):
        first = collection.run(self.plan, part_ids=[0])
        self.assertFalse(first['complete'])
        self.assertFalse((self.root / 'COLLECTION.json').exists())
        marker = self.root / 'shards/part_000/ACCEPTED.json'
        before = marker.read_bytes()
        second = collection.run(self.plan, resume=True, part_ids=[1])
        self.assertFalse(second['complete'])
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(len(self.calls), 2)

    def test_failed_attempt_is_preserved_and_retried_in_new_directory(self):
        collection.run(self.plan, part_ids=[0])
        seed = self.plan['parts'][1]['world_seed']
        self.failed.add(seed)
        with self.assertRaises(RuntimeError):
            collection.run(self.plan, resume=True, part_ids=[1])
        old = self.calls[-1][1]
        self.failed.clear()
        collection.run(self.plan, resume=True, part_ids=[1])
        new = self.calls[-1][1]
        self.assertNotEqual(old, new)
        self.assertEqual((old / 'failed-evidence.txt').read_text(), 'preserve me')
        self.assertEqual((old / 'actual_exit_code').read_text().strip(), '7')
        self.assertEqual(self.calls[-2][0], self.calls[-1][0])

    def test_explicit_completed_part_and_duplicate_ids_are_rejected(self):
        collection.run(self.plan, part_ids=[0])
        with self.assertRaisesRegex(ValueError, 'completed'):
            collection.run(self.plan, resume=True, part_ids=[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            collection.run(self.plan, resume=True, part_ids=[1, 1])
        self.assertEqual(len(self.calls), 1)

    def test_modified_completed_data_is_rejected_before_dispatch(self):
        collection.run(self.plan, part_ids=[0])
        Path(self.plan['parts'][0]['rgb']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed|checksum'):
            collection.run(self.plan, resume=True, part_ids=[1])
        self.assertEqual(len(self.calls), 1)

    def test_marker_cannot_redirect_rgb_away_from_hashed_files(self):
        collection.run(self.plan, part_ids=[0])
        marker = self.root / 'shards/part_000/ACCEPTED.json'
        saved = json.loads(marker.read_text())
        saved['part']['rgb'] = str(self.root / 'unverified.h5')
        marker.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'path|files'):
            collection.run(self.plan, resume=True, part_ids=[1])
        self.assertEqual(len(self.calls), 1)

    def test_unclosed_attempt_is_rejected_before_duplicate_dispatch(self):
        collection.run(self.plan, part_ids=[0])
        Path(self.plan['parts'][1]['directory']).mkdir()
        with self.assertRaisesRegex(FileExistsError, 'exit'):
            collection.run(self.plan, resume=True, part_ids=[1])
        self.assertEqual(len(self.calls), 1)

    def test_completed_receipt_cannot_rebind_unhashed_merged_path(self):
        collection.run(self.plan, part_ids=[0, 1])
        canonical = self.plan['parts'][2]
        actual = collection._attempt_part(canonical)
        directory = Path(actual['directory'])
        command = collection.child_command(self.plan, actual, self.root / 'runtime')
        (directory / 'config.json').write_text(json.dumps(dict(part=actual, command=command)))
        self.child(command, cwd=self.root / 'runtime/code/lewm_pcd')
        (directory / 'actual_exit_code').write_text('0\n')
        record = collection._part_acceptance(self.plan, actual, self.root / 'runtime')
        record.update(canonical_part=canonical, actual_exit_code=0)
        (directory / 'ACCEPTED.json').write_text(json.dumps(record))
        records = collection._accepted_parts(self.plan)
        data_files = {k: v for row in records.values() for k, v in row['data_files'].items()}
        (self.root / 'COLLECTION.json').write_text(json.dumps(dict(
            data_files=data_files, paths={'train_pcd': str(self.root / 'unverified.h5')})))
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError, 'receipt|manifest|checksum|binding'):
            collection.run(self.plan, resume=True)
        self.assertEqual(len(self.calls), count)

    def test_existing_root_still_requires_explicit_resume(self):
        collection.run(self.plan, part_ids=[0])
        with self.assertRaises(FileExistsError):
            collection.run(self.plan, part_ids=[1])

    def test_execution_order_preserves_canonical_ids_and_seeds(self):
        collection.run(self.plan, part_ids=[1, 0])
        self.assertEqual([seed for seed, _ in self.calls],
                         [self.plan['parts'][i]['world_seed'] for i in (1, 0)])
        for index in (0, 1):
            p = self.root / f'shards/part_{index:03d}/ACCEPTED.json'
            saved = json.loads(p.read_text())
            self.assertEqual(saved['canonical_part'], self.plan['parts'][index])
            self.assertEqual(saved['part']['policy_seed'], self.plan['parts'][index]['policy_seed'])

    def test_top_level_resume_retains_failed_dispatch_and_forwards_flag(self):
        cfg = dict(schema='cube_workflow_v1', model='lewm_pcd', cube_count=1,
                   mode='smoke', output_root=str(self.root / 'workflow'),
                   python=dict(world_model=sys.executable, downstream=sys.executable))
        plan = workflow.build_plan(cfg, 'collect')
        commands = []

        def child(command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                return SimpleNamespace(returncode=7)
            target = Path(plan['stages'][0]['receipt'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(dict(actual_exit_code=0, cube_count=1, mode='smoke')))
            return SimpleNamespace(returncode=0)

        with patch.object(workflow, 'verify_sources', lambda: {'test': True}), \
                patch.object(workflow.subprocess, 'run', child):
            self.assertEqual(workflow.execute(plan), 7)
            self.assertEqual(workflow.execute(plan, resume=True), 0)
        self.assertIn('--resume', commands[1])
        old_exit = Path(cfg['output_root']) / 'dispatch/collect/actual_exit_code'
        self.assertEqual(old_exit.read_text().strip(), '7')


if __name__ == '__main__':
    unittest.main()
