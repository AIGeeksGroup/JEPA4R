"""Focused durability checks; temporary byte fixtures never deserialize pickle."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('backup_copy'), 'backup_copy missing')
        self.copy = importlib.import_module('backup_copy')
        self.watch = importlib.import_module('backup_watch')
        self.gate = importlib.import_module('backup_gate')
        fixture_root = Path(__file__).resolve().parents[2]/'backup_test_work'
        fixture_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='backup_fixture_', dir=fixture_root)
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.cfg_path = self.home/'pilot_config.json'
        self.output = self.home/'pilot'/'gates'/'vjepa21_vitl1024_cube1'
        self.durability = dict(group_id='vjepa21_vitl1024_cube1',
            pilot_output=str(self.output), formal_output=str(self.home/'runs'/'vjepa21_vitl1024_cube1'),
            ack_root=str(self.home/'acks'), failure_root=str(self.home/'failures'))
        self.cfg_path.write_text(json.dumps(dict(group_id=self.durability['group_id'], purpose='pilot',
            execution_root=str(self.home/'pilot'),
            output=str(self.home/'pilot'/'runs'/self.durability['group_id']),
            durability=self.durability)), encoding='utf-8')
        for counter in (1, 2):
            directory = self.output/'checkpoints'/f'update_{counter:09d}'
            directory.mkdir(parents=True)
            params, sidecar = b'not a pickle; params', b'not a pickle; sidecar'
            (directory/f'params_{counter}.pkl').write_bytes(params)
            (directory/'run_state.pkl').write_bytes(sidecar)
            (directory/'manifest.json').write_text(json.dumps(dict(counter=counter,
                agent_sha256=hashlib.sha256(params).hexdigest(),
                sidecar_sha256=hashlib.sha256(sidecar).hexdigest())), encoding='utf-8')
        self.backup_cfg = dict(config_paths=[str(self.cfg_path)], case_root=str(self.home))
        self.remote = FakeTransport(self.watch, self.backup_cfg)

    def snapshot(self, counter=1):
        return self.watch.checkpoint_snapshot(self.cfg_path, 'pilot', counter)

    def test_first_copy_hashes_bytes_then_ack_and_identical_skip(self):
        snapshot = self.snapshot()
        result = self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        receipt = json.loads(Path(result['receipt_path']).read_text())
        self.assertEqual(receipt['actual_exit_code'], 0)
        self.assertEqual(self.remote.actions, ['poststat', 'ack'])
        self.assertEqual(len(self.remote.copied), 4)
        copied_before = list(self.remote.copied)
        self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        self.assertEqual(self.remote.copied, copied_before)
        self.assertTrue((self.home/'acks'/'pilot'/self.durability['group_id']/'update_000000001.json').exists())

    def test_partial_directory_is_preserved_and_never_acknowledged(self):
        snapshot = self.snapshot()
        destination = self.home/'D'/Path(*snapshot['target_parts'])
        destination.mkdir(parents=True)
        (destination/'params_1.pkl').write_bytes(b'partial')
        with self.assertRaisesRegex(RuntimeError, 'partial|unaccepted'):
            self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        self.assertEqual((destination/'params_1.pkl').read_bytes(), b'partial')
        self.assertNotIn('ack', self.remote.actions)
        self.assertTrue((self.home/'failures'/'global.json').exists())

    def test_hash_mismatch_fails_preserves_copy_and_never_acknowledges(self):
        self.remote.corrupt = True
        snapshot = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'hash|SHA'):
            self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        self.assertNotIn('ack', self.remote.actions)
        self.assertTrue((self.home/'D'/Path(*snapshot['target_parts'])/'failure.json').exists())

    def test_copy_exit_nonzero_and_source_mutation_never_acknowledge(self):
        self.remote.copy_exit = 7
        with self.assertRaisesRegex(RuntimeError, 'SCP|copy'):
            self.copy.copy_snapshot(self.snapshot(), self.home/'D', self.remote)
        self.assertNotIn('ack', self.remote.actions)
        self.remote.copy_exit = 0
        self.remote.mutate_after = True
        with self.assertRaisesRegex(RuntimeError, 'source|changed'):
            self.copy.copy_snapshot(self.snapshot(2), self.home/'D', self.remote)
        self.assertNotIn('ack', self.remote.actions)

    def test_existing_accepted_copy_cannot_be_silently_replaced(self):
        snapshot = self.snapshot()
        result = self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        copied = Path(result['receipt_path']).parent/'params_1.pkl'
        copied.write_bytes(b'tampered data')
        count = len(self.remote.copied)
        with self.assertRaisesRegex(RuntimeError, 'changed|size|stat'):
            self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        self.assertEqual(len(self.remote.copied), count)
        self.assertEqual(copied.read_bytes(), b'tampered data')

    def test_gate_requires_both_bound_acknowledgments_and_rejects_sticky_failure(self):
        self.assertIsNone(self.gate.check_once(self.cfg_path, 'gate'))
        for counter in (1, 2):
            self.copy.copy_snapshot(self.snapshot(counter), self.home/'D', self.remote)
        report = self.gate.check_once(self.cfg_path, 'gate')
        self.assertEqual(report['checkpoint_count'], 2)
        ack_path = self.home/'acks'/'pilot'/self.durability['group_id']/'update_000000001.json'
        ack = json.loads(ack_path.read_text())
        ack['config_sha256'] = '0'*64
        ack_path.write_text(json.dumps(ack))
        with self.assertRaisesRegex(RuntimeError, 'config|binding'):
            self.gate.check_once(self.cfg_path, 'gate')
        self.watch.write_failure(self.backup_cfg, dict(error='deliberate fixture failure'))
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            self.gate.check_once(self.cfg_path, 'gate')

    def test_only_drained_closed_phase_logs_are_copied_including_nonzero_exit(self):
        control = self.home/'control'/'smoke'
        runtime = control/'runtime'
        runtime.mkdir(parents=True)
        name = 'cube1_02_gate'
        (control/'launch_plan.json').write_text(json.dumps(dict(phases=[dict(name=name,
            argv=['python', 'runtime.py', 'gate', '--config', str(self.cfg_path), '--output', str(self.output)])])))
        (runtime/f'{name}_argv.json').write_text('{}')
        (runtime/f'{name}.log').write_text('failed evidence\n')
        exit_path = runtime/f'{name}_exit.json'
        exit_path.write_text(json.dumps(dict(actual_exit_code=5, residual_owned_members=[dict(pid=123)])))
        self.assertEqual(self.watch.scan(self.backup_cfg)['logs'], [])
        exit_path.write_text(json.dumps(dict(actual_exit_code=5, residual_owned_members=[])))
        jobs = self.watch.scan(self.backup_cfg)['logs']
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['phase_exit_code'], 5)
        result = self.copy.copy_snapshot(jobs[0], self.home/'D', self.remote)
        self.assertTrue(Path(result['receipt_path']).exists())
        self.assertNotIn('ack', self.remote.actions)

    def test_native_snapshot_remains_stable_when_later_audit_report_appears(self):
        control = self.home/'control'/'smoke'
        runtime = control/'runtime'
        runtime.mkdir(parents=True)
        output = self.home/'native_sr'
        output.mkdir()
        (output/'report.json').write_text('{}')
        name = 'cube1_05_native'
        (control/'launch_plan.json').write_text(json.dumps(dict(phases=[dict(name=name,
            argv=['python', 'native_sr.py', '--config', str(self.cfg_path), '--output', str(output)])])))
        (runtime/f'{name}_argv.json').write_text('{}')
        (runtime/f'{name}.log').write_text('done')
        (runtime/f'{name}_exit.json').write_text(json.dumps(dict(actual_exit_code=0, residual_owned_members=[])))
        before = self.watch.phase_snapshot(self.backup_cfg, 'smoke', name)
        (output/'independent_audit.json').write_text('{}')
        after = self.watch.phase_snapshot(self.backup_cfg, 'smoke', name)
        self.assertEqual(before, after)

    def test_metadata_over_16mib_is_rejected_before_copy(self):
        control = self.home/'control'/'smoke'
        runtime = control/'runtime'
        runtime.mkdir(parents=True)
        name = 'cube1_02_gate'
        (control/'launch_plan.json').write_text(json.dumps(dict(phases=[dict(name=name, argv=['python'])])))
        (runtime/f'{name}_argv.json').write_text('{}')
        with (runtime/f'{name}.log').open('wb') as handle:
            handle.truncate((16 << 20) + 1)
        (runtime/f'{name}_exit.json').write_text(json.dumps(dict(actual_exit_code=0, residual_owned_members=[])))
        with self.assertRaisesRegex(RuntimeError, '16MiB'):
            self.watch.phase_snapshot(self.backup_cfg, 'smoke', name)

    def test_formal_gate_requires_all_eleven_config_bound_checkpoints(self):
        cfg = json.loads(self.cfg_path.read_text())
        cfg['purpose'] = 'formal'
        cfg['execution_root'] = str(self.home)
        cfg['output'] = self.durability['formal_output']
        self.cfg_path.write_text(json.dumps(cfg))
        output = Path(self.durability['formal_output'])
        for counter in self.watch.COUNTERS['formal']:
            directory = output/'checkpoints'/f'update_{counter:09d}'
            directory.mkdir(parents=True)
            params, sidecar = b'formal params', b'formal sidecar'
            (directory/f'params_{counter}.pkl').write_bytes(params)
            (directory/'run_state.pkl').write_bytes(sidecar)
            (directory/'manifest.json').write_text(json.dumps(dict(counter=counter,
                agent_sha256=hashlib.sha256(params).hexdigest(), sidecar_sha256=hashlib.sha256(sidecar).hexdigest())))
            snapshot = self.watch.checkpoint_snapshot(self.cfg_path, 'formal', counter)
            self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        self.assertEqual(self.gate.check_once(self.cfg_path, 'train')['checkpoint_count'], 11)

    def test_cli_small_backup_timeout_records_elapsed_and_missing_progress(self):
        report = self.home/'timeout_report.json'
        result = subprocess.run([sys.executable, '-B', str(Path(self.gate.__file__)),
            '--config', str(self.cfg_path), '--stage', 'gate', '--report', str(report), '--timeout', '0.01'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        evidence = json.loads(report.read_text())
        self.assertEqual(evidence['actual_exit_code'], 1)
        self.assertEqual(evidence['timeout_seconds'], 0.01)
        self.assertGreaterEqual(evidence['waited_seconds'], 0.01)
        self.assertEqual(evidence['progress']['missing_ack_counters'], [1, 2])

    def test_config_output_binding_and_exact_five_durability_fields_are_required(self):
        cfg = json.loads(self.cfg_path.read_text())
        cfg['durability']['pilot_output'] = str(self.home/'wrong_gate')
        self.cfg_path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(RuntimeError, 'output|binding'):
            self.snapshot()
        cfg['durability'] = dict(self.durability, unexpected=True)
        self.cfg_path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(RuntimeError, 'five|fields'):
            self.snapshot()

    def test_accepted_receipt_missing_file_or_copy_record_is_rejected(self):
        snapshot = self.snapshot()
        result = self.copy.copy_snapshot(snapshot, self.home/'D', self.remote)
        receipt_path = Path(result['receipt_path'])
        original = json.loads(receipt_path.read_text())
        altered = json.loads(json.dumps(original))
        del altered['local_files']['run_state.pkl']
        receipt_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(RuntimeError, 'file|set'):
            self.copy.accepted_receipt(snapshot, receipt_path)
        altered = json.loads(json.dumps(original))
        altered['copy_records'].pop()
        receipt_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(RuntimeError, 'SCP|record|count'):
            self.copy.accepted_receipt(snapshot, receipt_path)

    def test_stop_is_optional_and_requires_exactly_52_unique_accepted_checkpoints(self):
        self.assertTrue(hasattr(self.copy, 'completion_requested'), 'graceful completion helper missing')
        runtime = self.home/'backup_runtime'
        runtime.mkdir()
        expected = {(stage, f'vjepa21_vitl1024_cube{cube}', counter)
            for stage, counters in self.watch.COUNTERS.items()
            for cube in range(1, 5) for counter in counters}
        self.assertEqual(len(expected), 52)
        self.assertFalse(self.copy.completion_requested(runtime, expected))
        stop = runtime/'STOP.json'
        stop.write_text(json.dumps(dict(reason='task_complete', expected_checkpoint_count=52)))
        with self.assertRaisesRegex(RuntimeError, '52|incomplete'):
            self.copy.completion_requested(runtime, set(list(expected)[:-1]))
        self.assertTrue(self.copy.completion_requested(runtime, expected))
        stop.write_text(json.dumps(dict(reason='task_complete', expected_checkpoint_count=51)))
        with self.assertRaisesRegex(RuntimeError, 'STOP|52'):
            self.copy.completion_requested(runtime, expected)

    def test_main_stop_after_drained_cycle_saves_completed_or_failed_actual_exit(self):
        snapshots = [dict(kind='checkpoint', stage=stage, group_id=f'vjepa21_vitl1024_cube{cube}',
            counter=counter, target_parts=['fixture', stage, str(cube), str(counter)])
            for stage, counters in self.watch.COUNTERS.items()
            for cube in range(1, 5) for counter in counters]
        for count in (51, 52):
            with self.subTest(accepted_count=count):
                local = self.home/f'worker_{count}'
                runtime = local/'backup_runtime'
                runtime.mkdir(parents=True)
                (runtime/'STOP.json').write_text(json.dumps(dict(reason='task_complete', expected_checkpoint_count=52)))
                cfg_path = local/'backup_config.json'
                cfg_path.write_text(json.dumps(dict(local_root=str(local))))
                worker = Mock()
                worker.request.return_value = dict(checkpoints=snapshots[:count], logs=[])
                with patch.object(self.copy, 'SSHTransport', return_value=worker), \
                     patch.object(self.copy, 'copy_snapshot', return_value={}), \
                     patch.object(self.copy.time, 'sleep', side_effect=AssertionError('STOP must not sleep')), \
                     patch.object(sys, 'argv', ['backup_copy.py', '--config', str(cfg_path)]):
                    if count == 52:
                        self.assertEqual(self.copy.main(), 0)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                            self.copy.main()
                status = json.loads((runtime/'status.json').read_text())
                self.assertEqual(status['phase'], 'completed' if count == 52 else 'failed')
                exit_records = list(runtime.glob('worker_exit_*.json'))
                self.assertEqual(len(exit_records), 1)
                self.assertEqual(json.loads(exit_records[0].read_text())['actual_exit_code'], 0 if count == 52 else 1)
                self.assertFalse((runtime/'worker.lock').exists())


class FakeTransport:
    def __init__(self, watch, cfg):
        self.watch, self.cfg = watch, cfg
        self.actions, self.copied = [], []
        self.corrupt = self.mutate_after = False
        self.copy_exit = 0

    def copy(self, source, target, log_prefix):
        self.copied.append(str(source))
        if self.copy_exit:
            return dict(actual_exit_code=self.copy_exit)
        shutil.copyfile(source, target)
        if self.corrupt and str(source).endswith('.pkl'):
            data = Path(target).read_bytes()
            Path(target).write_bytes(b'X' + data[1:])
        return dict(actual_exit_code=0)

    def request(self, action, payload=None):
        self.actions.append(action)
        if action == 'poststat' and self.mutate_after:
            Path(payload['snapshot']['files']['run_state.pkl']['path']).write_bytes(b'changed')
        return self.watch.dispatch(self.cfg, action, payload)


if __name__ == '__main__':
    unittest.main()
