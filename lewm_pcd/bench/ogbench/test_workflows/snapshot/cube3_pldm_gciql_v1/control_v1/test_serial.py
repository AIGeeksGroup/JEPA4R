"""Two fake-child CPU cases over the exact serial body; no NN, GPU or real jobs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
import serial
from serial import load_supervisor

REFERENCE = Path(__file__).resolve().parent/'reference_source/serial_relay.py'


class FakeChild:
    calls = []
    failure = None
    def __init__(self, argv, **kwargs):
        assert kwargs['start_new_session'] is True
        self.code = 7 if argv[-1] == self.failure else 0
        self.pid = 10000+len(self.calls); self.calls.append(argv[-1])
    def poll(self): return self.code
    def wait(self, **_kwargs): return self.code


def fixture(root, fail=False):
    control = root/'control'; runtime = control/'runtime_v1'
    control.mkdir()
    guard = root/'fake_guard.py'
    guard.write_text("def events(): return dict(oom=0,oom_kill=0)\ndef members(_group): return []\n")
    report = root/'report.json'; report.write_text(json.dumps(dict(actual_exit_code=0)))
    plan = dict(cube_count=3, data_id='cube-triple-play-v0', training_updates=1000000, SR_episodes=250,
        source_pins={str(guard):hashlib.sha256(guard.read_bytes()).hexdigest()},
        process_guard=str(guard), own_RSS_limit_bytes=24*1024**3,
        phases=[dict(name=name, argv=['fake_child',name], cwd=str(root), environment={},
            checks=[dict(path=str(report), equals=dict(actual_exit_code=0))]) for name in ('first','second')])
    (control/'launch_plan.json').write_text(json.dumps(plan))
    old = load_supervisor(control, runtime, REFERENCE)
    FakeChild.calls=[]; FakeChild.failure='second' if fail else None
    return old, runtime


class SerialTests(unittest.TestCase):
    def run_fake(self, old):
        with patch.object(old.os,'sched_getaffinity',return_value={0,1},create=True), \
             patch.object(old.os,'sched_setaffinity',return_value=None,create=True), \
             patch.object(old.subprocess,'Popen',FakeChild), \
             patch.object(serial,'CONTROL',old.HERE), \
             patch.object(serial,'load_supervisor',return_value=old), \
             patch.object(sys,'argv',['serial.py','--execute']):
            serial.main()

    def test_two_successful_phases_close_with_real_exit_fields(self):
        with tempfile.TemporaryDirectory() as name:
            old,runtime=fixture(Path(name)); self.run_fake(old)
            self.assertTrue(old.identity_binding['reverse_source_equals_original'])
            self.assertEqual(FakeChild.calls,['first','second'])
            closure=json.loads((runtime/'closure.json').read_text())
            self.assertEqual((closure['actual_exit_code'],closure['completed_phases'],closure['expected_features_per_modality']),(0,2,3303300))
            self.assertEqual((closure['training_updates'],closure['SR_episodes']),(1000000,250))
            self.assertEqual((runtime/'outer_exit_code').read_text().strip(),'0')
            self.assertEqual(json.loads((runtime/'status.json').read_text())['stage'],'pipeline_complete')
            for n in ('first','second'):
                self.assertEqual(json.loads((runtime/(n+'_exit.json')).read_text())['residual_owned_members'],[])

    def test_nonzero_child_propagates_without_success_closure(self):
        with tempfile.TemporaryDirectory() as name:
            old,runtime=fixture(Path(name),True)
            with self.assertRaises(AssertionError): self.run_fake(old)
            self.assertEqual(json.loads((runtime/'second_exit.json').read_text())['actual_exit_code'],7)
            self.assertEqual((runtime/'outer_exit_code').read_text().strip(),'1')
            self.assertFalse((runtime/'closure.json').exists())


if __name__ == '__main__': unittest.main()
