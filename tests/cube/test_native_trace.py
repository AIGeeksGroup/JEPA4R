# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_native_trace.py (verbatim except: imports / paths
# adapted to the package; lines wrapped at 120 columns)
"""Trace alignment does not require an encoder or simulator installation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql.native_trace import NativeTrace


class TraceTests(unittest.TestCase):
    def test_records_initial_and_outgoing_action_alignment_without_mutation(self):
        data = SimpleNamespace(qpos=np.arange(4.), qvel=np.arange(3.), ctrl=np.zeros(2), time=0.)
        base = SimpleNamespace(_data=data)
        initial = data.qpos.copy()
        trace = NativeTrace()
        trace.reset(base, np.zeros(28), np.zeros(28))
        trace.policy(np.ones(192), np.ones(192), np.ones(211), np.ones(195), np.zeros(5))
        data.qpos += .1
        data.time = .02
        trace.step(base, np.zeros(5), np.ones(28), 0., False, True, {'success': False})
        arrays = trace.arrays()
        self.assertEqual(arrays['qpos'].shape, (2, 4))
        np.testing.assert_array_equal(arrays['qpos'][0], initial)
        np.testing.assert_array_equal(arrays['qpos'][1], data.qpos)
        self.assertEqual(arrays['executed_actions'].shape, (1, 5))
        self.assertEqual(arrays['policy_current'].shape, (1, 211))
        self.assertTrue(arrays['truncated'][0])
        np.testing.assert_array_equal(data.qpos, initial + .1)

    def test_incomplete_policy_step_alignment_is_rejected(self):
        data = SimpleNamespace(qpos=np.zeros(4), qvel=np.zeros(3), ctrl=np.zeros(2), time=0.)
        trace = NativeTrace()
        trace.reset(SimpleNamespace(_data=data), np.zeros(28), np.zeros(28))
        trace.step(SimpleNamespace(_data=data), np.zeros(5), np.zeros(28), 0., False, True, {})
        with self.assertRaisesRegex(ValueError, 'alignment'):
            trace.arrays()


if __name__ == '__main__':
    unittest.main()
