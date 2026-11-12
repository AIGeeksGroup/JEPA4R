"""M2 verifies the full saved policy tree before and after the original restore."""
import copy
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from bench.ogbench import gate_checks as gate


class MatrixPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name) / 'params_2.pkl'
        self.template = dict(network=dict(params={'w': np.zeros((2,3),np.float32)},
                                        buffers={'mean': np.zeros(2,np.float32)}, step=np.int64(0)),
                             rng=np.zeros(2,np.uint32))
        self.saved = copy.deepcopy(self.template); self.saved['network']['params']['w'] += 1
        self.saved['network']['step'] = np.int32(2); self.calls = []

    def tearDown(self): self.tmp.cleanup()

    def pin(self, state):
        self.path.write_bytes(pickle.dumps({'agent': state}))
        return dict(path=str(self.path.parent),counter=2,sha256=gate.sha(self.path))

    def run_restore(self, state, restored=None):
        pin = self.pin(state)
        def restore(template, directory, counter):
            self.calls.append((template,directory,counter))
            return copy.deepcopy(state if restored is None else restored)
        return gate._matrix_restore_policy(self.template,pin,restore,to_state_dict=lambda value:value)

    def test_complete_trained_tree_calls_original_restore_and_records_counter_dtype(self):
        agent, arrays, evidence = self.run_restore(self.saved)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(set(arrays), {'network/params/w','network/buffers/mean','network/step','rng'})
        self.assertTrue(evidence['comparison']['passed'])
        self.assertEqual(evidence['integer_dtype_allowances'][0]['path'],'network/step')
        np.testing.assert_array_equal(agent['network']['params']['w'],np.ones((2,3),np.float32))

    def test_extra_missing_empty_key_shape_dtype_and_nonfinite_reject_before_restore(self):
        for change in ['missing','extra','empty_extra','shape','dtype','nonfinite']:
            value=copy.deepcopy(self.saved)
            if change=='missing': del value['network']['buffers']
            elif change=='extra': value['hidden']=np.zeros(1,np.float32)
            elif change=='empty_extra': value['hidden']={}
            elif change=='shape': value['network']['params']['w']=np.ones((3,2),np.float32)
            elif change=='dtype': value['network']['params']['w']=np.ones((2,3),np.float64)
            else: value['network']['params']['w'][0,0]=np.nan
            with self.subTest(change=change),self.assertRaises(ValueError): self.run_restore(value)
            self.assertEqual(self.calls,[])

    def test_ignored_or_changed_buffer_after_restore_rejects(self):
        wrong=copy.deepcopy(self.saved);wrong['network']['buffers']['mean'][1]=1
        with self.assertRaisesRegex(ValueError,'restored policy'):self.run_restore(self.saved,wrong)
        self.assertEqual(len(self.calls),1)


if __name__=='__main__':unittest.main()
