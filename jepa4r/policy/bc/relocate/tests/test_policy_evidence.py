# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_policy_evidence.py (verbatim except: module names / paths of the moved code adapted)
"""Real-input policy evidence contracts; no Torch or simulator required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import random
import numpy as np

FILE = Path(__file__).resolve().parents[5] / 'jepa4r/policy/bc/relocate/policy_evidence.py'

class Model:
    training = False
    def __init__(self, value=2.): self.weight = np.array([value], dtype=np.float32)
    def state_dict(self): return {'weight': self.weight}

class Policy:
    n, m = 3, 2
    def __init__(self, value=2.): self.model = Model(value)
    def get_action(self, feature):
        np.random.normal(); random.random()
        return None, {'evaluation': np.asarray(feature[:2]) * self.model.weight[0]}

class PolicyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('tested_policy_evidence', FILE)
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)

    def features(self): return np.arange(12, dtype=np.float32).reshape(4, 3)

    def test_real_input_replay_matches_corresponding_model(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.api.write_policy_reference(d, 'round000', Policy(), self.features(), epoch=5, score=10.)
            result = self.api.verify_policy_reference(Policy(), r['metadata_path'])
            self.assertTrue(result['passed'])
            self.assertEqual(result['epoch'], 5)
            self.assertEqual(result['maximum_action_difference'], 0.)
            self.assertEqual(result['feature_shape'], [4, 3])

    def test_wrong_selected_epoch_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.api.write_policy_reference(d, 'round000', Policy(2.), self.features(), epoch=5, score=10.)
            with self.assertRaisesRegex(ValueError, 'policy state'):
                self.api.verify_policy_reference(Policy(3.), r['metadata_path'])

    def test_feature_payload_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.api.write_policy_reference(d, 'round000', Policy(), self.features(), epoch=5, score=10.)
            p = Path(r['arrays_path']);p.write_bytes(p.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError, 'payload'):
                self.api.verify_policy_reference(Policy(), r['metadata_path'])

    def test_scalar_model_buffer_keeps_its_original_shape(self):
        policy=Policy()
        policy.model.state_dict=lambda: {'weight':policy.model.weight,'num_batches_tracked':np.array(7,dtype=np.int64)}
        with tempfile.TemporaryDirectory() as d:
            r=self.api.write_policy_reference(d,'round000',policy,self.features(),epoch=5,score=10.)
            with np.load(r['arrays_path'],allow_pickle=False) as arrays:
                self.assertEqual(arrays['state.num_batches_tracked'].shape,())
                self.assertEqual(arrays['state.num_batches_tracked'].dtype,np.dtype('int64'))

    def test_capture_preserves_python_and_numpy_rng(self):
        random.seed(31);np.random.seed(41)
        py = random.getstate();npy = np.random.get_state()
        self.api.policy_actions(Policy(), self.features())
        self.assertEqual(py, random.getstate())
        after=np.random.get_state();self.assertEqual(npy[0],after[0]);np.testing.assert_array_equal(npy[1],after[1]);self.assertEqual(npy[2:],after[2:])

    def test_capture_refuses_training_mode_and_nonfinite_input(self):
        p=Policy();p.model.training=True
        with self.assertRaisesRegex(ValueError, 'eval'):self.api.policy_actions(p,self.features())
        x=self.features();x[0,0]=np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):self.api.policy_actions(Policy(),x)

    def test_reference_is_exclusive_and_can_move_directory(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.api.write_policy_reference(root/'first','round000',Policy(),self.features(),epoch=5,score=10.)
            with self.assertRaises(FileExistsError):self.api.write_policy_reference(root/'first','round000',Policy(),self.features(),epoch=5,score=10.)
            shutil.copytree(root/'first',root/'moved')
            self.assertTrue(self.api.verify_policy_reference(Policy(),root/'moved'/Path(r['metadata_path']).name)['passed'])

    def test_best_mean_reward_can_precede_final(self):
        rows=[{'epoch':10,'score':100.,'metadata_path':'early'}, {'epoch':95,'score':80.,'metadata_path':'late'}]
        selected=self.api.select_best_reference(rows)
        self.assertEqual(selected['metadata_path'],'early')
        self.assertEqual(rows[-1]['metadata_path'],'late')

    def test_best_mean_reward_tie_selects_latest_like_original(self):
        rows=[{'epoch':10,'score':100.,'metadata_path':'early'}, {'epoch':95,'score':100.,'metadata_path':'late'}]
        self.assertEqual(self.api.select_best_reference(rows)['metadata_path'],'late')

    def test_native_opt_in_flag_precedes_hydra_argument_separator(self):
        from types import SimpleNamespace
        path = FILE.with_name('adroit_workflow.py')
        spec = importlib.util.spec_from_file_location('policy_evidence_workflow', path)
        workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=Path(d)/'new',mode='smoke',model='jepa4r',
                sim_python='sim-python',wm_python='wm-python',seed=42)
            default=workflow.plan(args,{})
            stages=workflow.plan(args,{'policy_evidence':True})
        self.assertFalse(any('--policy-evidence' in s['command'] for s in default))
        bc=next(s for s in stages if s['name']=='bc-seed100')['command']
        self.assertLess(bc.index('--policy-evidence'),bc.index('--'))
        reload=next(s for s in stages if s['name']=='policy-reload-seed100')['command']
        self.assertIn('--policy-evidence',reload)

if __name__ == '__main__': unittest.main()
