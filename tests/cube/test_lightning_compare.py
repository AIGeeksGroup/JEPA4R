"""Normalize only independent run paths and explicit checkpoint sentinels."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import lightning_compare as checker
from jepa4r.policy.gciql.gates.arrays import compare_nested

class MetadataTests(unittest.TestCase):
    def state(self,root):
        return dict(
            spt_run_dir=root,
            state_dict={'weight': np.ones(2, dtype=np.float32)},
            callbacks={
                'ModelCheckpoint-fixture': dict(
                    dirpath=root + '/checkpoints',
                    best_model_path=root + '/checkpoints/last.ckpt',
                    last_model_path='',
                    kth_best_model_path='',
                    best_k_models={},
                    kth_value=np.asarray(np.inf, dtype=np.float32),
                ),
                'actual_custom_callback': {'next_event': 2},
            },
        )
    def test_all_state_except_owned_run_paths_is_preserved(self):
        a,ar=checker.normalise(self.state('/case/continuous'));b,br=checker.normalise(self.state('/case/resume'))
        self.assertTrue(compare_nested(a,b,exact=True)['passed'])
        self.assertEqual(a['callbacks']['actual_custom_callback'],{'next_event':2})
        self.assertEqual(a['callbacks']['ModelCheckpoint-fixture']['kth_value']['identity']['dtype'],'float32')
        self.assertEqual(len(ar['path_aliases']),3)
    def test_escape_path_is_rejected_instead_of_erased(self):
        state = self.state('/case/run')
        state['callbacks']['ModelCheckpoint-fixture']['best_model_path'] = '/another/run.ckpt'
        with self.assertRaisesRegex(ValueError,'outside its actual'):checker.normalise(state)
    def test_model_nonfinite_and_changed_events_do_not_get_sentinel_exemption(self):
        a,_=checker.normalise(self.state('/case/a'));state=self.state('/case/b')
        state['state_dict']['weight'][0]=np.inf;b,_=checker.normalise(state)
        self.assertFalse(compare_nested(a,b,exact=True)['passed'])
        state=self.state('/case/b');state['callbacks']['actual_custom_callback']['next_event']=3
        b,_=checker.normalise(state);self.assertFalse(compare_nested(a,b,exact=True)['passed'])
    def test_nan_selection_metadata_is_rejected(self):
        state = self.state('/case/a')
        state['callbacks']['ModelCheckpoint-fixture']['kth_value'] = np.asarray(np.nan, np.float32)
        with self.assertRaisesRegex(ValueError,'NaN'):checker.normalise(state)

if __name__=='__main__':unittest.main()
