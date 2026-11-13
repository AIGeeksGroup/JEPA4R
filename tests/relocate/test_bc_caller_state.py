# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_bc_caller_state.py (verbatim except: module names / paths
# of the moved code adapted; lines wrapped at 120 columns)
"""Caller bookkeeping is sealed separately from the unchanged BC recovery hook."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa4r.policy.bc.relocate.bc_caller_state import save_caller_state,load_caller_state,epoch_metric_prefix


class CallerStateTests(unittest.TestCase):
    def fixture(self,root):
        checkpoint=root/'update_000001.pt';checkpoint.write_bytes(b'actual model/Adam bytes')
        meta = {
            'progress': {'committed_updates': 1},
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'actual_exit_code': 0,
        }
        checkpoint.with_suffix('.pt.json').write_text(json.dumps(meta))
        return checkpoint

    def test_roundtrip_and_content_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.fixture(Path(d));identity={'source':{'adapter':'a','hook':'b'},'config':{'seed':100}}
            state={'count':1,'learning_rates':[[.001]]}
            save_caller_state(path,identity=identity,state=state,committed_updates=1,model_sha256='head')
            self.assertEqual(
                load_caller_state(path, identity=identity, committed_updates=1, model_sha256='head', adam_steps=[1.0]),
                state,
            )
            for changes in (
                {'identity': {'source': 'changed'}},
                {'committed_updates': 2},
                {'model_sha256': 'changed'},
                {'adam_steps': [2.0]},
            ):
                kwargs = dict(identity=identity, committed_updates=1, model_sha256='head', adam_steps=[1.0])
                kwargs.update(changes)
                with self.assertRaises(ValueError):load_caller_state(path,**kwargs)
            path.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                load_caller_state(path, identity=identity, committed_updates=1, model_sha256='head', adam_steps=[1.0])

    def test_missing_sidecar_and_changed_main_progress_reject(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.fixture(Path(d))
            with self.assertRaises(FileNotFoundError):
                load_caller_state(path, identity={}, committed_updates=1, model_sha256='head', adam_steps=[1.0])
            save_caller_state(path,identity={},state={},committed_updates=1,model_sha256='head')
            meta = path.with_suffix('.pt.json')
            row = json.loads(meta.read_text())
            row['progress']['committed_updates'] = 2
            meta.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_caller_state(path, identity={}, committed_updates=1, model_sha256='head', adam_steps=[1.0])

    def test_metric_prefix_preserves_original_epoch_loss(self):
        events=[{'epoch':0,'running_loss':3.25,'minibatches':1},
                {'epoch':1,'running_loss':7.,'minibatches':2,'evaluation':{'eval/epoch':1,'eval/success':0.}}]
        self.assertEqual(epoch_metric_prefix(events),[{'epoch_loss':3.25,'_step':1},
            {'epoch_loss':3.5,'_step':2},{'eval/epoch':1,'eval/success':0.,'_step':2}])

    def test_rh_two_epoch_budget_is_explicit_and_profile_only(self):
        from jepa4r.envs.twins.rh56.bc_bridge import build_run_spec
        self.assertEqual(build_run_spec('profile',1,100,None)['epochs'],1)
        self.assertEqual(build_run_spec('profile',1,100,None,bc_recovery=True)['epochs'],2)
        for mode in ('smoke','formal'):
            with self.assertRaises(ValueError):build_run_spec(mode,10,100,None,bc_recovery=True)

    def test_native_bc_flags_default_off_and_explicit_resume(self):
        from jepa4r.policy.bc.relocate import adroit_learning as native
        base=['adroit_learning','bc','--model','jepa4r','--output','fresh']
        for extra,enabled in (([],False),(['--bc-recovery','--bc-resume-checkpoint','paused.pt'],True)):
            captured=[]
            with (
                patch('sys.argv', base + extra),
                patch.object(native, 'configure_runtime'),
                patch.object(native, 'bc', captured.append),
            ):
                native.main()
            self.assertEqual(captured[0].bc_recovery,enabled)
        with patch('sys.argv',base+['--bc-pause-after-updates','1']),patch.object(native,'configure_runtime'):
            with self.assertRaises(SystemExit) as caught:native.main()
            self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
