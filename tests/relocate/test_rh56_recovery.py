"""Thin RH56 full-state context contracts, independent of simulation imports."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import subprocess
import sys
import json

from jepa4r.envs.twins.rh56.wm_recovery import epoch_progress,restore_epoch_progress,EvidenceState


class EpochEvidenceTests(unittest.TestCase):
    def test_mid_epoch_statistics_survive_existing_epoch_start(self):
        callback=SimpleNamespace(epoch_sum=1.25,epoch_batches=1,val_sum=0.,val_batches=0)
        saved=epoch_progress(callback,0)
        other=SimpleNamespace(epoch_sum=99.,epoch_batches=99,val_sum=99.,val_batches=99)
        self.assertTrue(restore_epoch_progress(other,saved,0))
        self.assertEqual(vars(other),vars(callback))
        self.assertFalse(restore_epoch_progress(other,saved,1))

    def test_missing_or_wrong_epoch_statistics_reject(self):
        with self.assertRaises(ValueError):epoch_progress(SimpleNamespace(epoch_sum=1.),0)
        saved=dict(epoch=0,epoch_sum=1.,epoch_batches=-1,val_sum=0.,val_batches=0)
        with self.assertRaises(ValueError):restore_epoch_progress(SimpleNamespace(),saved,0)
    def test_mid_epoch_load_restores_live_callback_without_epoch_start(self):
        # Lightning can resume directly at the next batch without invoking
        # on_train_epoch_start. The real restoration path must initialize it now.
        with tempfile.TemporaryDirectory() as directory:
            callback=SimpleNamespace();capture={'evidence_callback':callback};evidence={}
            saved = dict(
                evidence={'train_batches': 1},
                epoch_progress=dict(epoch=0, epoch_sum=0.25, epoch_batches=1, val_sum=0.0, val_batches=0),
                real_reload_batch=None,
            )
            state=EvidenceState(evidence,capture,directory);state.load_state_dict(saved)
            self.assertEqual(epoch_progress(callback,0),saved['epoch_progress'])
            self.assertEqual(capture['epoch_index'],0);self.assertEqual(evidence['train_batches'],1)

    @unittest.skipUnless(os.environ.get('RH_RECOVERY_INTEGRATION'),'requires existing Torch/SPT CPU profile')
    def test_rh_context_uses_original_spt_loop_independent_resume(self):
        import torch
        from test_recovery import equal
        root=Path(os.environ['RH_RECOVERY_TEST_ARTIFACT_ROOT']);root.mkdir(parents=True,exist_ok=False)
        for mode, expected, resume in [
            ('continuous', 0, None),
            ('pause', 75, None),
            ('resume', 0, root / 'pause/recovery/pause_batch_0001.ckpt'),
        ]:
            command=[sys.executable,'-B',__file__,'--worker',mode,str(root/mode)]
            if resume:command.append(str(resume))
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
            )
            (root / (mode + '.log')).write_text(process.stdout + process.stderr)
            (root / (mode + '.exit.json')).write_text(
                json.dumps(
                    dict(actual_exit_code=process.returncode, expected_exit_code=expected, command=command), indent=2
                )
            )
            self.assertEqual(process.returncode,expected,process.stdout+process.stderr)
        a=torch.load(root/'continuous/recovery/final.ckpt',map_location='cpu',weights_only=False)
        b=torch.load(root/'resume/recovery/final.ckpt',map_location='cpu',weights_only=False)
        for key in (
            'state_dict',
            'optimizer_states',
            'lr_schedulers',
            'cube_smoke_state',
            'adroit_train_loader_state',
            'adroit_optimizer_observers',
        ):
            equal(self,a[key],b[key])
        equal(self,a['callbacks']['AdroitT4Recovery-v1'],b['callbacks']['AdroitT4Recovery-v1'])


def worker(mode,out,resume=None):
    import lightning as pl
    import torch
    import stable_pretraining as spt
    from unittest.mock import patch
    from contextlib import contextmanager
    import test_recovery
    from jepa4r.models.relocate import wm_recovery as native_recovery
    from jepa4r.envs.twins.rh56.wm_recovery import training_context
    @contextmanager
    def rh_context(args,observed,counts,restored):
        pl.seed_everything(3072,workers=True)
        def save():return {key:value.history_state_dict() for key,value in observed['optimizer_observers'].items()}
        def restore(saved):
            if set(saved)!=set(observed['optimizer_observers']):raise ValueError('RH unit observer identities differ')
            for key,value in saved.items():
                observed['optimizer_observers'][key].load_history_state_dict(value)
                counts[key] = restored[key] = value['count']
        original_loader=torch.utils.data.DataLoader;original_manager=spt.Manager
        try:
            with training_context(
                SimpleNamespace(spt=spt),
                lambda: native_recovery._identity(args),
                args.output / 'recovery',
                seed=3072,
                pause_after_batches=args.pause_after_batches,
                resume_checkpoint=args.resume_checkpoint,
                evidence_provider=save,
                evidence_restorer=restore,
            ):yield
        finally:
            assert torch.utils.data.DataLoader is original_loader and spt.Manager is original_manager
    with patch.object(native_recovery,'training_recovery',rh_context):
        return test_recovery.worker(mode,Path(out),resume)


if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        raise SystemExit(worker(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
    unittest.main()
