"""Small contract tests for the optional Cube recovery wiring."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.ogbench import cube_recovery as recovery


class IdentityTests(unittest.TestCase):
    def plan(self):
        return dict(method='lewm_pcd',cube_count=1,mode='smoke',workers=0,
            data_manifest='bound.json',data_manifest_sha256='a'*64,
            recipe={'epochs':1,'batch_size':128},
            config=dict(trainer=dict(max_epochs=1,limit_train_batches=2,limit_val_batches=1),
                        loader=dict(num_workers=0)))

    def identity(self,plan):
        with patch('bench.ogbench.fresh_data.load_binding',return_value={'producer_binding':{'old':'source'}}), \
             patch('bench.ogbench.fresh_train._source_binding',return_value={'actual':'source'}), \
             patch.object(recovery,'sha',return_value='a'*64):
            return recovery.recovery_identity(plan)

    def test_plan_manifest_is_rechecked_before_running(self):
        plan=self.plan();plan['data_manifest_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'manifest changed'):self.identity(plan)

    def test_larger_budget_is_rejected_before_original_loop(self):
        plan=self.plan();plan['config']['trainer']['limit_train_batches']=3
        with self.assertRaisesRegex(ValueError,'two-update'):self.identity(plan)

    def test_identity_binds_executor_and_shared_loader(self):
        identity=self.identity(self.plan())
        self.assertEqual(identity['training_seed'],0)
        self.assertIn('recovery_executor_sha256',identity['model'])
        self.assertEqual(identity['model']['shared_loader_sha256'],recovery.SHARED_LOADER_SHA256)


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.original_loader=object()
        self.wrapped=object()
        def manager(*args,**kwargs):self.calls.append(kwargs);return kwargs
        self.original_manager=manager
        self.torch=SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(DataLoader=self.original_loader)))
        self.spt=SimpleNamespace(Manager=manager)
        self.shared=ModuleType('bench.relocate.recovery');self.shared.__file__='frozen_recovery.py'
        self.shared.stateful_loader_class=lambda original:self.wrapped
        self.shared.make_lightning_recovery_callback=lambda *a,**k:('callback',a,k)
        self.validate_calls=[]
        self.wm=ModuleType('bench.relocate.wm_recovery')
        def validate(path,identity):self.validate_calls.append((path,identity));return Path('checked.ckpt')
        self.wm.validate_resume=validate
        self.parent=ModuleType('bench.relocate');self.parent.recovery=self.shared
        self.modules={'torch':self.torch,'stable_pretraining':self.spt,'bench.relocate':self.parent,
                      'bench.relocate.recovery':self.shared,'bench.relocate.wm_recovery':self.wm}

    def test_optional_context_restores_all_aliases_even_on_failure(self):
        alias=SimpleNamespace(DataLoader=self.original_loader)
        with patch.dict(sys.modules,self.modules),patch.object(recovery,'sha',return_value=recovery.SHARED_LOADER_SHA256):
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with recovery.training_context({'model':'cube'},'output',aliases=(alias,)):
                    self.assertIs(alias.DataLoader,self.wrapped)
                    trainer=SimpleNamespace(callbacks=[])
                    result=self.spt.Manager(trainer=trainer,seed=0,ckpt_path=None)
                    self.assertFalse(result['weights_only'])
                    self.assertEqual(len(trainer.callbacks),1)
                    raise RuntimeError('fixture')
        self.assertIs(alias.DataLoader,self.original_loader)
        self.assertIs(self.torch.utils.data.DataLoader,self.original_loader)
        self.assertIs(self.spt.Manager,self.original_manager)

    def test_resume_uses_sealed_identity_validation_and_preserves_seed0(self):
        identity={'model':'cube'}
        with patch.dict(sys.modules,self.modules),patch.object(recovery,'sha',return_value=recovery.SHARED_LOADER_SHA256):
            with recovery.training_context(identity,'output',resume_checkpoint='pause.ckpt'):
                result=self.spt.Manager(trainer=SimpleNamespace(callbacks=[]),seed=0)
                self.assertEqual(result['ckpt_path'],'checked.ckpt')
                with self.assertRaisesRegex(ValueError,'seed0'):
                    self.spt.Manager(trainer=SimpleNamespace(callbacks=[]),seed=3072)
        self.assertEqual(self.validate_calls,[('pause.ckpt',identity)])


if __name__=='__main__':unittest.main()
