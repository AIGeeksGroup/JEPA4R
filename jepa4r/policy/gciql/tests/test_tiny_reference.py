# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_tiny_reference.py (verbatim except: imports / paths adapted to the package)
# ruff: noqa: F401  (moved verbatim: unused imports)
"""Passive recorder transparency on an unchanged CPU fixture, not a Tiny run."""
import ast
import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from jepa4r.policy.gciql import tiny_reference as reference


class ReferencePlanTests(unittest.TestCase):
    def test_native_markers_are_before_forward_and_after_original_scheduler(self):
        path=Path(__file__).resolve().parents[1]/'native/tiny_train.py'
        # Marker analysis works for the frozen old loop and the hook-bearing
        # loop; the execution CLI separately rejects anything but f437 bytes.
        pre,post=reference.step_markers(path)
        self.assertLess(pre,post)


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires existing Torch CPU')
class PassiveRecorderTests(unittest.TestCase):
    def test_observer_uses_original_step_and_preserves_result_and_rng(self):
        import torch
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            module_file=root/'old_fixture.py'
            module_file.write_text('''import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
def main():
    torch.manual_seed(0)
    train=DataLoader(torch.zeros(2,3),batch_size=2,shuffle=True,drop_last=True)
    net=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
    sigreg=torch.nn.Identity()
    opt=torch.optim.AdamW(net.parameters(),lr=.002)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
    scaler=None
    committed=0
    for epoch in range(10):
        net.train()
        for step,vs in enumerate(train):
            with nullcontext():
                emb=net(vs);proj=emb
                inv_loss=emb.square().mean();sigreg_loss=inv_loss*0;loss=inv_loss
            opt.zero_grad();loss.backward();opt.step()
            scheduler.step()
            committed+=1
            if committed==2:return dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=torch.get_rng_state())
''')
            spec=importlib.util.spec_from_file_location('_tiny_old_fixture',module_file)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            batch_path=root/'batches.npz'
            np.savez(batch_path,**{f'b{i}':np.arange(6,dtype=np.float32).reshape(2,3)+i for i in range(3)})
            config=dict(batch_arrays=dict(path=str(batch_path),sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest()),
                batches=[dict(id=f'epoch{i}:batch0',array_key=f'b{i}',epoch=i,batch_index=0)for i in range(3)],
                loader_samples=2,loader_batches_per_epoch=1,stop_after=2,
                model='lejepa',seed=0,precision='bf16',
                identity=dict(model='lejepa',data=hashlib.sha256(batch_path.read_bytes()).hexdigest(),config='fixture',split='train',action_dim=5))
            control=reference.PassiveTinyRecorder(old,config,root/'unused_control',enforce_native=False)
            original=old.DataLoader
            old.DataLoader=control.loader_factory(original)
            try:wanted=old.main()
            finally:old.DataLoader=original;control.close()
            recorder=reference.PassiveTinyRecorder(old,config,root/'observed',enforce_native=False)
            result=recorder.run()
            self.assertEqual(result['optimizer_updates'],2)
            self.assertFalse(result['new_gate_interface_used'])
            state=torch.load(root/'observed/step_2.pt',map_location='cpu',weights_only=False)
            for name,value in wanted['model'].items():
                self.assertTrue(torch.equal(value,state['model']['encoder.'+name]),name)
            for key,value in wanted['optimizer']['state'].items():
                for field,item in value.items():self.assertTrue(torch.equal(item,state['optimizers']['main']['state'][key][field]))
            self.assertEqual(wanted['scheduler'],state['schedulers']['main'])
            self.assertTrue(torch.equal(wanted['rng'],state['rng']['torch']))
            self.assertEqual(state['cursor']['next_batch_id'],'epoch2:batch0')
            self.assertEqual(state['progress']['step'],2)
            # The new gate must start from the independently observed original
            # initial state, not reseed away its original factory RNG position.
            from jepa4r.policy.gciql.gate_checks import TinyFixedGate
            initial=root/'observed/initial_state.pt'
            config['initial_training_state']=dict(path=str(initial),sha256=hashlib.sha256(initial.read_bytes()).hexdigest())
            new_gate=TinyFixedGate(config,root/'new_gate_not_executed')
            net=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2))
            opt=torch.optim.AdamW(net.parameters(),lr=.002)
            scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
            self.assertEqual(new_gate.attach(net,torch.nn.Identity(),opt,scheduler,None),0)
            initial_state=torch.load(initial,map_location='cpu',weights_only=False)
            self.assertTrue(torch.equal(torch.get_rng_state(),initial_state['rng']['torch']))
            self.assertEqual(new_gate.cursor.position,0)
            for key,value in net.state_dict().items():self.assertTrue(torch.equal(value,initial_state['model']['encoder.'+key]))


if __name__=='__main__':unittest.main()
