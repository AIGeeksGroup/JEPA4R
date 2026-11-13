# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_recovery.py (verbatim except: module names / paths of the moved code adapted)
"""CPU tests for original Lightning stateful-loader recovery; CUDA must be hidden."""
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from jepa4r.policy.bc.relocate.recovery import stateful_loader_class,make_lightning_recovery_callback,RecoveryPause
from jepa4r.utils.training_state import capture_rng

class RandomRows(Dataset):
    def __len__(self):return 18
    def __getitem__(self,i):
        x=torch.tensor([i/18.,np.random.uniform(),random.random()],dtype=torch.float32)+torch.rand(3)*.01
        return {'x':x,'y':torch.tensor([.3,-.2]),'id':i}

def seed():random.seed(43);np.random.seed(44);torch.manual_seed(45)
def equal(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b));return
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)

class RecoveryTests(unittest.TestCase):
    def test_default_sampler_and_rng_sequence_are_preserved(self):
        seed();a=DataLoader(RandomRows(),batch_size=2,shuffle=True,drop_last=True,generator=torch.Generator().manual_seed(17));first=list(a);rng=capture_rng();gen=a.generator.get_state()
        seed();cls=stateful_loader_class(DataLoader);b=cls(RandomRows(),batch_size=2,shuffle=True,drop_last=True,generator=torch.Generator().manual_seed(17));second=[]
        self.assertEqual(b.batch_size,a.batch_size);self.assertEqual(b.drop_last,a.drop_last);self.assertEqual(type(b.sampler),type(a.sampler))
        for batch in b:second.append(batch);b.commit_batch()
        equal(self,first,second);equal(self,rng,capture_rng());equal(self,gen,b.generator.get_state())

    def test_mid_epoch_loader_restore_has_exact_next_batch_and_rng(self):
        seed();cls=stateful_loader_class(DataLoader);a=cls(RandomRows(),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(17));it=iter(a);next(it);a.commit_batch();saved=a.state_dict();expected=next(it);a.commit_batch();rng=capture_rng()
        seed();b=cls(RandomRows(),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(17));b.load_state_dict(saved);actual=next(iter(b));b.commit_batch();equal(self,expected,actual);equal(self,rng,capture_rng())

    def test_worker_prefetch_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError,'workers=0'):stateful_loader_class(DataLoader)(RandomRows(),batch_size=2,shuffle=True,num_workers=1)

    def test_validation_short_iterator_can_restart_after_sanity(self):
        seed();original=DataLoader(RandomRows(),batch_size=2);expected=[next(iter(original)),next(iter(original))];rng=capture_rng()
        seed();loader=stateful_loader_class(DataLoader)(RandomRows(),batch_size=2);actual=[next(iter(loader)),next(iter(loader))]
        equal(self,expected,actual);equal(self,rng,capture_rng())

    def test_independent_lightning_resume_matches_complete_state(self):
        from contextlib import nullcontext
        keep=os.environ.get('RECOVERY_TEST_ARTIFACT_ROOT')
        if keep:Path(keep).mkdir(parents=True,exist_ok=False)
        with (nullcontext(keep) if keep else tempfile.TemporaryDirectory()) as d:
            root=Path(d)
            def run(mode,resume=None):
                out=root/mode;cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',mode,str(out)]
                if resume:cmd.append(str(resume))
                p=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                (root/(mode+'.log')).write_text(p.stdout+p.stderr)
                self.assertEqual(p.returncode,75 if mode=='pause' else 0,p.stdout+p.stderr)
                return out/'recovery'
            continuous=run('continuous');pause=run('pause');resumed=run('resume',pause/'pause_batch_0001.ckpt')
            a=torch.load(continuous/'final.ckpt',map_location='cpu',weights_only=False);b=torch.load(resumed/'final.ckpt',map_location='cpu',weights_only=False)
            for key in ('state_dict','optimizer_states','lr_schedulers','epoch','global_step','cube_smoke_state','adroit_optimizer_observers'):
                equal(self,a[key],b[key])
            equal(self,a['loops']['fit_loop']['state_dict']['combined_loader'],b['loops']['fit_loop']['state_dict']['combined_loader'])
            self.assertEqual(a['global_step'],2)
            self.assertTrue(a['lr_schedulers'],'actual SPT scheduler state must be retained')
            self.assertEqual(a['callbacks']['AdroitT4Recovery-v1']['records'],b['callbacks']['AdroitT4Recovery-v1']['records'])

def worker(mode,out,resume=None):
    import lightning as pl
    import stable_pretraining as spt
    from types import SimpleNamespace
    from unittest.mock import patch
    from jepa4r.policy.bc.relocate.adroit_learning import OptimizerObserver
    from jepa4r.policy.bc.relocate.wm_recovery import training_recovery
    torch.set_num_threads(1);seed();out.mkdir()
    def Tiny():
        def forward(self,batch,stage):
            if stage!='train':torch.rand(())
            return {'loss':(self.model(batch['x'])-batch['y']).square().mean()}
        return spt.Module(model=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.1),torch.nn.Linear(8,2)),forward=forward,
            optim={'model_opt':{'modules':'model','optimizer':{'type':'AdamW','lr':.001,'weight_decay':.001},
                'scheduler':{'type':'LinearWarmupCosineAnnealingLR'},'interval':'epoch'}})
    identity={'model':'toy-original-lightning','data':'real-on-fetch-rng-test','config':'two-step-zero-then-positive-lr','split':'train','action_dim':2}
    observed={'optimizer_observers':{}};counts={};restored={};handles=[]
    class Spy(pl.Callback):
        def on_fit_start(self,trainer,module):
            opt=trainer.optimizers[0];counts['0']=0;observed['optimizer_observers']['0']=OptimizerObserver(opt)
            def count(*unused):counts['0']+=1
            handles.append(opt.register_step_post_hook(count))
    args=SimpleNamespace(mode='smoke',model='jepa4r',output=out,pause_after_batches=1 if mode=='pause' else None,resume_checkpoint=Path(resume) if resume else None)
    spt.set(cache_dir=str(out/'spt_cache'),cleanup={'checkpoints':False,'logs':False,'callback_artifacts':False})
    try:
        with patch('jepa4r.policy.bc.relocate.wm_recovery._identity',return_value=identity),training_recovery(args,observed,counts,restored):
            train=torch.utils.data.DataLoader(RandomRows(),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(17));val=torch.utils.data.DataLoader(RandomRows(),batch_size=2)
            trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,max_steps=2,limit_train_batches=2,limit_val_batches=1,num_sanity_val_steps=1,logger=False,enable_checkpointing=True,enable_progress_bar=False,enable_model_summary=False,callbacks=[Spy()])
            spt.Manager(trainer=trainer,module=Tiny(),data=spt.data.DataModule(train=train,val=val))()
        observed['optimizer_observers']['0'].report()
        assert counts['0']==2 and counts['0']-restored.get('0',0)==(1 if resume else 2)
    except RecoveryPause:return 75
    finally:
        for handle in handles:handle.remove()
        for observer in observed['optimizer_observers'].values():observer.remove()
    return 0

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':raise SystemExit(worker(sys.argv[2],Path(sys.argv[3]),sys.argv[4] if len(sys.argv)>4 else None))
    unittest.main()
