# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_cube_recovery_cpu.py (verbatim except: imports / paths adapted to the package)
"""Real CPU SPT Manager fixture for Cube's optional recovery context."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))


def equal(test,left,right):
    import torch,numpy as np
    if torch.is_tensor(left):test.assertTrue(torch.equal(left,right));return
    if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right);return
    if isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for name in left:equal(test,left[name],right[name])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):equal(test,a,b)
    else:test.assertEqual(left,right)


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires the existing Torch/SPT CPU environment')
class RealManagerTests(unittest.TestCase):
    def test_independent_cube_context_resume_matches_original_spt_state(self):
        import torch
        keep=os.environ.get('CUBE_RECOVERY_TEST_ARTIFACTS')
        if keep:Path(keep).mkdir(parents=True,exist_ok=False)
        with (nullcontext(keep)if keep else tempfile.TemporaryDirectory())as folder:
            root=Path(folder)
            def run(mode,resume=None):
                output=root/mode
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',mode,str(output)]
                if resume:command.append(str(resume))
                child=subprocess.run(command,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                (root/(mode+'.log')).write_text(child.stdout+child.stderr)
                (root/(mode+'.exit')).write_text(str(child.returncode)+'\n')
                self.assertEqual(child.returncode,75 if mode=='pause'else 0,child.stdout+child.stderr)
                return output/'recovery'
            continuous=run('continuous');pause=run('pause');resumed=run('resume',pause/'pause_batch_0001.ckpt')
            actual=torch.load(resumed/'final.ckpt',map_location='cpu',weights_only=False)
            reference=torch.load(continuous/'final.ckpt',map_location='cpu',weights_only=False)
            for name in ['state_dict','optimizer_states','lr_schedulers','epoch','global_step','cube_smoke_state',
                         'adroit_train_loader_state']:
                equal(self,actual[name],reference[name])
            equal(self,actual['callbacks']['AdroitT4Recovery-v1'],reference['callbacks']['AdroitT4Recovery-v1'])
            equal(self,actual['loops']['fit_loop']['state_dict']['combined_loader'],
                  reference['loops']['fit_loop']['state_dict']['combined_loader'])
            self.assertEqual(actual['global_step'],2)
            self.assertTrue(actual['lr_schedulers'])
            adam_steps={int(row['step'])for row in actual['optimizer_states'][0]['state'].values()}
            self.assertEqual(adam_steps,{2})
            self.assertEqual(len(actual['callbacks']['AdroitT4Recovery-v1']['records']),2)


def worker(mode,output,resume=None):
    import torch,numpy as np,random
    import lightning as pl
    import stable_pretraining as spt
    from jepa4r.policy.gciql.cube_recovery import training_context
    from jepa4r.policy.bc.relocate.recovery import RecoveryPause
    torch.set_num_threads(1);pl.seed_everything(0,workers=True);output.mkdir()
    class Rows(torch.utils.data.Dataset):
        def __len__(self):return 18
        def __getitem__(self,index):
            x=torch.tensor([index/18.,np.random.uniform(),random.random()])+torch.rand(3)*.01
            return {'x':x,'y':torch.tensor([.3,-.2]),'id':index}
    def forward(self,batch,stage):
        if stage!='train':torch.rand(())
        return {'loss':(self.model(batch['x'])-batch['y']).square().mean()}
    identity=dict(model='toy_original_SPT_via_Cube_context',data='real_on_fetch_rng_rows',config='2updates',split='train',action_dim=2)
    spt.set(cache_dir=str(output/'spt_cache'),cleanup={'checkpoints':False,'logs':False,'callback_artifacts':False})
    try:
        with training_context(identity,output/'recovery',pause_after_batches=1 if mode=='pause'else None,
                              resume_checkpoint=Path(resume)if resume else None):
            train=torch.utils.data.DataLoader(Rows(),batch_size=2,shuffle=True,drop_last=True,
                                             generator=torch.Generator().manual_seed(0))
            val=torch.utils.data.DataLoader(Rows(),batch_size=2)
            model=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.1),torch.nn.Linear(8,2))
            world=spt.Module(model=model,forward=forward,optim={'model_opt':{'modules':'model',
                'optimizer':{'type':'AdamW','lr':.001,'weight_decay':.001},
                'scheduler':{'type':'LinearWarmupCosineAnnealingLR'},'interval':'epoch'}})
            trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,max_steps=2,
                limit_train_batches=2,limit_val_batches=1,num_sanity_val_steps=1,logger=False,
                enable_checkpointing=True,enable_progress_bar=False,enable_model_summary=False)
            spt.Manager(trainer=trainer,module=world,data=spt.data.DataModule(train=train,val=val),seed=0)()
    except RecoveryPause:return 75
    return 0


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        raise SystemExit(worker(sys.argv[2],Path(sys.argv[3]),sys.argv[4]if len(sys.argv)>4 else None))
    unittest.main()
