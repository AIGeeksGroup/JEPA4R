"""Isolated method switch; native SPT training, logging and checkpoint paths reused."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from shared_data import MANIFEST_SHA, contract, loaders, require_sources

SOURCE_FREEZE = 'source_freeze_v1.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()


def method_module(method):return importlib.import_module('methods.'+method+'.method')


def binding(args,stats=None):
    return dict(method=args.method,method_sha256=sha(args.root/'methods'/args.method/'method.py'),
        manifest_sha256=MANIFEST_SHA,contract=contract(args.method),batch_size=128,
        precision='bf16-mixed',epochs=10,seed=0,train_workers=20,validation_workers=20,
        prefetch_factor=2,pin_memory=True,persistent_workers=False,gradient_clip_val=None,
        action_statistics=stats,source_freeze_sha256=sha(args.root/SOURCE_FREEZE))


def factory_config(args):
    return dict(_target_='methods.'+args.method+'.method.build_model',
                official_source=str(args.root/'vendor'/args.method/'source'),pretrained=False)


def native_module(method,module,model):
    """Only optimizer construction is specialized; SPT training_step stays native."""
    import stable_pretraining as spt
    class MethodModule(spt.Module):
        def configure_optimizers(self):
            if method=='pldm':
                opt=module.build_optimizer(self.model)
                self.native_author_schedule=module.build_scheduler(opt,epochs=10,
                    batch_steps=contract('pldm')['train_batches'],batch_size=128)
                return opt
            opt,scheduler=module.make_optimizer_and_scheduler(self.model.parameters())
            return [opt],[dict(scheduler=scheduler,interval='step')]
    def pldm_forward(self,batch,stage):
        out=self.model(batch['pixels'],batch['actions'])
        for key,value in out.items():
            if 'loss' in key:self.log(stage+'/'+key,value,on_step=True,on_epoch=True,sync_dist=True)
        return out
    return MethodModule(model=model,forward=pldm_forward if method=='pldm' else module.forward,optim=None)


def author_schedule_before_step(pl_module,step):
    pl_module.native_author_schedule.adjust_learning_rate(int(step))
    return int(step)


def trainer_options(phase,callbacks):
    options=dict(max_epochs=10,accelerator='gpu',devices=1,precision='bf16-mixed',
                 gradient_clip_val=None,accumulate_grad_batches=1,callbacks=callbacks,
                 num_sanity_val_steps=1,logger=False,enable_checkpointing=True)
    if phase=='gate':options.update(max_epochs=1,limit_train_batches=2,limit_val_batches=2)
    return options


def run(args):
    import torch
    import lightning as pl
    import stable_pretraining as spt
    from omegaconf import OmegaConf
    if spt.__version__!='0.1.8':raise ValueError('existing SPT0.1.8 runtime required; do not upgrade live env')
    pl.seed_everything(0,workers=True)
    module=method_module(args.method);train,val,stats=loaders(args.method,args.data_root,module)
    code=args.root/'data_bridge/code_front600_v1/lewm_pcd'
    spec=importlib.util.spec_from_file_location('baseline_original_exports',code/'utils.py')
    exports=importlib.util.module_from_spec(spec);spec.loader.exec_module(exports)
    expected=binding(args,stats);cfg=factory_config(args)
    model=module.build_model(cfg['official_source'],pretrained=False)
    world=native_module(args.method,module,model)
    label='quadruple600_'+args.method+'_'+args.phase+'_seed0_'+args.run_tag
    export_root=args.root/'outputs/checkpoints'/label
    export_root.mkdir(parents=True,exist_ok=False)
    write(export_root/'experiment.json',dict(binding=expected,model=cfg,argv=sys.argv))
    if stats is not None:write(args.output/'action_statistics.json',stats)
    class Evidence(pl.pytorch.callbacks.Callback):
        def __init__(self):self.last_schedule_index=None
        def on_before_optimizer_step(self,trainer,pl_module,optimizer):
            if args.method=='pldm':
                self.last_schedule_index=author_schedule_before_step(pl_module,trainer.global_step)
        def on_after_backward(self,trainer,pl_module):
            if args.phase=='gate':
                gradients=[p.grad for p in pl_module.model.parameters() if p.requires_grad]
                if not any(x is not None for x in gradients) or any(x is not None and not bool(torch.isfinite(x).all()) for x in gradients):
                    raise ValueError('gate missing/nonfinite gradients')
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            if args.phase=='gate':
                if not bool(torch.isfinite(outputs['loss']).all()):raise ValueError('gate loss is nonfinite')
                with (args.output/'finite_losses.jsonl').open('a') as f:f.write(json.dumps(dict(phase='train',step=int(trainer.global_step),loss=float(outputs['loss'])))+'\n')
        def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
            if args.phase=='gate':
                if not bool(torch.isfinite(outputs['loss']).all()):raise ValueError('gate validation loss nonfinite')
                with (args.output/'finite_losses.jsonl').open('a') as f:f.write(json.dumps(dict(phase='val',step=int(trainer.global_step),sanity=trainer.sanity_checking,loss=float(outputs['loss'])))+'\n')
        def on_save_checkpoint(self,trainer,pl_module,checkpoint):
            checkpoint['baseline_binding']=expected
            checkpoint['native_method_schedule']=dict(method=args.method,last_before_optimizer_step=self.last_schedule_index,
                total_updates=contract(args.method)['total_updates'],warmup_updates=contract(args.method)['total_updates']//10)
        def state_dict(self):return dict(last_schedule_index=self.last_schedule_index)
        def load_state_dict(self,state):self.last_schedule_index=state.get('last_schedule_index')
    callbacks=[exports.SaveCkptCallback(label,OmegaConf.create(cfg),epoch_interval=1),Evidence()]
    options=trainer_options(args.phase,callbacks)
    trainer=pl.Trainer(**options);data=spt.data.DataModule(train=train,val=val)
    write(args.output/'result/actual_protocol.json',dict(binding=expected,loader_counts=contract(args.method),
         actual_train_batches=len(train),actual_validation_batches=len(val),export_root=str(export_root),
         model_config=cfg,trainer_options={k:v for k,v in options.items() if k!='callbacks'},
         optimizer='native method builder',scheduler='native PLDM before optimizer; native LeJEPA step after optimizer'))
    spt.Manager(trainer=trainer,module=world,data=data,ckpt_path=None,seed=0)()
    preprocessing=train.dataset.gpu_transform.last_contract
    if preprocessing is None:raise ValueError('RGB preprocessing was not applied before the model')
    return dict(stage='baseline_train_completed',export_root=str(export_root),training_returned=True,
         preprocessing_contract=preprocessing)


def accept(args):
    """Strict CPU factory reload; no optimizer update, raw scan or full validation rerun."""
    import torch
    protocol=read(args.training_output/'result/actual_protocol.json');expected=2 if args.phase=='gate' else contract(args.method)['total_updates']
    if (args.training_output/'actual_exit_code').read_text().strip()!='0':raise ValueError('own trainer actual0 required')
    log=(args.training_output/'stdout.log').read_text(errors='replace')
    import re
    runs=set(re.findall(r'/[^\s\x1b]*?/runs/\d{8}/\d{6}/[0-9a-f]+',log))
    runs={x.rstrip('/.,:') for x in runs if Path(x,'run_meta.json').exists()}
    if len(runs)!=1:raise ValueError('one own native run required')
    run=Path(runs.pop());fullpath=run/'checkpoints/last.ckpt';full=torch.load(fullpath,map_location='cpu',weights_only=False)
    if full.get('global_step')!=expected or Path(full['spt_run_dir'])!=run or full.get('baseline_binding')!=protocol['binding']:
        raise ValueError('own checkpoint counter/run/binding differs')
    if args.method=='pldm' and full['native_method_schedule']['last_before_optimizer_step']!=expected-1:
        raise ValueError('native PLDM before-step schedule counter differs')
    if args.method=='lejepa' and full['lr_schedulers'][0]['last_epoch']!=expected:
        raise ValueError('native LeJEPA scheduler counter differs')
    steps=[float(x['step']) for opt in full['optimizer_states'] for x in opt['state'].values() if 'step' in x]
    if not steps or set(steps)!={expected}:raise ValueError('Adam counters differ')
    def finite(x):
        if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        if isinstance(x,(tuple,list)):return all(finite(v) for v in x)
        return True
    if not finite(full['state_dict']) or not finite(full['optimizer_states']):raise ValueError('nonfinite model/optimizer')
    export=Path(protocol['export_root']);epochs=[1] if args.phase=='gate' else list(range(1,11))
    paths=[export/f'weights_epoch_{e}.pt' for e in epochs]
    if set(export.glob('weights_epoch_*.pt'))!=set(paths):raise ValueError('own exact epoch exports differ')
    bare=torch.load(paths[-1],map_location='cpu',weights_only=True)
    extracted={k[6:]:v for k,v in full['state_dict'].items() if k.startswith('model.')}
    if bare.keys()!=extracted.keys() or any(not torch.equal(bare[k],extracted[k]) for k in bare):raise ValueError('full/bare differ')
    model=method_module(args.method).build_model(protocol['model_config']['official_source'],pretrained=False)
    result=model.load_state_dict(bare,strict=True)
    if result.missing_keys or result.unexpected_keys or any(not torch.equal(model.state_dict()[k],bare[k]) for k in bare):raise ValueError('strict factory reload differs')
    if args.phase=='gate':
        records=[json.loads(x) for x in (args.training_output/'result/finite_losses.jsonl').read_text().splitlines()]
        if len([x for x in records if x['phase']=='train'])!=2 or not any(x['phase']=='val' and not x['sanity'] and x['step']>0 for x in records):raise ValueError('real post-update gate validation missing')
    else:
        val=full['loops']['fit_loop']['epoch_loop.val_loop.batch_progress']
        summary=read(run/'summary.json')
        if full['epoch'] not in (9,10) or val['total']['completed']!=contract(args.method)['validation_batches'] or summary['metrics']['validate/loss_epoch']['count']<10:
            raise ValueError('ten complete native train/validation epochs required')
    return dict(stage='baseline_'+args.phase+'_accepted',global_step=expected,Adam_steps=sorted(set(steps)),
                own_run=str(run),full_checkpoint=dict(path=str(fullpath),sha256=sha(fullpath)),
                epoch_exports=[dict(epoch=e,path=str(p),sha256=sha(p)) for e,p in zip(epochs,paths)],
                strict_factory_reload=True,full_bare_exact=True,binding=protocol['binding'],extra_optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','data-root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--method',choices=('pldm','lejepa'),required=True)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--phase',choices=('gate','formal'),required=True)
    parser.add_argument('--mode',choices=('train','accept'),required=True)
    parser.add_argument('--training-output',type=Path)
    args=parser.parse_args();args.root=args.root.resolve();args.data_root=args.data_root.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['STABLEWM_HOME']=str(args.root/'outputs');os.environ['SPT_CACHE_DIR']=str(args.root/'outputs/checkpoints/_spt_cache')
    state=dict(stage='baseline_'+args.mode+'_running',started_at=now(),updated_at=now(),actual_exit_code=None,argv=sys.argv)
    write(args.output/'status.json',state)
    try:
        require_sources(args.root,args.data_root)
        if args.mode=='accept' and args.training_output is None:raise ValueError('explicit own training output required')
        state.update(run(args) if args.mode=='train' else accept(args));state['actual_exit_code']=0
    except Exception:state.update(stage='baseline_phase_failed',actual_exit_code=1,error=traceback.format_exc())
    state['finished_at']=now();write(args.output/'status.json',state)
    (args.output/'actual_exit_code').write_text(str(state['actual_exit_code'])+'\n');return state['actual_exit_code']


if __name__=='__main__':raise SystemExit(main())
