"""Thin protected-epoch2/read-task continuation. Prepare-only by default."""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
BASE_FREEZE='cdfd14b894fda7bf7f068c489bc550e819ade12ca16f36250ebf0dda2cd374d7'
F2B_FREEZE='f2b1d8cb7e7c752f3aa2f485d25ffd8e14d26662937edad8d149ff165c9731b2'
F2B_SCRIPT='777be3d4c91ad5953861a70a386355369947695c792b6ab7efe38e52aa46ab37'
MANIFEST='98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
PARENT_UPDATES,TOTAL_UPDATES=91406,457030
_BASE=None
_PROFILE=None


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def base():
    global _BASE
    if _BASE is None:
        p=HERE.parent/'pldm_continuation_v1/continuation.py'
        if sha(p)!=F2B_SCRIPT:raise ValueError('frozen prior continuation differs')
        spec=importlib.util.spec_from_file_location('readtask_original_continuation',p)
        _BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_BASE;spec.loader.exec_module(_BASE)
        _BASE.HERE=HERE;_BASE.PARENT_UPDATES=PARENT_UPDATES
        _BASE.protocol=protocol;_BASE.require_runtime=require_runtime
        _BASE.run_train=run_train;_BASE.run_accept=run_accept
    return _BASE


def protocol(p):
    keys={'name','read_batch_size','train_workers','train_prefetch','validation_workers','validation_prefetch','split_validation'}
    if set(p)!=keys or not isinstance(p['name'],str) or not p['name']:raise ValueError('exact selected profile schema required')
    shape=(p['read_batch_size'],p['train_workers'],p['train_prefetch'],p['split_validation'])
    if shape not in ((None,20,2,False),(32,20,2,True),(32,32,2,True),(32,32,4,True)) or (p['validation_workers'],p['validation_prefetch'])!=(20,2):
        raise ValueError('profile outside approved read-task resource choices')
    return dict(selected_profile=dict(p),batch_size=128,precision='bf16-mixed',epochs=10,
        parent_updates=PARENT_UPDATES,new_updates=TOTAL_UPDATES-PARENT_UPDATES,total_updates=TOTAL_UPDATES,
        train_workers=p['train_workers'],validation_workers=20,validation_prefetch=2,
        train_batches=45703,validation_batches=4571,validation_tail=40,new_validation_batches=8*4571)


def validate_parent(p2,p1,p1_sha):
    expected=dict(stage='pldm_epoch_boundary_verified',actual_exit_code=0,completed_epochs=2,
        global_step=PARENT_UPDATES,Adam_steps=[float(PARENT_UPDATES)],native_schedule_index=PARENT_UPDATES-1,
        model_bare_exact=True,model_optimizer_finite=True)
    if any(p2.get(k)!=v for k,v in expected.items()) or p2.get('epoch') not in (1,2):raise ValueError('complete protected epoch2 required')
    binding=p2['baseline_binding'];prior=binding.get('continuation',{})
    if (binding.get('method'),binding.get('source_freeze_sha256'),binding.get('manifest_sha256'),prior.get('source_freeze_sha256'),prior.get('parent_proof_sha256'))!=('pldm',BASE_FREEZE,MANIFEST,F2B_FREEZE,p1_sha):
        raise ValueError('actual epoch2 producer/data/epoch1 proof binding differs')
    rows=p2['epoch_exports']
    if len(rows)!=2 or [r['epoch'] for r in rows]!=[1,2] or [r['origin'] for r in rows]!=['inherited_parent','protected_parent']:
        raise ValueError('two distinct inherited/protected epoch records required')
    if (rows[0]['path'],rows[0]['sha256'])!=(p1['files']['bare']['path'],p1['files']['bare']['sha256']) or rows[0]['sha256']!=prior.get('parent_bare_sha256'):
        raise ValueError('external epoch1 export reference differs')
    if (rows[1]['path'],rows[1]['sha256'])!=(p2['files']['bare']['path'],p2['files']['bare']['sha256']):raise ValueError('protected epoch2 export reference differs')
    return p2


def install_loaders(train,val,p,adapter_factory):
    protocol(p);originals=(train,val);identity=[{k:getattr(x,k) for k in ('dataset','sampler','batch_sampler','generator')} for x in originals]
    split=p['read_batch_size'] is not None
    if split:
        adapter=adapter_factory()
        train=adapter.wrap_loader(train,workers=p['train_workers'],prefetch_factor=p['train_prefetch'],read_batch_size=32,measure=False)
        val=adapter.wrap_loader(val,workers=20,prefetch_factor=2,read_batch_size=32,measure=False)
    for i,x in enumerate((train,val)):
        if x is not originals[i] or any(getattr(x,k) is not v for k,v in identity[i].items()):raise ValueError('reader install replaced original sampler/dataset/generator')
        if (x.batch_size,len(x),x.drop_last,x.pin_memory,x.persistent_workers)!=(128,45703 if i==0 else 4571,i==0,True,False):raise ValueError('outer native loader contract changed')
        if split:
            s=x.readtask_statistics
            if (s['read_batch_size'],s['workers'],s['prefetch_factor'])!=(32,p['train_workers'] if i==0 else 20,p['train_prefetch'] if i==0 else 2):raise ValueError('actual inner read-task resources differ')
        elif (x.num_workers,x.prefetch_factor)!=(20,2):raise ValueError('native selected path must stay unchanged W20PF2')
    return train,val,dict(selected_profile=dict(p),split_installed=split,outer_batch_size=128,
        train_batches=len(train),validation_batches=len(val),validation_tail=40,
        original_dataset_sampler_batch_sampler_generator_preserved=True,
        train_readtask=dict(train.readtask_statistics) if split else None,
        validation_readtask=dict(val.readtask_statistics) if split else None)


def require_runtime(a):
    global _PROFILE
    old=base();trial,data_root,freeze=old.source_guard(a.root)
    output=a.output if a.mode=='full' else a.output.parents[2]
    release=read(a.release)
    expected=dict(stage='approved_pldm_readtask_epoch2_continuation',output=str(output),
        source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),parent_proof=str(a.parent_proof),parent_proof_sha256=sha(a.parent_proof),
        trial_report=str(a.trial_report),trial_report_sha256=sha(a.trial_report))
    if any(release.get(k)!=v for k,v in expected.items()):raise ValueError('separate matching Root read-task continuation release required')
    probe=old.load('readtask_closed_probe',a.root/'readtask_probe_v1/trial.py')
    report=probe.verify_closed_trial(a.trial_report,a.parent_proof);p=report['selected_profile'];protocol(p)
    if release.get('selected_profile')!=p:raise ValueError('Root selection differs from actual recomputed trial')
    p1path=a.root/'outputs/pldm_worker_boundary_v1/preserved_epoch1/status.json'
    p1=trial.validate_parent(read(p1path));parent=validate_parent(read(a.parent_proof),p1,sha(p1path))
    records=list(parent['files'].values())+[p1['files']['bare']]
    for item in records:
        s=Path(item['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('protected parent file stat changed')
    from serial_relay import require_prerequisite
    if require_prerequisite(a.root/'prerequisites/triple_complete_ready.json',data_root) is None:raise ValueError('original data prerequisite absent')
    if a.mode=='accept' and a.training_output is None:raise ValueError('explicit own training output required')
    _PROFILE=p
    return trial,data_root,parent,p,freeze


def run_train(a,trial,data_root,parent,p,freeze):
    import torch
    import lightning as pl
    import stable_pretraining as spt
    old=base();entry=old.load('readtask_native_train',a.root/'train.py')
    native_loaders,native_binding,native_manager=entry.loaders,entry.binding,spt.Manager
    if sha(inspect.getfile(native_manager))!=freeze['runtime_manager_sha256']:raise ValueError('native Manager differs')
    manager_ref={};observed={}
    def loaders(*args,**kwargs):
        train,val,stats=native_loaders(*args,**kwargs)
        if stats!=parent['baseline_binding']['action_statistics']:raise ValueError('action normalization differs')
        train,val,meta=install_loaders(train,val,p,lambda:old.load('readtask_frozen_adapter',a.root/'readtask_split32_v1/adapter.py'))
        old.write(a.output/'loader_resource_record.json',meta);return train,val,stats
    def binding(*args):
        return dict(native_binding(*args),train_workers=p['train_workers'],prefetch_factor=p['train_prefetch'],validation_prefetch_factor=2,
            continuation=dict(source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),parent_proof_sha256=sha(a.parent_proof),
                parent_full_sha256=parent['files']['full']['sha256'],parent_bare_sha256=parent['files']['bare']['sha256'],
                inherited_epoch_exports=parent['epoch_exports'],selected_profile=p,trial_report_sha256=sha(a.trial_report),
                diagnostic_updates_discarded=True,random_stream_policy='seed0/new loader; not exact original RNG/sample continuation'))
    class ResumeReadback(pl.pytorch.callbacks.Callback):
        def on_train_start(self,trainer,module):
            own=Path(manager_ref['value']._run_dir).resolve()
            if not own.is_relative_to(a.output.parents[2]/'_spt_cache') or own==Path(parent['own_run']).resolve():raise ValueError('new ownrun isolation differs')
            steps=[float(s['step']) for opt in trainer.optimizers for s in opt.state.values() if 'step' in s]
            evidence=[cb for cb in trainer.callbacks if hasattr(cb,'last_schedule_index')]
            if trainer.global_step!=PARENT_UPDATES or trainer.current_epoch!=2 or not steps or set(steps)!={float(PARENT_UPDATES)} or len(evidence)!=1 or evidence[0].last_schedule_index!=PARENT_UPDATES-1:raise ValueError('epoch2 full/Adam/native schedule restore differs')
            if trial.fingerprint(module.model.state_dict())!=parent['model_fingerprint'] or trial.fingerprint(trainer.optimizers[0].state_dict())!=parent['optimizer_fingerprint']:raise ValueError('complete parent model/Adam restore differs')
            for cb in trainer.callbacks:
                if isinstance(cb,pl.pytorch.callbacks.ModelCheckpoint) and not Path(cb.dirpath).resolve().is_relative_to(own):raise ValueError('checkpoint escaped new ownrun')
            observed.update(parent_full=parent['files']['full'],inherited_epoch_exports=parent['epoch_exports'],own_run=str(own),
                before_first_update=dict(epoch_index=2,global_step=PARENT_UPDATES,Adam_steps=sorted(set(steps)),native_schedule_index=evidence[0].last_schedule_index))
            old.write(a.output/'resume_readback.json',observed)
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if 'first_update' not in observed:
                steps=[float(s['step']) for opt in trainer.optimizers for s in opt.state.values() if 'step' in s]
                if trainer.global_step!=PARENT_UPDATES+1 or not steps or set(steps)!={float(PARENT_UPDATES+1)}:raise ValueError('first new optimizer update differs')
                observed['first_update']=dict(epoch_index=int(trainer.current_epoch),global_step=int(trainer.global_step),Adam_steps=sorted(set(steps)))
                old.write(a.output/'resume_readback.json',observed)
    def manager(**kwargs):
        kwargs.update(ckpt_path=parent['files']['full']['path'],weights_only=False);kwargs['trainer'].callbacks.append(ResumeReadback())
        value=native_manager(**kwargs);manager_ref['value']=value;return value
    entry.loaders,entry.binding,spt.Manager=loaders,binding,manager
    try:return entry.run(SimpleNamespace(root=a.root,data_root=data_root,output=a.output,method='pldm',phase='formal',run_tag=a.output.parents[2].name))
    finally:spt.Manager=native_manager


def acceptance_tree(source):
    counts=[0,0,0]
    class Patch(ast.NodeTransformer):
        def visit_FunctionDef(self,node):return self.generic_visit(node) if node.name=='accept' else node
        def visit_Assign(self,node):
            if ast.unparse(node)=="paths = [export / f'weights_epoch_{e}.pt' for e in epochs]":
                counts[0]+=1;return ast.copy_location(ast.parse("paths = [Path(p) for p in READTASK_PARENT_BARES] + [export / f'weights_epoch_{e}.pt' for e in epochs[2:]]").body[0],node)
            return self.generic_visit(node)
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text=="set(export.glob('weights_epoch_*.pt')) != set(paths)":counts[1]+=1;node.comparators=[ast.parse('set(paths[2:])',mode='eval').body]
            elif text=="summary['metrics']['validate/loss_epoch']['count'] < 10":counts[2]+=1;node.comparators=[ast.Constant(8)]
            return self.generic_visit(node)
    tree=Patch().visit(ast.parse(source))
    if counts!=[1,1,1]:raise ValueError('native acceptance AST boundaries changed')
    return ast.fix_missing_locations(tree)


def lineage(rows,parents):
    if [r['epoch'] for r in rows]!=list(range(1,11)) or any((rows[i]['path'],rows[i]['sha256'])!=(parents[i]['path'],parents[i]['sha256']) for i in (0,1)):raise ValueError('actual parent1/2 plus produced3..10 ledger differs')
    return [dict(row,origin=parents[i]['origin'] if i<2 else 'produced_by_this_run') for i,row in enumerate(rows)]


def run_accept(a,parent):
    old=base();p=a.root/'train.py';spec=importlib.util.spec_from_file_location('readtask_native_accept',p)
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry
    entry.READTASK_PARENT_BARES=[r['path'] for r in parent['epoch_exports']]
    exec(compile(acceptance_tree(p.read_text()),str(p),'exec'),entry.__dict__)
    value=entry.accept(SimpleNamespace(root=a.root,method='pldm',phase='formal',training_output=a.training_output))
    observed=read(a.training_output/'result/resume_readback.json');resources=read(a.training_output/'result/loader_resource_record.json')
    if observed['parent_full']!=parent['files']['full'] or observed['inherited_epoch_exports']!=parent['epoch_exports'] or observed['before_first_update']['global_step']!=PARENT_UPDATES or observed['first_update']['global_step']!=PARENT_UPDATES+1 or resources['selected_profile']!=_PROFILE:raise ValueError('actual epoch2/reader restore provenance differs')
    summary=read(Path(value['own_run'])/'summary.json')
    if summary['metrics']['validate/loss_step']['count']<8*4571:raise ValueError('eight new complete validation observations missing')
    value.update(export_lineage=lineage(value['epoch_exports'],parent['epoch_exports']),parent_proof_sha256=sha(a.parent_proof),
        parent_optimizer_updates=PARENT_UPDATES,new_optimizer_updates=TOTAL_UPDATES-PARENT_UPDATES,
        inherited_complete_validation_epochs=2,new_complete_validation_epochs=8,selected_profile=_PROFILE,
        random_stream_policy='seed0/new loader; no exact original RNG/sample continuation')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','full','train','accept'),default='prepare')
    for key in ('root','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('release','parent-proof','trial-report','training-output'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
    old=base()
    if a.mode=='prepare':
        old.source_guard(a.root)
        print(json.dumps(dict(stage='prepared_only_no_workload',parent_updates=PARENT_UPDATES,new_updates=TOTAL_UPDATES-PARENT_UPDATES,
            total_updates=TOTAL_UPDATES,epochs=10,validation_batches_each=4571,new_full_validation_epochs=8,next_method_started=False)));return 0
    return old.main()


if __name__=='__main__':raise SystemExit(main())
