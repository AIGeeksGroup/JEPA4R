"""Explicit parent1 + new2..10 PLDM continuation; prepare is the default."""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PARENT_UPDATES,TOTAL_UPDATES=45703,457030


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def protocol(workers):
    if workers not in (20,24):raise ValueError('only actual ABA-selected W20/24 allowed')
    return dict(train_workers=workers,validation_workers=20,batch_size=128,precision='bf16-mixed',
        epochs=10,parent_updates=PARENT_UPDATES,new_updates=TOTAL_UPDATES-PARENT_UPDATES,
        total_updates=TOTAL_UPDATES,train_batches=45703,validation_batches=4571,validation_tail=40)


def selected_workers(report,trial):
    if (report.get('stage'),report.get('exit_code'),report.get('all_diagnostic_updates_discarded'))!=(
        'worker_ABA_complete_awaiting_Root_formal_release',0,True):raise ValueError('actual completed ABA0 required')
    rows=[report['cases'][name] for name,_ in trial.CASES]
    if any(r.get('actual_exit_code')!=0 for r in rows):raise ValueError('every own ABA case actual0 required')
    trial.compare_cohort(rows)
    chosen=trial.recommend_workers([r['updates_per_second'] for r in rows])['recommended_train_workers']
    if report.get('recommended_train_workers')!=chosen:raise ValueError('actual ABA recommendation differs')
    protocol(chosen);return chosen


def lineage(rows,parent):
    if [r['epoch'] for r in rows]!=list(range(1,11)) or rows[0]['path']!=str(parent):
        raise ValueError('parent1 plus produced2..10 ledger required')
    return [dict(r,origin='inherited_parent' if r['epoch']==1 else 'produced_by_this_run') for r in rows]


def source_guard(root):
    freeze=read(HERE/'source_freeze_v1.json')
    if str(root)!=freeze['root']:raise ValueError('actual root differs')
    for name,h in freeze['files'].items():
        if sha(HERE/name)!=h:raise ValueError('continuation source differs: '+name)
    for name,h in freeze['reference_files'].items():
        if sha(root/name)!=h:raise ValueError('frozen reference differs: '+name)
    sys.path.insert(0,str(root))
    trial=load('continuation_frozen_trial',root/'pldm_worker_trial_v1/trial.py')
    data_root=trial.source_guard(root)
    return trial,data_root,freeze


def require_runtime(a):
    trial,data_root,freeze=source_guard(a.root);release=read(a.release)
    output=a.output if a.mode=='full' else a.output.parents[2]
    expected=dict(stage='approved_pldm_epoch1_continuation',output=str(output),
        source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),parent_proof=str(a.parent_proof),
        parent_proof_sha256=sha(a.parent_proof),trial_report=str(a.trial_report),trial_report_sha256=sha(a.trial_report))
    if any(release.get(k)!=v for k,v in expected.items()):raise ValueError('matching separate Root full-continuation release required')
    parent=trial.validate_parent(read(a.parent_proof));report=read(a.trial_report)
    workers=selected_workers(report,trial)
    if release.get('selected_train_workers')!=workers:raise ValueError('Root workers differ from actual ABA')
    for item in parent['files'].values():
        s=Path(item['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('protected parent stat differs')
    directory=a.trial_report.parent
    for p in (directory/'actual_exit_code',Path(str(directory)+'.outer_exit_code')):
        if p.read_text().strip()!='0':raise ValueError('actual ABA driver/outer0 required')
    for name,_ in trial.CASES:
        case=directory/name
        if (case/'actual_exit_code').read_text().strip()!='0' or not (case/'process_group_drain.json').exists():
            raise ValueError('actual ABA case exit/drain evidence missing')
        row=read(case/'result/trial_result.json')
        if row!=report['cases'][name] or row['parent_proof_sha256']!=sha(a.parent_proof):
            raise ValueError('actual trial case/parent evidence differs')
    from serial_relay import require_prerequisite
    if require_prerequisite(a.root/'prerequisites/triple_complete_ready.json',data_root) is None:
        raise ValueError('original Triple completion prerequisite missing')
    return trial,data_root,parent,workers,freeze


def run_train(a,trial,data_root,parent,workers,freeze):
    import torch
    import lightning as pl
    import stable_pretraining as spt
    entry=load('pldm_continuation_native',a.root/'train.py')
    native_loaders,native_binding,native_manager=entry.loaders,entry.binding,spt.Manager
    if sha(inspect.getfile(native_manager))!=freeze['runtime_manager_sha256']:
        raise ValueError('actual native Manager version differs')
    manager_ref={};readback={}
    def loaders(*args,**kwargs):
        train,val,stats=native_loaders(*args,**kwargs)
        if stats!=parent['baseline_binding']['action_statistics']:raise ValueError('parent action statistics differ')
        train.num_workers=workers
        if (train.batch_size,val.batch_size,val.num_workers,train.prefetch_factor,val.prefetch_factor,
            train.pin_memory,val.pin_memory,train.persistent_workers,val.persistent_workers)!=(128,128,20,2,2,True,True,False,False):
            raise ValueError('native loader protocol differs')
        return train,val,stats
    def binding(*args):
        return dict(native_binding(*args),train_workers=workers,continuation=dict(
            source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),parent_proof_sha256=sha(a.parent_proof),
            parent_full_sha256=parent['files']['full']['sha256'],parent_bare_sha256=parent['files']['bare']['sha256'],
            trial_report_sha256=sha(a.trial_report),diagnostic_updates_discarded=True,
            random_stream_policy='seed0 and rebuilt loader generator; not exact original sample/RNG continuation'))
    class ResumeReadback(pl.pytorch.callbacks.Callback):
        def on_train_start(self,trainer,module):
            own=Path(manager_ref['value']._run_dir).resolve()
            if not own.is_relative_to(a.output.parents[2]/'_spt_cache') or own==Path(parent['own_run']).resolve():raise ValueError('continuation ownrun escaped isolated output')
            if trainer.global_step!=PARENT_UPDATES or trainer.current_epoch!=1:raise ValueError('full epoch/global restore differs')
            steps=[float(x['step']) for opt in trainer.optimizers for x in opt.state.values() if 'step' in x]
            evidence=[cb for cb in trainer.callbacks if hasattr(cb,'last_schedule_index')]
            if not steps or set(steps)!={float(PARENT_UPDATES)} or len(evidence)!=1 or evidence[0].last_schedule_index!=PARENT_UPDATES-1:
                raise ValueError('parent Adam/native schedule state restore differs')
            if trial.fingerprint(module.model.state_dict())!=parent['model_fingerprint'] or trial.fingerprint(trainer.optimizers[0].state_dict())!=parent['optimizer_fingerprint']:
                raise ValueError('full model/Adam restore differs')
            for cb in trainer.callbacks:
                if isinstance(cb,pl.pytorch.callbacks.ModelCheckpoint) and not Path(cb.dirpath).resolve().is_relative_to(own):raise ValueError('checkpoint callback escaped new ownrun')
            readback.update(parent_full=parent['files']['full'],own_run=str(own),before_first_update=dict(epoch_index=1,
                global_step=int(trainer.global_step),Adam_steps=sorted(set(steps)),native_schedule_index=evidence[0].last_schedule_index))
            write(a.output/'resume_readback.json',readback)
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if 'first_update' not in readback:
                steps=[float(x['step']) for opt in trainer.optimizers for x in opt.state.values() if 'step' in x]
                if trainer.global_step!=PARENT_UPDATES+1 or not steps or set(steps)!={float(PARENT_UPDATES+1)}:raise ValueError('first restored optimizer update differs')
                readback['first_update']=dict(epoch_index=int(trainer.current_epoch),global_step=int(trainer.global_step),Adam_steps=sorted(set(steps)))
                write(a.output/'resume_readback.json',readback)
    def manager(**kwargs):
        kwargs.update(ckpt_path=parent['files']['full']['path'],weights_only=False)
        kwargs['trainer'].callbacks.append(ResumeReadback())
        value=native_manager(**kwargs);manager_ref['value']=value;return value
    entry.loaders,entry.binding,spt.Manager=loaders,binding,manager
    try:return entry.run(SimpleNamespace(root=a.root,data_root=data_root,output=a.output,
        method='pldm',phase='formal',run_tag=a.output.parents[2].name))
    finally:spt.Manager=native_manager


def acceptance_tree(source):
    counts=[0,0,0];tree=ast.parse(source)
    class Patch(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            return self.generic_visit(node) if node.name=='accept' else node
        def visit_Assign(self,node):
            if ast.unparse(node)=="paths = [export / f'weights_epoch_{e}.pt' for e in epochs]":
                counts[0]+=1
                return ast.copy_location(ast.parse("paths = [Path(CONTINUATION_PARENT_BARE)] + [export / f'weights_epoch_{e}.pt' for e in epochs[1:]]").body[0],node)
            return self.generic_visit(node)
        def visit_Compare(self,node):
            value=ast.unparse(node)
            if value=="set(export.glob('weights_epoch_*.pt')) != set(paths)":
                counts[1]+=1;node.comparators=[ast.parse('set(paths[1:])',mode='eval').body]
            elif value=="summary['metrics']['validate/loss_epoch']['count'] < 10":
                counts[2]+=1;node.comparators=[ast.Constant(9)]
            return self.generic_visit(node)
    tree=Patch().visit(tree)
    if counts!=[1,1,1]:raise ValueError('native acceptance AST boundaries changed: '+str(counts))
    return ast.fix_missing_locations(tree)


def run_accept(a,parent):
    spec=importlib.util.spec_from_file_location('pldm_continuation_accept',a.root/'train.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry
    entry.CONTINUATION_PARENT_BARE=parent['files']['bare']['path']
    exec(compile(acceptance_tree((a.root/'train.py').read_text()),str(a.root/'train.py'),'exec'),entry.__dict__)
    value=entry.accept(SimpleNamespace(root=a.root,method='pldm',phase='formal',training_output=a.training_output))
    observed=read(a.training_output/'result/resume_readback.json')
    if observed['parent_full']!=parent['files']['full'] or observed['before_first_update']['global_step']!=PARENT_UPDATES or observed['first_update']['global_step']!=PARENT_UPDATES+1:
        raise ValueError('actual protected parent resume evidence differs')
    own=Path(value['own_run']);summary=read(own/'summary.json')
    if summary['metrics']['validate/loss_step']['count']<9*4571:raise ValueError('nine new full validation observations missing')
    value.update(export_lineage=lineage(value['epoch_exports'],parent['files']['bare']['path']),
        parent_proof_sha256=sha(a.parent_proof),parent_optimizer_updates=PARENT_UPDATES,new_optimizer_updates=TOTAL_UPDATES-PARENT_UPDATES,
        new_complete_validation_epochs=9,inherited_complete_validation_epochs=1,random_stream_policy='seed0/new loader; no exact RNG or sampler continuation claim')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','full','train','accept'),default='prepare')
    for key in ('root','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('release','parent-proof','trial-report','training-output'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
    if a.mode=='prepare':
        source_guard(a.root);print(json.dumps(dict(stage='prepared_only_no_workload',protocols=[protocol(w) for w in (20,24)],next_method_started=False)));return 0
    if any(getattr(a,k) is None for k in ('release','parent_proof','trial_report')):raise ValueError('explicit Root release/parent/trial report required')
    for k in ('release','parent_proof','trial_report'):setattr(a,k,getattr(a,k).resolve())
    trial,data_root,parent,workers,freeze=require_runtime(a)
    os.environ['STABLEWM_HOME']=str(a.root/'outputs')
    full_output=a.output if a.mode=='full' else a.output.parents[2]
    os.environ['SPT_CACHE_DIR']=str(full_output/'_spt_cache')
    if a.mode in ('train','accept'):
        a.output.mkdir(parents=True,exist_ok=False);state=dict(stage='pldm_continuation_'+a.mode+'_running',actual_exit_code=None,argv=sys.argv)
        try:
            state.update(run_train(a,trial,data_root,parent,workers,freeze) if a.mode=='train' else run_accept(a,parent));state['actual_exit_code']=0
        except BaseException:state.update(stage='pldm_continuation_phase_failed',actual_exit_code=1,error=traceback.format_exc())
        write(a.output/'status.json',state);(a.output/'actual_exit_code').write_text(str(state['actual_exit_code'])+'\n');return state['actual_exit_code']
    a.output.mkdir(parents=True,exist_ok=False);state=dict(stage='pldm_continuation_starting',exit_code=None,started_at=time.time(),protocol=protocol(workers))
    old=load('continuation_owned',data_root/'training_serial_compact_relay_v1/serial_relay.py')
    legacy=old.load_frozen(data_root/'training_relay_v1/relay.py',old.LEGACY_RELAY_SHA,'continuation_resources')
    args=SimpleNamespace(root=data_root,code_root=data_root/'code_front600_v1/lewm_pcd',output=a.output)
    try:
        for mode in ('train','accept'):
            old.gpu_empty();directory=a.output/('formal' if mode=='train' else 'acceptance_process')/'pldm'
            cmd=[sys.executable,'-B',str(HERE/'continuation.py'),'--mode',mode,'--root',str(a.root),'--output',str(directory/'result'),
                '--release',str(a.release),'--parent-proof',str(a.parent_proof),'--trial-report',str(a.trial_report)]
            if mode=='accept':cmd=['env','CUDA_VISIBLE_DEVICES=',*cmd,'--training-output',str(a.output/'formal/pldm')]
            old.run_owned(cmd,directory,args,state,legacy)
        proof=read(a.output/'acceptance_process/pldm/result/status.json')
        if proof.get('actual_exit_code')!=0 or proof.get('stage')!='baseline_formal_accepted':raise ValueError('actual continuation strict acceptance0 missing')
        state.update(stage='pldm_full10_accepted_waiting_lejepa_uint8_release',exit_code=0,acceptance=proof,next_method_started=False)
    except BaseException:state.update(stage='pldm_continuation_failed_stopped',exit_code=1,error=traceback.format_exc())
    state['finished_at']=time.time();write(a.output/'status.json',state);(a.output/'actual_exit_code').write_text(str(state['exit_code'])+'\n');return state['exit_code']


if __name__=='__main__':raise SystemExit(main())
