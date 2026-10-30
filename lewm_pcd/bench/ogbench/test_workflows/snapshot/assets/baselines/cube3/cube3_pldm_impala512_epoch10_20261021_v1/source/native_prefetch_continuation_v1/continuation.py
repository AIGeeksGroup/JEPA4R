"""Native B128/PF2-or4 continuation from verified epoch4; prepare default."""
import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT_UPDATES,TOTAL_UPDATES=182812,457030
LEGACY_SHA='e9768305dded55793c9e3cd8e344f71a2439390c2513da7b335e5bc2837f7250'
EPOCH3_SCRIPT_SHA='b0305efb4b389cf32537795e57bd94747a0972312931efa725bfe695a1815a86'
EPOCH3_FREEZE='3738c2d5f8fadd1f6824973dec1805da5e2f6a7e87371af3a5eadc6e78bf075f'
_API=None


def epoch3_reference():
    p=HERE.parent/'readtask_restart_continuation_v2/continuation.py'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=EPOCH3_SCRIPT_SHA:raise ValueError('frozen epoch3 reference differs')
    s=importlib.util.spec_from_file_location('native_pf_reference_epoch3',p)
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m


def native_tree(source):
    tree=epoch3_reference().epoch3_tree(source);counts=[0]*7
    class Patch(ast.NodeTransformer):
        active=None
        def visit_FunctionDef(self,node):
            if self.active is not None:return self.generic_visit(node)
            if node.name not in ('protocol','run_train','acceptance_tree','run_accept','main'):return node
            self.active=node.name;node=self.generic_visit(node);self.active=None;return node
        def visit_Compare(self,node):
            if self.active=='run_train' and ast.unparse(node)=='trainer.current_epoch != 3':
                counts[0]+=1;node.comparators=[ast.Constant(4)]
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if self.active=='run_train' and node.arg=='epoch_index' and ast.unparse(node.value)=='3':counts[1]+=1;node.value=ast.Constant(4)
            if self.active=='run_accept' and node.arg in ('inherited_complete_validation_epochs','new_complete_validation_epochs'):
                expected=3 if node.arg.startswith('inherited') else 7
                if ast.unparse(node.value)!=str(expected):raise ValueError('native accept epoch metadata differs')
                counts[2]+=1;node.value=ast.Constant(4 if expected==3 else 6)
            if self.active=='main' and node.arg=='new_full_validation_epochs':
                if ast.unparse(node.value)!='7':raise ValueError('prepare epoch boundary differs')
                counts[3]+=1;node.value=ast.Constant(6)
            return self.generic_visit(node)
        def visit_BinOp(self,node):
            if self.active in ('protocol','run_accept') and ast.unparse(node)=='7 * 4571':counts[4]+=1;node.left=ast.Constant(6)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if self.active=='acceptance_tree' and isinstance(node.value,str):
                value=node.value.replace('epochs[3:]','epochs[4:]').replace('paths[3:]','paths[4:]')
                if value!=node.value:counts[5]+=1;return ast.copy_location(ast.Constant(value),node)
            return node
        def visit_Call(self,node):
            if self.active=='acceptance_tree' and ast.unparse(node.func)=='ast.Constant' and len(node.args)==1 and ast.unparse(node.args[0])=='7':counts[6]+=1;node.args[0]=ast.Constant(6)
            return self.generic_visit(node)
    tree=Patch().visit(tree)
    if counts!=[1,1,2,1,2,2,1]:raise ValueError('frozen epoch4 adaptation boundaries differ: '+str(counts))
    return ast.fix_missing_locations(tree)


def protocol(p):
    keys={'name','read_batch_size','train_workers','train_prefetch','validation_workers','validation_prefetch','split_validation'}
    if set(p)!=keys or not isinstance(p['name'],str) or not p['name'] or p['read_batch_size'] is not None or p['split_validation'] is not False:
        raise ValueError('native-only actual selected profile required')
    if (p['train_workers'],p['train_prefetch'],p['validation_workers'],p['validation_prefetch']) not in ((20,2,20,2),(20,4,20,2)):
        raise ValueError('only trainW20PF2or4/valW20PF2 approved')
    return dict(selected_profile=dict(p),batch_size=128,precision='bf16-mixed',epochs=10,
        parent_updates=PARENT_UPDATES,new_updates=TOTAL_UPDATES-PARENT_UPDATES,total_updates=TOTAL_UPDATES,
        train_workers=20,validation_workers=20,validation_prefetch=2,train_batches=45703,
        validation_batches=4571,validation_tail=40,new_validation_batches=6*4571)


def install_loaders(train,val,p,unused_adapter_factory):
    protocol(p);identity=[{k:getattr(x,k) for k in ('dataset','sampler','batch_sampler','generator')} for x in (train,val)]
    for i,x in enumerate((train,val)):
        if (x.batch_size,len(x),x.num_workers,x.prefetch_factor,x.drop_last,x.pin_memory,x.persistent_workers)!=(128,45703 if i==0 else 4571,20,2,i==0,True,False):raise ValueError('original native loader contract differs')
        if hasattr(x,'_readtask_measure') or hasattr(x,'readtask_statistics'):raise ValueError('split loader is forbidden in native-only continuation')
    train.prefetch_factor=p['train_prefetch']
    if any(getattr(x,k) is not v for x,refs in zip((train,val),identity) for k,v in refs.items()):raise ValueError('native prefetch changed original loader objects')
    return train,val,dict(selected_profile=dict(p),split_installed=False,outer_batch_size=128,
        train_batches=len(train),validation_batches=len(val),validation_tail=40,
        original_dataset_sampler_batch_sampler_generator_preserved=True,
        train_prefetch_factor=train.prefetch_factor,validation_prefetch_factor=val.prefetch_factor,
        train_readtask=None,validation_readtask=None)


def lineage(rows,parents):
    if len(parents)!=4 or [r['epoch'] for r in rows]!=list(range(1,11)):raise ValueError('four inherited plus six produced epoch records required')
    if any((rows[i]['path'],rows[i]['sha256'])!=(parents[i]['path'],parents[i]['sha256']) for i in range(4)):raise ValueError('actual inherited epoch1..4 references differ')
    return [dict(row,origin=parents[i]['origin'] if i<4 else 'produced_by_this_run') for i,row in enumerate(rows)]


def api():
    global _API
    if _API is None:
        p=HERE.parent/'readtask_continuation_v1/continuation.py'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=LEGACY_SHA:raise ValueError('frozen original continuation differs')
        spec=importlib.util.spec_from_file_location('native_pf_reused_continuation',p)
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
        exec(compile(native_tree(p.read_text()),str(p),'exec'),m.__dict__)
        m.HERE=HERE;m.PARENT_UPDATES=PARENT_UPDATES;m.protocol=protocol
        m.install_loaders=install_loaders;m.lineage=lineage;m.require_runtime=require_runtime
        _API=m
    return _API


def require_runtime(a):
    m=api();old=m.base();trial,data_root,freeze=old.source_guard(a.root)
    output=a.output if a.mode=='full' else a.output.parents[2];release=m.read(a.release)
    expected=dict(stage='approved_pldm_native_prefetch_continuation',output=str(output),
        source_freeze_sha256=m.sha(HERE/'source_freeze_v1.json'),parent_proof=str(a.parent_proof),
        parent_proof_sha256=m.sha(a.parent_proof),trial_report=str(a.trial_report),trial_report_sha256=m.sha(a.trial_report))
    if any(release.get(k)!=v for k,v in expected.items()):raise ValueError('matching separate Root native-prefetch release required')
    probe=old.load('native_pf_closed_probe',a.root/'native_prefetch_probe_v1/trial.py')
    retirement=probe.require_retirement(release['retirement_receipt'],release['retirement_receipt_sha256'],check_live=a.mode in ('full','train'))
    if (retirement.get('parent_proof'),retirement.get('parent_proof_sha256'))!=(str(a.parent_proof),m.sha(a.parent_proof)):raise ValueError('actual retirement refers to another parent')
    report=probe.verify_closed_trial(a.trial_report,a.parent_proof);parent=probe.validate_parent(m.read(a.parent_proof))
    p=report['selected_profile'];protocol(p)
    if release.get('selected_profile')!=p:raise ValueError('Root native profile differs from actual verified trial')
    ref=parent['prior_parent_references']['P3']
    if m.sha(ref['proof_path'])!=ref['proof_sha256'] or ref['global_step']!=137109:raise ValueError('actual P3 proof reference differs')
    p3=m.read(ref['proof_path']);prior=parent['baseline_binding']['continuation']
    if prior.get('source_freeze_sha256')!=EPOCH3_FREEZE or prior.get('parent_proof_sha256')!=ref['proof_sha256']:
        raise ValueError('actual epoch4 producer/P3 binding differs')
    if parent['epoch_exports'][:3]!=p3['epoch_exports']:raise ValueError('real inherited P1/P2/P3 ledger was changed')
    bare=parent['files']['bare']
    if (parent['epoch_exports'][3]['path'],parent['epoch_exports'][3]['sha256'])!=(bare['path'],bare['sha256']):raise ValueError('protected epoch4 export differs')
    for item in parent['files'].values():
        s=Path(item['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('protected actual parent stat changed')
    from serial_relay import require_prerequisite
    if require_prerequisite(a.root/'prerequisites/triple_complete_ready.json',data_root) is None:raise ValueError('original accepted data prerequisite missing')
    if a.mode=='accept' and a.training_output is None:raise ValueError('explicit own training output required')
    m._PROFILE=p;return trial,data_root,parent,p,freeze


if __name__=='__main__':raise SystemExit(api().main())
