"""Native B128/W20 PF2 -> PF4 -> PF2. Default prepare; no split adapter."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE_FREEZE='cdfd14b894fda7bf7f068c489bc550e819ade12ca16f36250ebf0dda2cd374d7'
MANIFEST='98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
RESTART_SHA='189e834110bc279dbe253985c856757474bb02eb15ac5d3d125be21e94b013e4'
PARENT_UPDATES,TOTAL_UPDATES=182812,457030
NAMES=['A1_native20_pf2','B_native20_pf4','A2_native20_pf2']
PROFILES={name:dict(name=name,read_batch_size=None,train_workers=20,train_prefetch=pf,
    validation_workers=20,validation_prefetch=2,split_validation=False) for name,pf in zip(NAMES,[2,4,2])}
EXIT_KEYS=('trainer','driver','outer','prior_controller','prior_root_outer')
_API=None


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parent(value):
    expected=dict(stage='pldm_native_prefetch_parent_verified',actual_exit_code=0,completed_epochs=4,
        global_step=PARENT_UPDATES,Adam_steps=[float(PARENT_UPDATES)],native_schedule_index=PARENT_UPDATES-1,
        model_bare_exact=True,model_optimizer_finite=True,IDM_state_present=True,optimizer_all_parameters_have_state=True)
    if any(value.get(k)!=v for k,v in expected.items()) or value.get('epoch') not in (3,4):
        raise ValueError('actual complete protected epoch4 required')
    binding=value['baseline_binding'];progress=value['validation_batch_progress']
    if (binding.get('method'),binding.get('source_freeze_sha256'),binding.get('manifest_sha256'))!=('pldm',BASE_FREEZE,MANIFEST):
        raise ValueError('actual parent method/source/data binding differs')
    if [x['epoch'] for x in value['epoch_exports']]!=[1,2,3,4] or progress['total']['completed']!=4571 or not progress['is_last_batch']:
        raise ValueError('four actual epoch exports and complete last validation required')
    if value['prior_parent_references']['P3']['global_step']!=137109:
        raise ValueError('actual P3 lineage required')
    return value


def validate_retirement(value):
    expected=dict(stage='owned_epoch4_retired',actual_exit_code=0,owned_active_after=0,gpu_empty=True)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('actual Root epoch4 retirement0 required')
    for key in ('source_formal_output','parent_proof','parent_proof_sha256','original_identity','original_group_drain'):
        if not value.get(key):raise ValueError('missing retirement identity: '+key)
    if set(value['original_process_exit_codes'])!=set(EXIT_KEYS) or set(value['original_exit_paths'])!=set(EXIT_KEYS):
        raise ValueError('all actual original exit paths/codes required')
    if any(type(x) is not int for x in value['original_process_exit_codes'].values()):raise ValueError('unknown original exit code')
    return value


def require_retirement(receipt_path,expected_sha=None,*,check_live=False):
    if expected_sha is not None and sha(receipt_path)!=expected_sha:raise ValueError('retirement receipt SHA differs')
    value=validate_retirement(read(receipt_path));current=read(HERE/'current_identity_v1.json')
    if value['source_formal_output']!=current['formal_output'] or value['original_identity']!=current['identity']:
        raise ValueError('retirement original owned identity differs')
    validate_parent(read(value['parent_proof']))
    if sha(value['parent_proof'])!=value['parent_proof_sha256']:raise ValueError('retirement protected parent SHA differs')
    if check_live:
        for key in EXIT_KEYS:
            path=Path(value['original_exit_paths'][key])
            code=read(path)['actual_exit_code'] if path.suffix=='.json' else int(path.read_text().strip())
            if type(code) is not int or code!=value['original_process_exit_codes'][key]:
                raise ValueError('actual original exit differs: '+key)
        item=value['original_group_drain'];expected=Path(current['formal_output'])/'formal/pldm/process_group_drain.json'
        if Path(item['path'])!=expected or sha(expected)!=item['sha256']:raise ValueError('actual owned drain binding differs')
        drain=read(expected)
        if drain['stage']!='empty' or drain['active']:raise ValueError('original trainer group not drained')
        if Path('/proc',str(current['identity']['pid'])).exists():raise ValueError('original trainer locator present')
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if result.stdout.strip():raise ValueError('GPU must be empty before native prefetch trial')
    return value


def compare_cohort(rows):
    keys=('ids','first2_inputs','RNG_before','RNG_after','start_global_step','end_global_step','native_schedule_index','parent_proof_sha256')
    for row in rows:
        if (row['start_global_step'],row['end_global_step'],row['native_schedule_index'])!=(PARENT_UPDATES,PARENT_UPDATES+416,PARENT_UPDATES+415):
            raise ValueError('diagnostic update budget differs')
        if len(row['ids'])!=416 or any(len(x)!=128 for x in row['ids']) or len(row['first2_inputs'])!=2 or row.get('first2_pinned_outputs')!=[True,True]:
            raise ValueError('actual416 B128 consumption and first2 pinned CPU inputs required')
        if any(row[k]!=rows[0][k] for k in keys):raise ValueError('cohort/input/main RNG differs')


def adapt_loaders(train,val,profile):
    if profile not in PROFILES.values():raise ValueError('only native W20 PF2/PF4 profiles allowed')
    for loader,drop in ((train,True),(val,False)):
        if (loader.batch_size,loader.drop_last,loader.pin_memory,loader.persistent_workers)!=(128,drop,True,False):
            raise ValueError('native B128/pin/persistent/drop contract differs')
    if (val.num_workers,val.prefetch_factor)!=(20,2):raise ValueError('native validation W20/PF2 required')
    train.num_workers=20;train.prefetch_factor=profile['train_prefetch']
    return train,val


def loader_configuration(train,val):
    def fields(loader):
        return dict(loader_type=type(loader).__module__+'.'+type(loader).__qualname__,
            class_mro=[x.__module__+'.'+x.__qualname__ for x in type(loader).__mro__],
            batch_size=loader.batch_size,num_workers=loader.num_workers,prefetch_factor=loader.prefetch_factor,
            drop_last=loader.drop_last,pin_memory=loader.pin_memory,persistent_workers=loader.persistent_workers,
            batches=len(loader),readtask_statistics=dict(getattr(loader,'readtask_statistics',{})))
    result=dict(train=fields(train),validation=fields(val))
    check_loader_configuration(result)
    return result


def check_loader_configuration(value,profile=None):
    for key,drop,batches,pf in [('train',True,45703,None),('validation',False,4571,2)]:
        row=value[key]
        expected=(128,20,drop,True,False,batches)
        if tuple(row[k] for k in ('batch_size','num_workers','drop_last','pin_memory','persistent_workers','batches'))!=expected:
            raise ValueError('actual native loader metadata differs: '+key)
        if row['readtask_statistics'] or not any(x=='torch.utils.data.dataloader.DataLoader' for x in row['class_mro']):
            raise ValueError('native complete-batch DataLoader required')
        wanted=profile['train_prefetch'] if key=='train' and profile else pf
        if wanted is not None and row['prefetch_factor']!=wanted:raise ValueError('actual loader prefetch differs')
    return value


def resource_snapshot(parents):
    value=_RESOURCE_BASE(parents)
    # Same original run_owned cadence. Metadata only; no added per-update sync.
    cgroup=Path('/sys/fs/cgroup')
    for name in ('memory.stat','memory.pressure','io.pressure','cpu.pressure','cpu.stat'):
        path=cgroup/name
        if path.exists():value['cgroup_'+name]=path.read_text().strip()
    return value


def probe_tree(source):
    tree=base()._original_probe_tree(source);count=[0,0,0]
    class Patch(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.unparse(node)=='trainer.current_epoch != 2':count[0]+=1;node.comparators=[ast.Constant(4)]
            return self.generic_visit(node)
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='state.update' and any(k.arg=='measured_updates' for k in node.keywords):
                count[1]+=1;node.keywords.append(ast.keyword('native_loader_configuration',ast.parse("loader_configuration(loader_ref['train'],loader_ref['val'])",mode='eval').body))
            if ast.unparse(node.func)=='old.run_owned':
                count[2]+=1
                return ast.copy_location(ast.Call(ast.Name('run_owned_with_resources',ast.Load()),[ast.Name('old',ast.Load()),ast.Name('legacy',ast.Load())]+node.args,node.keywords),node)
            return node
        def visit_Constant(self,node):
            if node.value=='Protected epoch2 plus external1 + remaining8epochs; seed0/new loader; not exact original sample/RNG continuation':
                return ast.copy_location(ast.Constant('Protected epochs1..4 + remaining6epochs; seed0/new loader; not exact original sample/RNG continuation'),node)
            return node
    tree=Patch().visit(tree)
    if count!=[1,1,1]:raise ValueError('frozen native probe metadata boundaries changed: '+str(count))
    return ast.fix_missing_locations(tree)


def require_runtime(a):
    data_root=base().source_guard(a.root);release=read(a.release);output=a.output.parent if a.mode=='case' else a.output
    expected=dict(stage='approved_pldm_native_prefetch_trial',source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),
        output=str(output),parent_proof=str(a.parent_proof),parent_proof_sha256=sha(a.parent_proof))
    if any(release.get(k)!=v for k,v in expected.items()):raise ValueError('matching separate Root native prefetch trial release required')
    parent=validate_parent(read(a.parent_proof))
    for item in parent['files'].values():
        z=Path(item['path']).stat()
        if (z.st_size,z.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('protected epoch4 file stat differs')
    receipt=require_retirement(release['retirement_receipt'],release['retirement_receipt_sha256'],check_live=True)
    if (receipt['parent_proof'],receipt['parent_proof_sha256'])!=(str(a.parent_proof),sha(a.parent_proof)):
        raise ValueError('retirement/trial protected parent differs')
    if int(os.environ.get('SLURM_RESTART_COUNT','0')):raise ValueError('new isolated trial ownrun required')
    return data_root,parent


def base():
    global _API
    if _API is None:
        path=HERE.parent/'readtask_restart_probe_v2/trial.py'
        if sha(path)!=RESTART_SHA:raise ValueError('frozen restart probe changed')
        spec=importlib.util.spec_from_file_location('native_prefetch_restart_base',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        value=module.base();value.HERE=HERE;value.PARENT_UPDATES=PARENT_UPDATES
        value.NAMES=NAMES;value.PROFILES=PROFILES;value.probe_tree=probe_tree;value.validate_parent=validate_parent
        value.compare_cohort=compare_cohort;value.require_runtime=require_runtime;value.adapt_loaders=adapt_loaders
        original=value.old_api
        def old_api(patched=False,profile=None):
            old=original(patched,profile);old.loader_configuration=loader_configuration;old.run_owned_with_resources=run_owned_with_resources
            return old
        value.old_api=old_api
        _API=value
    return _API


def source_guard(root):return base().source_guard(root)
def select_profile(rows_by_case):return base().select_profile(rows_by_case)
def verify_closed_trial(report_path,parent_proof_path):
    value=base().verify_closed_trial(report_path,parent_proof_path)
    for row in value['cases'].values():check_loader_configuration(row['native_loader_configuration'],row['profile'])
    return value
def fingerprint(value):return base().fingerprint(value)
def write(path,value):return base().write(path,value)


def run_owned_with_resources(old,legacy,*args,**kwargs):
    global _RESOURCE_BASE
    _RESOURCE_BASE=legacy.resources;legacy.resources=resource_snapshot
    try:return old.run_owned(*args,**kwargs)
    finally:legacy.resources=_RESOURCE_BASE


if __name__=='__main__':raise SystemExit(base().main())
