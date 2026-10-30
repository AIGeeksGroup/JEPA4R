"""Epoch2 read-task comparison. Reuses the frozen native probe; default prepare only."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE_FREEZE='cdfd14b894fda7bf7f068c489bc550e819ade12ca16f36250ebf0dda2cd374d7'
MANIFEST='98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
PARENT_UPDATES,TOTAL_UPDATES,WARM,WINDOW,TIMED=91406,457030,32,128,384
NAMES=['A1_native20_pf2','B_split20_pf2','C_split32_pf2','D_split32_pf4','A2_native20_pf2']
PROFILES={name:dict(name=name,read_batch_size=32 if split else None,train_workers=w,train_prefetch=pf,
    validation_workers=20,validation_prefetch=2,split_validation=split)
    for name,w,pf,split in zip(NAMES,[20,20,32,32,20],[2,2,2,4,2],[False,True,True,True,False])}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def load(name,path,tree=None):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    if tree is None:spec.loader.exec_module(m)
    else:exec(compile(tree,str(path),'exec'),m.__dict__)
    return m


def select_profile(rows_by_case):
    aggregate={};median={}
    for name in NAMES:
        windows=rows_by_case[name]['timing_windows']
        if len(windows)!=3 or any(x['updates']!=128 or not math.isfinite(x['seconds']) or x['seconds']<=0 for x in windows):
            raise ValueError('three finite 128-update windows required')
        aggregate[name]=384/sum(x['seconds'] for x in windows)
        median[name]=statistics.median(128/x['seconds'] for x in windows)
    drift=lambda rates:max(rates[NAMES[0]],rates[NAMES[-1]])/min(rates[NAMES[0]],rates[NAMES[-1]])-1
    controls=dict(aggregate_drift=drift(aggregate),median_drift=drift(median));candidates={}
    for name in NAMES[1:-1]:
        ag=aggregate[name]/max(aggregate[NAMES[0]],aggregate[NAMES[-1]])-1
        mg=median[name]/max(median[NAMES[0]],median[NAMES[-1]])-1
        candidates[name]=dict(aggregate_gain=ag,median_gain=mg,
            eligible=controls['aggregate_drift']<=.05 and controls['median_drift']<=.05 and ag>=.05 and mg>=.05)
    eligible=[name for name in NAMES[1:-1] if candidates[name]['eligible']]
    chosen=max(eligible,key=lambda name:aggregate[name]) if eligible else NAMES[0]
    return dict(selected_profile=PROFILES[chosen],aggregate=aggregate,median=median,controls=controls,candidates=candidates)


def validate_parent(p):
    expected=dict(stage='pldm_epoch_boundary_verified',actual_exit_code=0,completed_epochs=2,
        global_step=91406,Adam_steps=[91406.0],native_schedule_index=91405,model_bare_exact=True,model_optimizer_finite=True)
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('epoch') not in (1,2):raise ValueError('complete epoch2 parent required')
    b=p['baseline_binding']
    if (b.get('method'),b.get('manifest_sha256'),b.get('source_freeze_sha256'))!=('pldm',MANIFEST,BASE_FREEZE):raise ValueError('parent method/data/source differs')
    if [x['epoch'] for x in p['epoch_exports']]!=[1,2]:raise ValueError('external1 plus protected2 ledger required')
    return p


def source_guard(root):
    f=read(HERE/'source_freeze_v1.json')
    if str(root)!=f['root']:raise ValueError('actual root differs')
    for name,h in f['files'].items():
        if sha(HERE/name)!=h:raise ValueError('probe source differs: '+name)
    for name,h in f['base_files'].items():
        if sha(root/name)!=h:raise ValueError('frozen reference differs: '+name)
    sys.path.insert(0,str(root));import shared_data
    data_root=Path(f['data_root']);shared_data.require_sources(root,data_root);return data_root


def compare_cohort(rows):
    keys=('ids','first2_inputs','RNG_before','RNG_after','start_global_step','end_global_step','native_schedule_index','parent_proof_sha256')
    for row in rows:
        if (row['start_global_step'],row['end_global_step'],row['native_schedule_index'])!=(91406,91822,91821):raise ValueError('diagnostic update budget differs')
        if len(row['ids'])!=416 or any(len(x)!=128 for x in row['ids']) or len(row['first2_inputs'])!=2:raise ValueError('actual416 B128 consumption required')
        if row.get('first2_pinned_outputs')!=[True,True]:raise ValueError('real GPU-visible trial must return pinned CPU inputs')
        if any(row[k]!=rows[0][k] for k in keys):raise ValueError('cohort/input/main RNG differs')


def verify_closed_trial(report_path,parent_proof_path):
    report_path,parent_proof_path=Path(report_path),Path(parent_proof_path)
    report=read(report_path);parent=validate_parent(read(parent_proof_path));output=report_path.parent
    if (report.get('stage'),report.get('exit_code'),report.get('selection_requires_both_metrics'),report.get('all_diagnostic_updates_discarded'))!=(
        'readtask_trial_complete_awaiting_Root_formal_release',0,True,True):raise ValueError('actual complete readtask trial0 required')
    if report.get('source_freeze_sha256')!=sha(HERE/'source_freeze_v1.json'):raise ValueError('actual readtask source freeze differs')
    for p in (output/'actual_exit_code',Path(str(output)+'.outer_exit_code')):
        if p.read_text().strip()!='0':raise ValueError('trial driver/outer not actual0')
    rows=[]
    for name in NAMES:
        directory=output/name;row=read(directory/'result/trial_result.json');drain=read(directory/'process_group_drain.json')
        if (directory/'actual_exit_code').read_text().strip()!='0' or drain['stage']!='empty' or drain['active']:raise ValueError('case exit/drain incomplete')
        if row!=report['cases'][name] or row.get('actual_exit_code')!=0 or row['profile']!=PROFILES[name]:raise ValueError('actual case/report/profile differs')
        binding=read(directory/'result/actual_protocol.json')['binding']
        if binding.get('worker_trial_source_freeze_sha256')!=report['source_freeze_sha256'] or binding.get('readtask_profile')!=PROFILES[name]:raise ValueError('actual case source/resource binding differs')
        if row['parent_proof_sha256']!=sha(parent_proof_path) or row['measured_updates']!=384:raise ValueError('case parent/update evidence differs')
        if row['updates_per_second']!=384/sum(w['seconds'] for w in row['timing_windows']):raise ValueError('case aggregate arithmetic differs')
        rows.append(row)
    compare_cohort(rows);computed=select_profile(report['cases'])
    if any(report.get(k)!=v for k,v in computed.items()):raise ValueError('actual two-metric selection differs')
    if report.get('parent_proof_sha256')!=sha(parent_proof_path) or report.get('formal_started') is not False:raise ValueError('trial parent/formal policy differs')
    return report


def require_runtime(a):
    data_root=source_guard(a.root);r=read(a.release);output=a.output.parent if a.mode=='case' else a.output
    expected=dict(stage='approved_readtask_trial',source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),output=str(output),parent_proof=str(a.parent_proof),parent_proof_sha256=sha(a.parent_proof))
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('matching separate Root trial release required')
    p=validate_parent(read(a.parent_proof))
    for item in p['files'].values():
        z=Path(item['path']).stat()
        if (z.st_size,z.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('protected epoch2 changed')
    stop=Path(r['retirement_receipt'])
    if sha(stop)!=r['retirement_receipt_sha256'] or read(stop).get('actual_exit_code')!=0:raise ValueError('actual Root retirement receipt0 required')
    current=read(HERE/'current_identity_v1.json');f=Path(current['formal_output'])
    for z in (f/'formal/pldm/actual_exit_code',f/'actual_exit_code',Path(str(f)+'.outer_exit_code')):
        if not z.exists():raise ValueError('original full process exits not recorded')
        int(z.read_text().strip()) # controlled retirement may truthfully retain nonzero exits
    drain=read(f/'formal/pldm/process_group_drain.json')
    if drain['stage']!='empty' or drain['active']:raise ValueError('original owned trainer group not drained')
    if Path('/proc',str(current['identity']['pid'])).exists():raise ValueError('current trainer still present')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip() or int(os.environ.get('SLURM_RESTART_COUNT','0')):raise ValueError('GPUempty and fresh isolated ownrun required')
    return data_root,p


def adapt_loaders(train,val,profile):
    if profile['read_batch_size'] is None:return train,val
    f=read(HERE/'source_freeze_v1.json');adapter=load('readtask_frozen_adapter',HERE.parent/f['adapter_entry'])
    train=adapter.wrap_loader(train,workers=profile['train_workers'],prefetch_factor=profile['train_prefetch'],read_batch_size=32,measure=True)
    if profile['split_validation']:val=adapter.wrap_loader(val,workers=20,prefetch_factor=2,read_batch_size=32,measure=True)
    return train,val


def probe_tree(source):
    tree=ast.parse(source);counts=[0]*9
    class Patch(ast.NodeTransformer):
        active=False
        def visit_FunctionDef(self,node):
            if not self.active and node.name not in ('run_case','run_aba'):return node
            previous=self.active;self.active=True;result=self.generic_visit(node);self.active=previous;return result
        def visit_Assign(self,node):
            text=ast.unparse(node)
            if text=='train.num_workers = a.workers':
                counts[0]+=1;return ast.copy_location(ast.parse("train, val = adapt_loaders(train, val, CURRENT_PROFILE)").body[0],node)
            if text=="loader_ref['train'] = train":
                counts[1]+=1;return [node,ast.copy_location(ast.parse("loader_ref['val'] = val").body[0],node)]
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='dict' and any(isinstance(t,ast.Name) and t.id=='state' for t in node.targets):
                if any(k.arg=='cases' for k in node.value.keywords):node.value.keywords += [ast.keyword('parent_proof_sha256',ast.parse('sha(a.parent_proof)',mode='eval').body),ast.keyword('source_freeze_sha256',ast.parse("sha(HERE/'source_freeze_v1.json')",mode='eval').body)]
                if any(k.arg=='workers' for k in node.value.keywords):node.value.keywords += [ast.keyword('profile',ast.Name('CURRENT_PROFILE',ast.Load()))]
            return self.generic_visit(node)
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text=='train.prefetch_factor != 2':counts[2]+=1;node.comparators=[ast.parse("CURRENT_PROFILE['train_prefetch']",mode='eval').body]
            elif text=='trainer.current_epoch != 1':counts[3]+=1;node.comparators=[ast.Constant(2)]
            return self.generic_visit(node)
        def visit_If(self,node):
            self.generic_visit(node)
            if ast.unparse(node.test)=='index < 2':
                counts[4]+=1;node.body+=ast.parse("row['pinned_output'] = batch['pixels'].is_pinned()").body
            elif ast.unparse(node.test)=='self.done < 2':
                counts[5]+=1;node.body+=ast.parse("state.setdefault('first2_pinned_outputs', []).append(row['pinned_output'])").body
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='recommend_workers':
                counts[6]+=1;return ast.copy_location(ast.parse("select_profile(state['cases'])",mode='eval').body,node)
            if ast.unparse(node.func)=='state.update' and any(k.arg=='measured_updates' for k in node.keywords):
                counts[7]+=1
                node.keywords += [ast.keyword('readtask_statistics',ast.parse("dict(getattr(loader_ref['train'], 'readtask_statistics', {}))",mode='eval').body),ast.keyword('validation_readtask_statistics',ast.parse("dict(getattr(loader_ref['val'], 'readtask_statistics', {}))",mode='eval').body)]
            if ast.unparse(node.func)=='state.update' and any(k.arg=='formal_started' for k in node.keywords):node.keywords += [ast.keyword('selection_requires_both_metrics',ast.Constant(True))]
            return node
        def visit_Dict(self,node):
            self.generic_visit(node)
            if any(isinstance(k,ast.Constant) and k.value=='worker_trial_source_freeze_sha256' for k in node.keys):
                counts[8]+=1
                for key,value in [('prefetch_factor',"CURRENT_PROFILE['train_prefetch']"),('validation_prefetch_factor','2'),('readtask_profile','CURRENT_PROFILE')]:
                    node.keys.append(ast.Constant(key));node.values.append(ast.parse(value,mode='eval').body)
            return node
        def visit_Constant(self,node):
            replacements={'worker_ABA_complete_awaiting_Root_formal_release':'readtask_trial_complete_awaiting_Root_formal_release','worker_ABA_failed_stopped':'readtask_trial_failed_stopped','Protected epoch1 + remaining9epochs; new seed0 loader, not exact original sample/RNG continuation':'Protected epoch2 plus external1 + remaining8epochs; seed0/new loader; not exact original sample/RNG continuation'}
            if isinstance(node.value,str) and node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    tree=Patch().visit(tree)
    if counts!=[1]*9:raise ValueError('old native probe AST boundaries changed: '+str(counts))
    return ast.fix_missing_locations(tree)


def old_api(patched=False,profile=None):
    path=HERE.parent/'pldm_worker_trial_v1/trial.py';module=load('readtask_old_probe_'+str(patched),path,probe_tree(path.read_text()) if patched else None)
    module.HERE=HERE;module.PARENT_UPDATES=PARENT_UPDATES;module.TOTAL_UPDATES=TOTAL_UPDATES;module.WARM=WARM;module.WINDOW=WINDOW;module.TIMED=TIMED
    module.CASES=[(n,PROFILES[n]['train_workers']) for n in NAMES]
    module.require_runtime=require_runtime;module.compare_cohort=compare_cohort;module.select_profile=select_profile;module.adapt_loaders=adapt_loaders
    module.CURRENT_PROFILE=profile
    module.resources=lambda workers:dict(profile or {},batch_size=128,pin_memory=True,persistent_workers=False)
    return module


def fingerprint(value):return old_api().fingerprint(value)


def boundary_tree(source):
    tree=ast.parse(source);counts=[0]*7
    class Patch(ast.NodeTransformer):
        def visit_FunctionDef(self,node):return self.generic_visit(node) if node.name in ('ready','main') else node
        def visit_Assign(self,node):
            text=ast.unparse(node)
            for name,key in [('F','formal_output'),('RUN','own_run'),('BARE','bare_epoch2')]:
                if any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                    counts[{'F':0,'RUN':1,'BARE':2}[name]]+=1
                    return ast.copy_location(ast.parse(name+" = Path(CURRENT_BINDING["+repr(key)+"])").body[0],node)
            if any(isinstance(t,ast.Name) and t.id=='proof' for t in node.targets):
                counts[3]+=1;node.value=ast.Call(ast.Name('_extend_parent',ast.Load()),[node.value],[])
            return self.generic_visit(node)
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text=="summary.get('epoch', -1) >= 1":counts[4]+=1;node.comparators=[ast.Constant(2)]
            elif text=="full['epoch'] not in (0, 1)":counts[5]+=1;node.comparators=[ast.Tuple([ast.Constant(1),ast.Constant(2)],ast.Load())]
            return self.generic_visit(node)
        def visit_Constant(self,node):
            replacements={'approved_pldm_epoch1_preservation':'approved_pldm_epoch2_preservation','preserved_epoch1':'preserved_epoch2','last_epoch1.ckpt':'last_epoch2.ckpt','weights_epoch_1.pt':'weights_epoch_2.pt','pldm_epoch1_verified':'pldm_epoch_boundary_verified','waiting_original_full_epoch1_save':'waiting_original_complete_epoch2_save','epoch1_preservation_failed':'epoch2_preservation_failed'}
            if isinstance(node.value,str) and node.value==r'\[Epoch\s+1/10\]\s+step\s+':counts[6]+=1;return ast.copy_location(ast.Constant(r'\[Epoch\s+2/10\]\s+step\s+'),node)
            if isinstance(node.value,str) and node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    tree=Patch().visit(tree)
    if counts!=[1]*7:raise ValueError('old boundary AST boundaries changed: '+str(counts))
    return ast.fix_missing_locations(tree)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','aba','case'),default='prepare')
    for key in ('root','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('parent-proof','release'):p.add_argument('--'+key,type=Path)
    p.add_argument('--workers',type=int,choices=(20,32),default=20);a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
    if a.mode=='prepare':source_guard(a.root);print(json.dumps(dict(stage='prepared_only_no_workload',profiles=PROFILES,parent_updates=PARENT_UPDATES,diagnostic_updates=416)));return 0
    if not a.parent_proof or not a.release:raise ValueError('explicit parent/Root release required')
    a.parent_proof=a.parent_proof.resolve();a.release=a.release.resolve()
    profile=PROFILES.get(a.output.name) if a.mode=='case' else None
    if a.mode=='case' and (profile is None or a.workers!=profile['train_workers']):raise ValueError('case/worker profile differs')
    old=old_api(True,profile);return old.run_case(a) if a.mode=='case' else old.run_aba(a)


if __name__=='__main__':raise SystemExit(main())
