"""Final-only closed-role selector; default is plan-only, never reloads a model."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from archive_io import read,sha,save,stat,pack,verified_copy,strict_scope

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1'
GROUPS=[f'{family}_cube{n}' for family in ('visreg_vitb768','vjepa21_vitl1024') for n in (1,2,3,4)]
COUNTERS=[1]+list(range(100000,1000001,100000))

def select(group,out):
    """Read closure metadata first. No large payload hashing before every gate passes."""
    runtime=CASE/'control/formal_w8/runtime';launchpath=CASE/'control/formal_w8/launch_plan.json'
    launch=read(launchpath);phases=[p for p in launch['phases'] if p['name'].startswith(group+'_')]
    assert len(phases)==8
    paths=set();expected={}
    def add(path):
        path=strict_scope(path,ROOT);assert path.is_file(),str(path)
        paths.add(path)
    for phase in phases:
        prefix=runtime/phase['name'];record=read(str(prefix)+'_exit.json')
        assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
        assert read(str(prefix)+'_argv.json')['argv']==phase['argv']
        for suffix in ('_exit.json','_argv.json','.log'):add(str(prefix)+suffix)
    train=next(p for p in phases if p['name'].endswith('_05_train'))['argv']
    configpath=Path(train[train.index('--config')+1]);cfg=read(configpath);run=Path(cfg['output'])
    assert cfg['purpose']=='formal' and cfg['group_id']==cfg['role']==group
    assert run==CASE/'runs'/group
    assert cfg['total_updates']==1000000 and cfg['seed']==0 and cfg['batch_size']==1024
    acceptedpath=CASE/'acceptance'/f'{group}.json';accepted=read(acceptedpath)
    assert accepted['actual_exit_code']==0 and accepted['full153_state_exact'] is True
    assert accepted['state']['leaves']==153 and accepted['Adam_count']==1000000 and accepted['network_step']==1000001
    assert accepted['role']==group and accepted['CSV_validation_records']==200
    assert accepted['finite_actions'] is True and accepted['synthetic_zero_padding'] is False
    assert [r['counter'] for r in accepted['checkpoint_transactions']]==COUNTERS
    assert (accepted['actual_current_dim'],accepted['actual_goal_dim'])==(cfg['input_contract']['current_dim'],cfg['input_contract']['goal_dim'])
    assert (run/'trainer_exit_code').read_text().strip()=='0'
    status=read(run/'status.json');resolved=read(run/'resolved.json')
    assert status['exit_code']==0 and status['completed_updates']==1000000 and status['validation_count']==200
    assert status['adam_count']==1000000 and resolved['resumed_from'] is None and resolved['initial_counter']==0
    assert resolved['initial_network_step']==1 and resolved['initial_adam_count']==0
    with (run/'train.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==200 and [int(r['step']) for r in rows]==list(range(5000,1000001,5000))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    assert all(float(row['training/grad/norm'])>0 for row in rows)
    native=CASE/'native_sr'/group;audit=read(native/'independent_audit.json');report=read(native/'report.json')
    assert audit['actual_exit_code']==0 and audit['mode']=='full' and audit['group_id']==group
    assert audit['actual_step_arrays_checked'] is True and audit['diagnostic_excluded_from_formal_SR'] is False
    assert audit['config_sha256']==sha(configpath) and audit['source_binding_sha256']==sha(native/'native_source_binding.json')
    assert report['actual_exit_code']==0 and report['group_id']==group and report['mode']=='full'
    assert report['episodes']==report['unique_seeds']==250 and (native/'actual_exit_code').read_text().strip()=='0'
    binding=read(native/'native_source_binding.json')
    assert binding['mode']=='full' and binding['gate_smoke'] is False and binding['config_sha256']==sha(configpath)
    episodes=sorted((native/'episodes').glob('task*/episode_*'));assert len(episodes)==250
    successes=0
    for task in range(1,6):
        for index in range(50):
            episode=native/'episodes'/f'task{task}'/f'episode_{index:03d}';item=read(episode/'result.json')
            assert item['actual_exit_code']==0 and item['group_id']==group
            assert (item['task_id'],item['episode_index'],item['seed'])==(task,index,970000000+task*100000+index)
            assert 1<=item['steps']<=cfg['native_max_episode_steps']
            assert (episode/'attempt_exit_code').read_text().strip()=='0'
            assert item['success'] in (0,1);successes+=item['success']
            for name in ('result.json','steps.npz','attempt_exit_code'):add(episode/name)
    assert report['successes']==successes and report['success_rate']==successes/250
    for name in ('report.json','independent_audit.json','native_source_binding.json','actual_exit_code'):add(native/name)
    if (native/'runtime_threads.json').exists():add(native/'runtime_threads.json')
    for name in ('resolved.json','status.json','train.csv','trainer_exit_code','final_state_fingerprint.json','final_val_probe.npz'):add(run/name)
    add(configpath);add(acceptedpath)
    ledger=[]
    for counter,transaction in zip(COUNTERS,accepted['checkpoint_transactions']):
        checkpoint=run/'checkpoints'/f'update_{counter:09d}';manifestpath=checkpoint/'manifest.json';manifest=read(manifestpath)
        assert manifest['counter']==counter and sha(manifestpath)==transaction['manifest_sha256'];add(manifestpath)
        policy=checkpoint/f'params_{counter}.pkl';sidecar=checkpoint/'run_state.pkl'
        ledger.append(dict(counter=counter,manifest_sha256=sha(manifestpath),policy_path=str(policy),
            policy_stat=stat(policy),policy_sha256=manifest['agent_sha256'],sidecar_sha256=manifest['sidecar_sha256'],
            payload_included=counter==1000000,periodic_payload_SHA_adopted_from_completed_CPU_accept=counter!=1000000))
        if counter==1000000:
            assert accepted['final_checkpoint']==str(checkpoint) and accepted['final_manifest_sha256']==sha(manifestpath)
            add(policy);add(sidecar);expected[str(policy)]=manifest['agent_sha256'];expected[str(sidecar)]=manifest['sidecar_sha256']
    deliverypath=Path(cfg['delivery_binding']['path']);delivery=read(deliverypath)
    assert sha(deliverypath)==cfg['delivery_binding']['sha256'];add(deliverypath)
    cachegroup=deliverypath.parent;cacheplan=read(cachegroup/'plan.json')
    for name in ('plan.json','data_binding.json','camera.json'):add(cachegroup/name)
    for name in ('cache_manifest.json','cache_seal.json','protocol.json','batch1_alignment.json'):add(Path(cacheplan['cache_dir'])/name)
    cacheaudit=Path(delivery['cache_audit']['path']);assert sha(cacheaudit)==delivery['cache_audit']['sha256']
    add(cacheaudit);add(cacheaudit.parent/'actual_exit_code');add(Path(cacheplan['shared_state_root'])/'source_arrays.json')
    encoder=cfg['encoder_binding'];specpath=Path(encoder['spec_path']);assert sha(specpath)==encoder['spec_sha256'];add(specpath)
    add(encoder['adapter']['path'])
    for relative,digest in cacheplan['source_pins'].items():
        path=ROOT/relative;assert sha(path)==digest;add(path)
    for path,digest in resolved['binding']['code'].items():assert sha(path)==digest;add(path)
    for path in (CASE/'functional').glob('*.py'):add(path)
    for path in (Path(__file__),Path(__file__).with_name('archive_io.py')):add(path)
    # Native's completed auditor bound these exact small sources and the final
    # downstream checkpoint; encoder checkpoint remains a SHA reference only.
    for path,digest in binding['source_pins'].items():
        path=Path(path)
        if path.suffix in ('.py','.json') and path.stat().st_size<16*(1<<20):
            assert sha(path)==digest;add(path)
    for path in paths:
        if path.suffix=='.pkl':assert path.parent==run/'checkpoints/update_001000000'
        if path.suffix=='.npz':assert path==run/'final_val_probe.npz' or path.parent.parent.parent==native/'episodes'
        assert path.suffix not in ('.pt','.pth','.ckpt','.npy','.h5','.hdf5')
        assert not (path.parent==runtime and path.name in ('status.json','resources.jsonl'))
    save(out/'checkpoint_SHA_ledger.json',dict(actual_exit_code=0,checkpoints=ledger,periodic_payloads_not_read_or_copied=True))
    save(out/'group_phase_plan.json',dict(group_id=group,parent_plan_path=str(launchpath),parent_plan_sha256=sha(launchpath),phases=phases))
    paths.update((out/'checkpoint_SHA_ledger.json',out/'group_phase_plan.json'))
    return paths,expected,dict(group_id=group,all_eight_phase_exits=0,CPU_acceptance_actual_exit_code=0,
        final_Native_episodes=250,independent_audit_actual_exit_code=0,other_group_or_active_aggregate_not_included=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--pack-child',action='store_true');args=p.parse_args()
    out=CASE/'closed_final_backup_v1'/args.group
    if not args.execute and not args.pack_child:
        print(__import__('json').dumps(dict(prepared_only=True,no_live_source_reads=True,group=args.group,output=str(out),
            closure_required='eight actual0/drained phases + full153/Adam1M/11manifests/200CSV + Native250/audit0',
            payload='only final1M params+sidecar, eleven manifests, Native steps/results and closed metadata')));return
    if args.pack_child:
        code=1
        try:
            os.nice(19)
            result=subprocess.run(['ionice','-c','3','-p',str(os.getpid())],capture_output=True,text=True);assert result.returncode==0
            paths,expected,metadata=select(args.group,out)
            print(__import__('json').dumps(pack(ROOT,out,paths,metadata,expected)));code=0
        except BaseException:
            save(out/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()));raise
        finally:
            with (out/'inner_exit_code').open('x') as stream:stream.write(str(code)+'\n')
        return
    assert args.execute and not args.pack_child
    out.mkdir(parents=True,exist_ok=False)
    with (out/'builder.stdout.log').open('x') as stdout,(out/'builder.stderr.log').open('x') as stderr:
        child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--group',args.group,'--pack-child'],stdout=stdout,stderr=stderr)
    with (out/'outer_exit_code').open('x') as stream:stream.write(str(child.returncode)+'\n')
    print(__import__('json').dumps(dict(actual_exit_code=child.returncode,output=str(out))))
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
