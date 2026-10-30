"""Select only genuinely closed Quad SR evidence; accepted generic pack does the IO."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from reuse import ROOT,Q,load,ASSET,ARCHIVE

HERE=Path(__file__).resolve().parent
OUT=Q/'closed_SR_backup_v1/completed_v1'
LAUNCH=Q/'closed_SR_backup_v1/execute_v1'
PLAN_SHA='acdb58486db6bd8499f7f34823b950232c52e2d82c553c8defd6996847a595a2'
RELEASE_SHA='cbaff3f63ec67ad3caebc0c6bc428c775146c3bcdfd05fafc860f109da73a601'
SOURCE_SHA='9039664bdc756aeab85f35b8df5571258493793bb85fec99e30bac2a902b2d6f'
NAMES=('01_G11_smoke','02_G11_smoke_audit','03_G11_full','04_G11_full_audit',
       '05_G12_smoke','06_G12_smoke_audit','07_G12_full','08_G12_full_audit')


def verify_closed(runtime,closure,exits,codes,reports):
    assert runtime['stage']=='Native_SR_pipeline_complete' and runtime['actual_exit_code']==0
    assert closure['actual_exit_code']==0 and closure['completed_phases']==8
    assert (closure['cube_count'],closure['data_id'],closure['training_updates'],closure['SR_episodes'])==(4,'cube-quadruple-play-v0',0,500)
    assert len(exits)==8 and all(x['actual_exit_code']==0 and x['residual_owned_members']==[] for x in exits)
    assert codes==['0','0']
    for group in ('G11','G12'):
        producer,audit=reports[group]
        assert producer['actual_exit_code']==audit['actual_exit_code']==0
        assert producer['group_id']==audit['group_id']==group and producer['mode']==audit['mode']=='full'
        assert producer['episodes']==audit['episodes']==producer['unique_seeds']==audit['unique_seeds']==250
        assert producer['successes']==audit['successes']==sum(audit['successes_by_task'].values())
        assert producer['success_rate']==audit['success_rate']==producer['successes']/250


def collect(io):
    assert io.sha(Q/'launch_plan.json')==PLAN_SHA and io.sha(Q/'ROOT_RELEASE_v1.json')==RELEASE_SHA
    assert io.sha(Q/'SOURCE_FREEZE_v1.json')==SOURCE_SHA
    plan=io.read(Q/'launch_plan.json');binding=io.read(Q/'bindings.json');runtime=Q/'runtime_v1'
    assert [p['name'] for p in plan['phases']]==list(NAMES)
    exits=[io.read(runtime/(n+'_exit.json')) for n in NAMES]
    reports={g:(io.read(Q/f'full/{g}/report.json'),io.read(Q/f'full/{g}/independent_audit.json')) for g in ('G11','G12')}
    verify_closed(io.read(runtime/'status.json'),io.read(runtime/'closure.json'),exits,
        [p.read_text().strip() for p in (runtime/'outer_exit_code',Q/'wrapper_exit_code')],reports)
    paths={Q/'ROOT_RELEASE_v1.json',Q/'launch_plan.json',Q/'bindings.json',Q/'SOURCE_FREEZE_v1.json',
        Q/'HANDOFF_v1.json',Q/'STATIC_REVIEW_v1.json',Q/'wrapper_exit_code',
        runtime/'status.json',runtime/'closure.json',runtime/'outer_exit_code',runtime/'actual_dispatch.json',runtime/'resources.jsonl'}
    for name in ('reuse.py','build_backup.py','copy_backup.py','SOURCE_FREEZE_v1.json','HANDOFF_v1.json'):
        paths.add(HERE/name)
    for name in NAMES:paths.update(runtime/(name+ending) for ending in ('_argv.json','_exit.json','.log'))
    summaries={}
    for group in ('G11','G12'):
        for mode in ('tiny','full'):
            output=Q/mode/group;count=1 if mode=='tiny' else 250;maximum=4 if mode=='tiny' else 1000
            producer=io.read(output/'report.json');audit=io.read(output/'independent_audit.json')
            assert producer['actual_exit_code']==audit['actual_exit_code']==0 and producer['episodes']==audit['episodes']==count
            assert (output/'actual_exit_code').read_text().strip()=='0'
            threads=io.read(output/'runtime_threads.json')
            assert threads['torch_num_threads']==1 and threads['numpy_physics']['BLAS_threads']==4
            paths.update(output/n for n in ('status.json','report.json','actual_exit_code','runtime_threads.json','independent_audit.json'))
            success=0;seeds=set()
            for task in range(1,2 if mode=='tiny' else 6):
                for index in range(1 if mode=='tiny' else 50):
                    episode=output/'episodes'/f'task{task}'/f'episode_{index:03d}'
                    result=io.read(episode/'result.json');seed=970000000+task*100000+index
                    assert (episode/'attempt_exit_code').read_text().strip()=='0' and result['actual_exit_code']==0
                    assert (result['group_id'],result['task_id'],result['episode_index'],result['seed'])==(group,task,index,seed)
                    assert 1<=result['steps']<=maximum and result['online_encoder_calls']==2*result['steps'] and result['goal_online_encoder_calls']==result['steps']
                    seeds.add(seed);success+=int(result['success'])
                    paths.update(episode/n for n in ('result.json','steps.npz','attempt_exit_code'))
            assert len(seeds)==count and success==producer['successes']==audit['successes']
            if mode=='full':summaries[group]=dict(episodes=count,successes=success,success_rate=success/count,
                successes_by_task=audit['successes_by_task'],producer_report_sha256=io.sha(output/'report.json'),audit_sha256=io.sha(output/'independent_audit.json'))
    # Same accepted exact source pins, excluding saved gate arrays and all model/raw payloads.
    for path,digest in plan['source_pins'].items():
        if Path(path).suffix in io.FORBIDDEN+('.npz',):continue
        assert io.sha(path)==digest;paths.add(Path(path))
    external={}
    for group,cfg in binding['groups'].items():
        training,accepted=Path(cfg['training']),Path(cfg['acceptance']);accept=io.read(accepted)
        assert io.sha(training)==cfg['training_sha256'] and io.sha(accepted)==cfg['acceptance_sha256']
        assert accept['actual_exit_code']==0 and accept['full153_state_exact'] and len(accept['checkpoint_transactions'])==11
        final=Path(cfg['final']);manifest=final/'manifest.json';identity=io.read(manifest)
        assert identity['counter']==1000000 and io.sha(manifest)==cfg['manifest_sha256']==accept['final_manifest_sha256']
        policy,wm=final/'params_1000000.pkl',Path(cfg['wm'])
        external[group]=dict(config_sha256=cfg['training_sha256'],acceptance_sha256=cfg['acceptance_sha256'],
            final_manifest_sha256=cfg['manifest_sha256'],policy=dict(path=str(policy),sha256=identity['agent_sha256'],stat=io.stat(policy),content_not_read=True),
            sidecar=dict(path=str(final/'run_state.pkl'),sha256=identity['sidecar_sha256'],content_not_read=True),
            world_model=dict(path=str(wm),sha256=cfg['wm_sha256'],stat=io.stat(wm),content_not_read=True),gate_array_reference=dict(path=cfg['gate'],sha256=cfg['gate_sha256'],content_not_read=True))
        paths.update((training,accepted,manifest,wm.parent/'config.json',wm.parent/'config.yaml'))
    return paths,dict(actual_exit_code=0,asset_kind=ASSET,groups=summaries,formal_episodes=500,diagnostic_episodes_excluded_from_SR=2,
        eight_actual_phase_exits=exits,outer_wrapper_actual0=True,checkpoint_and_WM_external_references=external,
        source_release_sha256=RELEASE_SHA,launch_plan_sha256=PLAN_SHA,source_freeze_sha256=SOURCE_SHA,
        models_cache_dataset_or_old_SR_contents_read=False,no_NN_or_SR_or_audit_started=True,
        low_or_zero_SR_results_preserved_without_filtering=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--build-child',action='store_true');args=p.parse_args()
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True,asset_kind=ASSET,output=str(OUT),archive=ARCHIVE,no_pack_or_copy=True)));return 0
    if args.build_child:
        code=1
        try:
            io=load('pack');io.R=Q;io.OUT=OUT;io.collect=lambda:collect(io);io.main();code=0
        finally:
            with (LAUNCH/'builder_actual_exit_code').open('x') as f:f.write(str(code)+'\n')
        return code
    assert not OUT.exists(),'Existing archive/attempt: adopt receipt, do not duplicate'
    LAUNCH.mkdir(exist_ok=False)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--build-child']
    with (LAUNCH/'builder_actual_argv.json').open('x') as f:json.dump(dict(argv=argv,started=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
    with (LAUNCH/'builder_stdout.log').open('xb') as stdout,(LAUNCH/'builder_stderr.log').open('xb') as stderr:result=subprocess.run(argv,stdout=stdout,stderr=stderr)
    with (LAUNCH/'builder_outer_exit_code').open('x') as f:f.write(str(result.returncode)+'\n')
    if OUT.is_dir():
        for name in ('builder_actual_exit_code','builder_outer_exit_code','builder_actual_argv.json','builder_stdout.log','builder_stderr.log'):
            assert not (OUT/name).exists();shutil.copyfile(LAUNCH/name,OUT/name)
    if result.returncode==0:print((OUT/'receipt.json').read_text())
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
