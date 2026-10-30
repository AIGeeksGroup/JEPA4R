"""Run the unchanged feature audit once, then audit mixed execution metadata only."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if not __debug__:
    raise RuntimeError('Audit requires assertions enabled')

HERE = Path(__file__).resolve().parent
H = HERE.parent
FEATURE_AUDIT_SHA = '8b1d14046cc8ea0e97d9fd3304902a00c42bc549861dfb5b7b3db9d8491d68f7'
WRAPPER_SHA = '402a53952cc1d8f304d97449ff9e2cf008c29eb6ffbcb5f07f7be06437f48f08'
ADAPTER_SHA = '7929544aab09cb17cd25b269ef0830259456c9fa76fcd62cf07e5d19819e35bf'
BASE_SHA = '7e4c201bb259a005e6b3cc49e4a2df3b3963e347c4a6aa26b17ea22666dce4db'
LEGACY_COUNT = 199


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(path):
    value=Path(path).stat()
    return value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_ino


def ast_sha(source):
    return hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest()


def audit_execution(cache_dir,plan_path,runtime_dir,stop_dir,feature_report):
    """Do not open vectors; bind every legacy/new receipt to actual runtime evidence."""
    cache_dir,runtime_dir,stop_dir=map(lambda p:Path(p).resolve(),(cache_dir,runtime_dir,stop_dir))
    tracked={}

    def read(path,expected=None,label='metadata SHA'):
        path=Path(path).resolve()
        before=stat(path)
        assert tracked.setdefault(path,before)==before
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert expected is None or digest==expected,label
        assert stat(path)==before
        return json.loads(raw),digest

    assert feature_report['actual_exit_code']==0
    plan,plan_sha=read(plan_path)
    assert plan_sha==feature_report['plan_sha256']
    manifest,manifest_sha=read(cache_dir/'cache_manifest.json',feature_report['manifest_sha256'])
    assert manifest['protocol']['renderer_workers']==2, 'Historical base resource field changed'
    companion,companion_sha=read(cache_dir/'runtime_profiles_manifest.json')
    assert companion['actual_exit_code']==0 and companion['mixed_resource_profiles'] is True
    assert companion['base_manifest_sha256']==manifest_sha
    launch,launch_sha=read(runtime_dir/'actual_launch.json')
    identity=launch['identity']
    profile,profile_sha=read(identity['profile_path'],identity['profile_sha256'],'profile SHA')
    wrapper_path=Path(identity['wrapper_path'])
    assert sha(wrapper_path)==identity['wrapper_sha256']==WRAPPER_SHA, 'Runtime wrapper SHA'
    tracked[wrapper_path.resolve()]=stat(wrapper_path)
    assert sha(profile['base_producer'])==profile['base_producer_sha256']==identity['base_producer_sha256']==BASE_SHA
    base=Path(profile['base_producer']);tracked[base.resolve()]=stat(base)
    assert profile['renderer_workers']==identity['renderer_workers'] and identity['renderer_workers'] in (8,16)
    assert profile['prefetch_jobs']==identity['prefetch_jobs']==2*identity['renderer_workers']
    assert profile['affinity_cores']==identity['affinity_cores']==16
    assert profile['outer_owned_RSS_cap_bytes']==24*1024**3
    assert identity['next_job_submitted_before_infer'] is identity['FIFO_commit_order'] is True
    assert identity['XYZ_dtype']=='float32' and identity['RGB_dtype']=='uint8'
    assert identity['base_protocol_renderer_workers_is_historical'] is True
    assert launch['reverse_AST_equals_base'] is launch['inference_AST_equals_base'] is True
    spec=importlib.util.spec_from_file_location('_frozen_resource_overlay_metadata',wrapper_path)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    expected=wrapper.overlay(base.read_text())
    effective_path=runtime_dir/'effective_source.py'
    before=stat(effective_path);effective=effective_path.read_text();tracked[effective_path]=before
    assert effective==expected and stat(effective_path)==before, 'Actual execution snapshot differs'
    assert ast_sha(effective)==identity['effective_AST_sha256']
    assert ast_sha(base.read_text())==launch['base_AST_sha256']
    argv=launch['actual_argv']
    assert Path(argv[0]).resolve()==wrapper_path.resolve()
    for flag,path in (('--profile',identity['profile_path']),('--runtime-dir',runtime_dir),('--plan',plan_path)):
        assert argv.count(flag)==1 and Path(argv[argv.index(flag)+1]).resolve()==Path(path).resolve()
    assert '--execute' in argv and '--pilot' not in argv
    gate_sha=None
    if profile['GPU_preprocess']:
        assert identity['preprocess_device']=='cuda'
        assert identity['GPU_preprocess_source_sha256']==ADAPTER_SHA
        adapter=wrapper_path.with_name('gpu_preprocess.py')
        assert sha(adapter)==ADAPTER_SHA;tracked[adapter.resolve()]=stat(adapter)
        assert identity['GPU_preprocess_gate']==profile['GPU_preprocess_gate']
        gate,gate_sha=read(profile['GPU_preprocess_gate']['path'],profile['GPU_preprocess_gate']['sha256'])
        assert gate['actual_exit_code']==0 and gate['accepted'] is True
        assert gate['prepared_bitexact'] is gate['latent_bitexact'] is gate['model_frozen_exact'] is True
        assert gate['prepared_max_abs_diff']==gate['latent_max_abs_diff']==0.
        assert gate['candidate_median_seconds']<=gate['baseline_median_seconds']*.95
        assert gate['plan_sha256']==plan_sha and gate['checkpoint_sha256']==plan['checkpoint_sha256']
        assert gate['candidate_source_sha256']==ADAPTER_SHA
        assert gate['batch_size']==32 and gate['paired_alternating_trials']==20 and gate['warmup_pairs']==3
        assert gate['XYZ_input_dtype']=='float32' and gate['RGB_input_dtype']=='uint8'
        assert gate['timing_scope']=='H2D+original_preprocessor+encode_only_no_render_no_cache_writes'
        assert gate['training_updates']==gate['render_frames']==0
    else:
        assert identity['preprocess_device']=='cpu'
        assert identity['GPU_preprocess_source_sha256'] is None and identity['GPU_preprocess_gate'] is None
    stopped,stop_before_sha=read(stop_dir/'before_stop.json')
    stop_receipt,stop_receipt_sha=read(stop_dir/'receipt.json')
    assert stop_receipt['actual_stop_helper_exit_code']==0 and stop_receipt['preserved_episode_receipts']==LEGACY_COUNT
    assert stop_receipt['every_existing_receipt_SHA_unchanged'] and stop_receipt['old_exits_not_rewritten']
    assert stop_receipt['uncommitted_paths']==stop_receipt['owned_residual_members']==[]
    preserved={str(Path(item['path']).resolve()):item['sha256'] for item in stopped['preserved_receipts']}
    assert len(preserved)==len(stopped['preserved_receipts'])==LEGACY_COUNT
    old_argv=stopped['producer_actual_argv']['argv']
    assert str(base) in old_argv and '--execute' in old_argv and '--profile' not in old_argv
    expected_entries={(split,item['episode']):item['receipt_sha256']
        for split,entries in feature_report['receipt_bindings'].items() for item in entries}
    rows=companion['episodes']
    assert len(rows)==feature_report['episodes_verified']==len(expected_entries)
    legacy,new,seen,legacy_paths=0,0,set(),set()
    for row in rows:
        key=row['split'],row['episode']
        assert key in expected_entries and key not in seen
        seen.add(key)
        path=cache_dir/row['split']/f"episode_{row['episode']:06d}/receipt.json"
        assert Path(row['receipt_path']).resolve()==path.resolve()
        record,digest=read(path,expected_entries[key])
        assert digest==row['receipt_sha256']
        actual=record.get('runtime_resource_profile')
        assert row['runtime_resource_profile']==actual
        if actual is None:
            assert row['profile_kind']=='legacy_base_workers2_prefetch2'
            assert preserved.get(str(path))==digest, 'Unknown EP without runtime profile'
            legacy_paths.add(str(path));legacy+=1
        else:
            assert row['profile_kind']=='resource_overlay' and actual==identity
            assert str(path) not in preserved, 'Old receipt was rewritten with new metadata'
            new+=1
    assert legacy==LEGACY_COUNT and legacy_paths==set(preserved) and new>0
    assert seen==set(expected_entries)
    for path,before in tracked.items():assert stat(path)==before,'Runtime evidence changed during audit'
    return dict(stage='mixed_3d_execution_metadata_audit_pass',actual_exit_code=0,
        companion_sha256=companion_sha,base_manifest_sha256=manifest_sha,
        original_feature_auditor_sha256=FEATURE_AUDIT_SHA,
        base_protocol_renderer_workers_historical=2,actual_renderer_workers=identity['renderer_workers'],
        actual_prefetch_jobs=identity['prefetch_jobs'],actual_affinity_cores=16,
        legacy_base_workers2_episodes=legacy,resource_profile_episodes=new,
        all_episode_receipt_SHA_and_execution_profile_bindings=True,
        profile_sha256=profile_sha,wrapper_sha256=WRAPPER_SHA,effective_AST_sha256=identity['effective_AST_sha256'],
        actual_launch_sha256=launch_sha,GPU_preprocess_gate_sha256=gate_sha,
        preprocess_device=identity['preprocess_device'],stop_before_sha256=stop_before_sha,
        stop_receipt_sha256=stop_receipt_sha,vectors_read_again=False,NN_executed=False,
        training_updates=0,SR_episodes=0,original_failed_exit_codes_rewritten=False,
        ready_for_downstream=False,outer_actual_exit_still_required=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--stop-dir',type=Path,default=HERE/'controlled_stop_v1')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    code,started=1,time.time()

    def save(name,value):
        with (args.output_dir/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

    try:
        source=H/'audit_cache.py'
        assert sha(source)==FEATURE_AUDIT_SHA,'Original feature auditor changed'
        argv=[sys.executable,'-B',str(source),'--cache-dir',str(args.cache_dir),
              '--plan',str(args.plan),'--output-dir',str(args.output_dir/'features')]
        save('feature_audit_actual_argv.json',dict(argv=argv,started=time.time()))
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (args.output_dir/'feature_audit.log').open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=environment)
            actual=child.wait()
        save('feature_audit_actual_exit.json',dict(actual_exit_code=actual,completed=time.time()))
        assert actual==0
        assert (args.output_dir/'features/actual_exit_code').read_text().strip()=='0'
        feature_report=json.loads((args.output_dir/'features/report.json').read_text())
        report=audit_execution(args.cache_dir,args.plan,args.runtime_dir,args.stop_dir,feature_report)
        report.update(auditor_sha256=sha(__file__),started=started,completed=time.time(),
            feature_report_path=str(args.output_dir/'features/report.json'),
            feature_report_sha256=sha(args.output_dir/'features/report.json'))
        save('runtime_report.json',report)
        code=0
        print(json.dumps(dict(actual_exit_code=0,feature_report=report['feature_report_path'],
            supplemental_report=str(args.output_dir/'runtime_report.json'))))
    except BaseException as error:
        save('failure.json',dict(actual_exit_code=1,error=repr(error),started=started,completed=time.time()))
        raise
    finally:
        (args.output_dir/'runtime_audit_actual_exit_code').write_text(str(code)+'\n')
        (args.output_dir/'actual_exit_code').write_text(str(code)+'\n')


if __name__=='__main__':
    main()
