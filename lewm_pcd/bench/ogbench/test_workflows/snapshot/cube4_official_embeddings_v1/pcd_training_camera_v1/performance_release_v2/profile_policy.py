"""Explicit positive-speed release; preserve old 5% failure and all numeric guards."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
CASE=HERE.parent
SHADOW=CASE/'workspace'
V4=SHADOW/'official_state_3d_cache_v1/acceleration_v1/v4'
PLAN=SHADOW/'official_state_3d_cache_v1/producer_plan.json'
REPORT=CASE/'3d/GPU_preprocess_report.json'
DECISION_SHA='f2d622395ff1c6c197d4d91b3ff872b5fb62a41998add932466c73d6b0235301'
WRAPPER_SHA='7d51103b4c078ac158aebfd97acc37970503b933d03fd5493d7e578efbb2714d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data if isinstance(data,bytes) else (json.dumps(data,indent=2,allow_nan=False)+'\n').encode())


def check_report(decision,report):
    assert decision['positive_gain_release'] is True and decision['original_accepted'] is False
    assert decision['original_5percent_time_reduction_failed'] is True
    assert decision['numeric_tolerances_changed'] is False and decision['fallback_to_CPU'] is False
    assert report['actual_exit_code']==0 and report['accepted'] is False
    assert report['prepared_bitexact'] is True and report['latent_bitexact'] is True
    assert report['prepared_max_abs_diff']==report['latent_max_abs_diff']==0.
    assert report['model_frozen_exact'] is True and report['training_updates']==report['render_frames']==0
    for left,right in [('plan_sha256','plan_sha256'),('checkpoint_sha256','checkpoint_sha256'),
                       ('candidate_source_sha256','GPU_preprocess_source_sha256')]:
        assert report[left]==decision[right],left
    baseline,candidate=report['baseline_median_seconds'],report['candidate_median_seconds']
    assert baseline==decision['baseline_median_seconds'] and candidate==decision['candidate_median_seconds']
    assert 0 < candidate < baseline and candidate > baseline*.95
    assert report['timing_scope']==decision['timing_scope']
    assert report['batch_size']==32 and report['XYZ_input_dtype']=='float32' and report['RGB_input_dtype']=='uint8'


def policy_source(original):
    edits=[
        ("assert accepted['accepted'] is True and accepted['candidate_source_sha256'] == sha(Path(__file__).with_name('gpu_preprocess.py'))",
         "assert _PERFORMANCE_RELEASE['positive_gain_release'] is True and accepted['accepted'] is False and accepted['candidate_source_sha256'] == sha(Path(__file__).with_name('gpu_preprocess.py'))"),
        ("assert accepted['candidate_median_seconds'] <= accepted['baseline_median_seconds'] * .95",
         "assert 0 < accepted['candidate_median_seconds'] < accepted['baseline_median_seconds'] and _PERFORMANCE_RELEASE['original_5percent_time_reduction_failed'] is True")]
    updated=original
    for before,after in edits:
        assert updated.count(before)==1
        updated=updated.replace(before,after)
    restored=updated
    for before,after in reversed(edits):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored==original
    ast.parse(updated)
    return updated,edits


def validate():
    decision_path=HERE/'release_decision.json'
    assert sha(decision_path)==DECISION_SHA
    decision=json.loads(decision_path.read_text())
    assert sha(REPORT)==decision['report_sha256'] and sha(PLAN)==decision['plan_sha256']
    assert sha(V4/'run_profile.py')==WRAPPER_SHA==decision['original_v4_wrapper_sha256']
    assert sha(V4/'gpu_preprocess.py')==decision['GPU_preprocess_source_sha256']
    report=json.loads(REPORT.read_text())
    check_report(decision,report)
    plan=json.loads(PLAN.read_text())
    assert plan['checkpoint_sha256']==decision['checkpoint_sha256']
    assert sha(plan['checkpoint'])==decision['checkpoint_sha256']
    assert plan['split_episodes']==dict(train=5000,val=500) and plan['raw_rows_per_episode']==1001
    assert decision['actual_exit_code']==0 and decision['cube_count']==4 and decision['data_id']=='cube-quadruple-play-v0'
    assert decision['original_engineering_5percent_gate_passed'] is False and decision['positive_gain_released'] is True
    assert decision['unchanged_numeric_tolerances'] is True and decision['gate_report_sha256']==decision['report_sha256']
    assert report['plan_path']==str(PLAN) and report['source_samples_count']==32
    assert report['real_raw_ids']==list(range(29))+[999,1000,1001]
    assert report['benchmark_source_sha256']==decision['benchmark_source_sha256']==sha(V4/'bench_gpu_preprocess.py')
    assert Path(report['samples_path'])==Path(plan['gate_dir'])/'real_PCD32.npz'
    assert sha(report['samples_path'])==report['samples_sha256']==decision['samples_sha256']
    assert sha(SHADOW/'official_state_3d_cache_v1/produce.py')==plan['source_pins']['official_state_3d_cache_v1/produce.py']==decision['base_producer_sha256']
    assert decision['positive_gain_fraction']==(report['baseline_median_seconds']-report['candidate_median_seconds'])/report['baseline_median_seconds']
    effective,edits=policy_source((V4/'run_profile.py').read_text())
    return decision,effective,edits


def profile_from_plan(plan,binding):
    # Same seven resource keys as the frozen Quad materializer; extra metadata binds this exception.
    return dict(base_producer=str(SHADOW/'official_state_3d_cache_v1/produce.py'),
        base_producer_sha256=plan['source_pins']['official_state_3d_cache_v1/produce.py'],
        renderer_workers=16,prefetch_jobs=32,affinity_cores=16,GPU_preprocess=True,
        GPU_preprocess_gate=dict(path=str(REPORT),sha256=binding['original_report_sha256']),
        status='Root_explicit_positive_speed_release_original5percent_gate_failed',performance_release=binding)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare-profile','run'),required=True)
    args,forwarded=parser.parse_known_args()
    decision,effective,edits=validate()
    profile_path=HERE/'chosen_profile.json'
    binding=dict(path=str(HERE/'release_decision.json'),sha256=DECISION_SHA,
        original_report_sha256=decision['report_sha256'],original_accepted=False,
        policy_wrapper_sha256=sha(__file__),original_5percent_failed=True)
    if args.mode=='prepare-profile':
        assert not forwarded
        profile=profile_from_plan(json.loads(PLAN.read_text()),binding)
        write_once(profile_path,profile)
        write_once(HERE/'profile_preparation_receipt.json',dict(actual_exit_code=0,
            profile_sha256=sha(profile_path),decision_sha256=DECISION_SHA,
            old_report_accepted=False,old_performance_criterion_passed=False,
            GPU_NN_or_render_executed=False,checkpoint_bytes_SHA_verified=True,real_samples_SHA_verified=True))
        print(json.dumps(dict(actual_exit_code=0,mode=args.mode,profile=str(profile_path))))
        return
    profile_file=Path(forwarded[forwarded.index('--profile')+1])
    runtime=Path(forwarded[forwarded.index('--runtime-dir')+1])
    plan_file=Path(forwarded[forwarded.index('--plan')+1])
    assert profile_file.resolve()==profile_path.resolve() and plan_file.resolve()==PLAN.resolve()
    profile=json.loads(profile_file.read_text())
    assert profile['performance_release']==binding and profile['GPU_preprocess'] is True
    assert profile['renderer_workers']==16 and profile['prefetch_jobs']==32 and profile['affinity_cores']==16
    write_once(runtime/'effective_policy_source.py',effective.encode())
    write_once(runtime/'policy_actual_launch.json',dict(started=time.time(),actual_argv=sys.argv,
        decision_path=str(HERE/'release_decision.json'),decision_sha256=DECISION_SHA,
        old_report_path=str(REPORT),old_report_sha256=decision['report_sha256'],
        old_report_accepted=False,original_5percent_failed=True,positive_speed_release=True,
        original_wrapper_sha256=WRAPPER_SHA,policy_wrapper_sha256=sha(__file__),
        effective_policy_source_sha256=sha(runtime/'effective_policy_source.py'),exact_guard_edits=edits,
        profile_sha256=sha(profile_file),plan_sha256=sha(plan_file),all_numeric_guards_preserved=True))
    sys.path.insert(0,str(V4))
    sys.argv=[str(V4/'run_profile.py')]+forwarded
    namespace=dict(__name__='_explicit_performance_release',__file__=str(V4/'run_profile.py'),
                   _PERFORMANCE_RELEASE=decision)
    exec(compile(effective,str(V4/'run_profile.py'),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':
    main()
