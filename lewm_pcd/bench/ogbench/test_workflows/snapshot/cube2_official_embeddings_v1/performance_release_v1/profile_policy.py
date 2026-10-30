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
DECISION_SHA='18d4e89134dfd70211ebb9198f825e8a5daf73dd9516904a91391ca52d73cc68'
WRAPPER_SHA='7cbb13be10f73a92f4958de5a5a6af209be6a36e4df50e3e90c1b6211ecc90d1'


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
    assert sha(V4/'run_profile.py')==WRAPPER_SHA==decision['original_cube2_v4_wrapper_sha256']
    assert sha(V4/'gpu_preprocess.py')==decision['GPU_preprocess_source_sha256']
    report=json.loads(REPORT.read_text())
    check_report(decision,report)
    plan=json.loads(PLAN.read_text())
    assert plan['checkpoint_sha256']==decision['checkpoint_sha256']
    assert sha(plan['checkpoint'])==decision['checkpoint_sha256']
    assert plan['split_episodes']==dict(train=1000,val=100) and plan['raw_rows_per_episode']==1001
    effective,edits=policy_source((V4/'run_profile.py').read_text())
    return decision,effective,edits


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
        original_root=SHADOW.parent.parent
        template=original_root/'official_state_3d_cache_v1/acceleration_v1/chosen_profile_v1.json'
        profile=json.loads(template.read_text())
        profile.update(base_producer=str(SHADOW/'official_state_3d_cache_v1/produce.py'),
            base_producer_sha256=sha(SHADOW/'official_state_3d_cache_v1/produce.py'),
            status='Root_explicit_positive_speed_release_original5percent_gate_failed',
            GPU_preprocess_gate=dict(path=str(REPORT),sha256=decision['report_sha256']),
            performance_release=binding,legacy199_receipt_SHA_preservation=False,
            resource_profile_reused_from_Triple=True,Double_renderer_speed_not_claimed_before_actual_run=True)
        profile.pop('renderer_selection',None)
        write_once(profile_path,profile)
        write_once(HERE/'profile_preparation_receipt.json',dict(actual_exit_code=0,
            profile_sha256=sha(profile_path),decision_sha256=DECISION_SHA,
            old_report_accepted=False,old_performance_criterion_passed=False,
            GPU_NN_or_render_executed=False,only_checkpoint_bytes_SHA_read=True))
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
