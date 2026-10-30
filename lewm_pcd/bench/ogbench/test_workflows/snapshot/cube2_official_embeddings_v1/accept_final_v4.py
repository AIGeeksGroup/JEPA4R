"""Run the first complete corrected CPU audit, then publish the accepted lineage."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
RUN=HERE/'runtime_v4'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)


def positive(runtime,name):
    r=load(HERE/runtime/(name+'_exit.json'))
    assert r['actual_exit_code']==0 and not r['residual_owned_members'],name


def main():
    RUN.mkdir(exist_ok=False)
    control=load(HERE/'control_v3/launch_plan.json')
    for p,h in control['source_pins'].items():assert sha(p)==h,p
    assert sha(HERE/'audit_final_v2.py')=='5eabdd9f475ca30868f10f0d66d6beaf6b11ab77f995bc888305c8453fed1b0c'
    for n in ('2d_pilot','2d_full','2d_independent_audit','3d_reference_gate','3d_fresh_gate','3d_gpu_preprocessing_gate'):
        positive('runtime_v2',n)
    for n in ('released_profile_prepare','3d_released_pilot','3d_released_full'):positive('runtime_v3',n)
    assert load(HERE/'runtime_v3/3d_independent_audit_exit.json')['actual_exit_code']==1
    assert (HERE/'wrapper_v2_exit_code').read_text().strip()==(HERE/'wrapper_v3_exit_code').read_text().strip()=='1'
    argv=[sys.executable,'-B',str(HERE/'audit_final_v2.py'),'--cache-dir',str(HERE/'3d/cache'),
          '--plan',str(HERE/'workspace/official_state_3d_cache_v1/producer_plan.json'),
          '--output-dir',str(HERE/'3d/independent_audit_v2')]
    save(RUN/'actual_argv.json',dict(argv=argv,started=time.time(),CUDA_VISIBLE_DEVICES='',new_NN=False))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (RUN/'audit.log').open('x') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        code=child.wait()
    spec=importlib.util.spec_from_file_location('_existing_cpu_process_guard',HERE.parent/'prepare_runtime.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    remaining=guard.members(child.pid)
    save(RUN/'audit_exit.json',dict(actual_exit_code=code,residual_owned_members=remaining,completed=time.time()))
    assert code==0 and not remaining,'corrected CPU audit failed'
    summaries={}
    for mode in ('2d','3d'):
        audit=HERE/mode/('independent_audit' if mode=='2d' else 'independent_audit_v2')
        r=load(audit/'report.json')
        assert r['actual_exit_code']==0 and (audit/'actual_exit_code').read_text().strip()=='0'
        assert r['episodes_verified']==1100 and r['raw_rows_verified']==1101100 and r['main_rows_verified']==1100000
        assert r['latent_dim']==192 and r['dtype']=='float32' and r['all_new_latent_SHA_shape_dtype_finite']
        assert r['all_raw_ID_and_true_terminal_next_contracts' if mode=='2d' else 'all_canonical_ID_and_terminal_next_contracts']
        summaries[mode]=dict(episodes=1100,features=1101100,latent_dim=192,dtype='float32',audit_report=str(audit/'report.json'),audit_sha256=sha(audit/'report.json'))
    for p,h in control['source_pins'].items():assert sha(p)==h,p
    original=load(HERE/'3d/GPU_preprocess_report.json')
    assert original['accepted'] is False and original['prepared_bitexact'] and original['latent_bitexact']
    for mode,info in summaries.items():
        ready=dict(info,actual_exit_code=0,ready_for_downstream=True,cube_count=2,data_id='cube-double-play-v0',
            modality=mode,manifest_sha256=sha(HERE/mode/'cache/cache_manifest.json'),
            producer_runtime='runtime_v2' if mode=='2d' else 'runtime_v3',final_acceptance_runtime='runtime_v4',
            original_performance_gate_accepted=False,performance_release=str(HERE/'performance_release_v1/release_decision.json'),
            original_path_auditor_failed=True,corrected_3D_audit_dir=str(HERE/'3d/independent_audit_v2'),
            previous_wrapper_v2_and_v3_exit_codes_preserved=1,final_wrapper_exit_required=str(HERE/'wrapper_v4_exit_code'),
            original_cube3_protected_source_pins_exact=True,training_updates=0,SR_episodes=0,
            full_vectors_local_backup=False,completed=time.time())
        save(HERE/mode/'cache/READY.json',ready)
    save(HERE/'BOTH_CACHES_READY.json',dict(actual_exit_code=0,cube_count=2,modes=summaries,
        features_per_modality=1101100,final_wrapper_exit_required=str(HERE/'wrapper_v4_exit_code'),completed=time.time()))
    save(RUN/'closure.json',dict(actual_exit_code=0,modes=summaries,old_failed_records_preserved=True,completed=time.time()))
    print(json.dumps(dict(actual_exit_code=0,modes=summaries)))


if __name__=='__main__':
    code=1
    try:
        main();code=0
    except BaseException:
        if RUN.exists():save(RUN/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()))
        raise
    finally:
        if RUN.exists():(RUN/'outer_exit_code').write_text(str(code)+'\n')
