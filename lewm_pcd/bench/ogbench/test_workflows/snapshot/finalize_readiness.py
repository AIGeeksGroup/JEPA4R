"""CPU-only final readback; adopts actual completed tests, never reruns NN."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def read(name):return json.loads((ROOT/name).read_text())


def main():
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    assert (ROOT/'resume_offline_actual_exit_code').read_text().strip()=='0'
    exits={}
    for name in ['pip_check_offline_resume','smoke_interface_CPU','smoke_renderer',
        'smoke_v3_RGB_layout','smoke_v3_encoders','smoke_v4_gciql_two_updates',
        'smoke_v4_fresh_GCIQL_reload']:
        data=read(name+'_exit.json')
        assert data['actual_exit_code']==0 and not data['residual_members'],name
        exits[name]=data
    freeze=read('deployment_freeze_v4.json')
    for name,expected in freeze['files'].items():
        data=(ROOT/name).read_bytes()
        assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],name
    versions_script="""import importlib.metadata as m,json,pathlib
p=pathlib.Path(__import__('sys').argv[1]);bad={};actual={}
for line in p.read_text().splitlines():
 if '==' not in line:continue
 name,want=line.split('==');got=m.version(name);actual[name]=got
 if got!=want:bad[name]={'expected':want,'actual':got}
assert not bad,bad
print(json.dumps({'actual_exit_code':0,'verified_count':len(actual),'versions':actual}))
"""
    proc=subprocess.run([str(ROOT/'env_runtime/bin/python'),'-c',versions_script,
        str(ROOT/'requirements.runtime.lock')],capture_output=True,text=True,check=True)
    versions=json.loads(proc.stdout)
    (ROOT/'installed_exact_versions_acceptance.json').write_text(json.dumps(versions,indent=2))
    migration=read('migration_acceptance.json');assert migration['actual_exit_code']==0
    # Explicitly consume gzip to EOF, including its trailer CRC.
    unpacked=0
    with gzip.open(ROOT/'incoming/cube_downstream_sources_models_v1.tar.gz','rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):unpacked+=len(block)
    gzip_receipt=dict(actual_exit_code=0,gzip_CRC_EOF_passed=True,uncompressed_tar_bytes=unpacked)
    (ROOT/'gzip_CRC_acceptance.json').write_text(json.dumps(gzip_receipt,indent=2))
    names=['renderer_report.json','rgb_layout_report.json','encoders_report.json',
           'gciql_v4_report.json','reload_report_v4.json']
    reports={}
    for name in names:
        data=(ROOT/'smoke'/name).read_bytes();record=json.loads(data)
        assert record['actual_exit_code']==0,name
        reports[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    g=read('smoke/gciql_v4_report.json');fresh=read('smoke/reload_report_v4.json')
    assert g['saved_state']==fresh['full_state_exact'] and g['saved_state']['leaves']==153
    assert g['network_step']==3 and g['Adam_count']==2
    assert g['official_task_alpha']==1 and g['official_sampler_reward_and_host_RNG_equal']
    assert g['current_dims']==211 and g['goal_dims']==201 and fresh['finite_actions']
    encoder=read('smoke/encoders_report.json')
    assert encoder['encoder_batches']['2d']['accepted_final_checkpoint']
    assert not encoder['encoder_batches']['3d']['accepted_final_checkpoint']
    base_path='/root/miniconda3/lib/python3.12/site-packages'
    import importlib.metadata as metadata
    base={d.metadata['Name'].lower().replace('_','-'):d.version for d in metadata.distributions(path=[base_path]) if d.metadata.get('Name')}
    protected={'torch':'2.7.0+cu128','torchvision':'0.22.0+cu128','numpy':'2.2.6',
               'nvidia-cudnn-cu12':'9.7.1.26','nvidia-cublas-cu12':'12.8.3.14'}
    assert all(base[k]==v for k,v in protected.items())
    events={a:int(b) for a,b in map(str.split,Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
    assert not any(events[k] for k in ['max','oom','oom_kill'])
    resource_rows=[json.loads(line) for line in (ROOT/'prepare_resources.jsonl').read_text().splitlines()]
    ready=dict(status='migration_and_limited_GPU_smoke_accepted',actual_exit_code=0,
        completed=time.time(),host='101732Test',root=str(ROOT),source_commit=read('migration_manifest.json')['official_commit'],
        migration=migration,gzip_CRC=gzip_receipt,selected_actual_exits=exits,
        reports=reports,exact_dependency_pins_verified=versions['verified_count'],
        full_state_reload_exact=g['saved_state'],source_and_input_contracts_unchanged=True,
        original_target_base_metadata_unchanged=protected,
        observed_max_own_group_RSS=max(r['RSS_sum'] for r in resource_rows),
        RSS_monitor_is_conservative_not_PSS=True,memory_events=events,
        Torch_encoder_peak_allocated=encoder['peak_cuda_allocated'],
        checkpoints={'2d':{'epoch':10,'accepted_final':True},'3d':{'epoch':6,'accepted_final':False}},
        tested_Cube_input_counts=[1,2,3,4],active_Cube_count=3,
        formal_training_started=False,SR_started=False,full_dataset_or_cache_present=False,
        formal_B1024_memory_proven=False,Triple_Native_full_binding_proven=False,
        actual_GCIQL_smoke_batch=16,accepted_smoke_updates=2,
        retired_auxiliary_failures_preserved=['RGB_layout_v2','save_directory_v3','report_filename_collector_v4'],
        report_alias_contents_changed=False,NN_repeated_for_final_aggregation=False)
    (ROOT/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
    state=read('root_status.json')
    state.update(current_stage=ready['status'],minimal_GPU_smoke_actual_exit_code=0,
        environment_pins_verified=versions['verified_count'],pending=[
        'accepted_final_3D_checkpoint','full_official_data_and_aligned_embedding_cache',
        'formal_B1024_capacity_check','Triple_Native_full_binding'],
        authoritative_paths=dict(ready='READY.json',finalize_exit='finalize_actual_exit_code',
            producer_reports='smoke/',runtime='resume_offline_status.json',freeze='deployment_freeze_v4.json'),
        updated_at=time.time())
    (ROOT/'root_status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(status=ready['status'],actual_exit_code=0,
        versions=versions['verified_count'],own_RSS=ready['observed_max_own_group_RSS'],
        Torch_peak=encoder['peak_cuda_allocated'],memory_events=events)))


if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException as exc:
        (ROOT/'finalize_failure.json').write_text(json.dumps(dict(actual_exit_code=1,error=repr(exc),at=time.time()),indent=2))
        raise
    finally:(ROOT/'finalize_actual_exit_code').write_text(str(code)+'\n')
