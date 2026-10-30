"""Keep accepted encoder/render evidence; finish diagnostic save/reload only."""
import hashlib
import json
import os
import time
from prepare_runtime import ROOT,run,write


def main():
    os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    assert (ROOT/'smoke_pipeline_v3_actual_exit_code').read_text().strip()=='1'
    assert json.loads((ROOT/'smoke_v3_gciql_two_updates_exit.json').read_text())['residual_members']==[]
    for name in ['renderer_report.json','rgb_layout_report.json','encoders_report.json']:
        assert json.loads((ROOT/'smoke'/name).read_text())['actual_exit_code']==0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.20',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for key in ['PYTHONPATH','LD_LIBRARY_PATH','XLA_FLAGS','JAX_DEFAULT_MATMUL_PRECISION','JAX_THREEFRY_PARTITIONABLE']:
        env.pop(key,None)
    py=str(ROOT/'env_runtime/bin/python')
    for name,args in [('gciql_two_updates',[]),('fresh_GCIQL_reload',['--fresh'])]:
        write('smoke_pipeline_v4_status.json',dict(stage=name,actual_exit_code=None,updated=time.time()))
        run('smoke_v4_'+name,[py,str(ROOT/'smoke_gciql_v4.py'),*args],env)
    reports={}
    for name in ['renderer_report.json','rgb_layout_report.json','encoders_report.json',
                 'gciql_report_v4.json','reload_report_v4.json']:
        b=(ROOT/'smoke'/name).read_bytes();assert json.loads(b)['actual_exit_code']==0
        reports[name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    write('smoke_pipeline_v4_status.json',dict(stage='limited_migration_GPU_smoke_passed',
        actual_exit_code=0,completed=time.time(),reports=reports,
        reused_passed_encoder_and_renderer=True,actual_GCIQL_smoke_batch=16,
        formal_B1024_memory_proven=False,final_2d_checkpoint_present=True,
        final_3d_checkpoint_present=False,source_3d_epoch=6,full_dataset_cache_present=False,
        formal_training_started=False,SR_started=False,
        preserved_failed_auxiliary_attempts=['RGB_layout_v2','save_directory_v3']))


if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException as exc:
        write('smoke_pipeline_v4_failure.json',dict(actual_exit_code=1,error=repr(exc),at=time.time()))
        raise
    finally:(ROOT/'smoke_pipeline_v4_actual_exit_code').write_text(str(code)+'\n')
