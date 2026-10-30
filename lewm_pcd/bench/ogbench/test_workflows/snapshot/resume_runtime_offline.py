"""Install transported exact large wheels first, then only missing small pins."""
import hashlib
import json
import os
import time
from prepare_runtime import ROOT,run,write


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    os.nice(10)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    py=str(ROOT/'env_runtime/bin/python')
    manifest=json.loads((ROOT/'large_wheels_manifest.json').read_text())
    wheels=[]
    for member in manifest['files']:
        path=ROOT/'large_wheels'/member['filename']
        assert path.stat().st_size==member['bytes'] and sha(path)==member['sha256'],path
        wheels.append(str(path))
    write('transported_wheels_acceptance.json',dict(actual_exit_code=0,
        files=len(wheels),bytes=manifest['bytes'],all_SHA_length_exact=True,at=time.time()))
    write('resume_offline_status.json',dict(stage='install_exact_transport_wheels',actual_exit_code=None,updated=time.time()))
    run('install_explicit_transport_wheels',[py,'-m','pip','install','--no-deps','--no-index',*wheels],env)
    small=[str(p) for p in sorted((ROOT/'reused_download_wheels').glob('*.whl'))]
    if small:run('install_explicit_reused_small_wheels',[py,'-m','pip','install','--no-deps','--no-index',*small],env)
    write('resume_offline_status.json',dict(stage='install_only_missing_pins',actual_exit_code=None,updated=time.time()))
    run('install_remaining_exact_pins',[py,'-m','pip','install','--prefer-binary','--no-cache-dir',
        '--progress-bar','off','--timeout','60','--retries','4',
        '--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',
        '-r',str(ROOT/'requirements.runtime.lock')],env)
    run('pip_check_offline_resume',[py,'-m','pip','check'],env)
    run('pip_freeze_offline_resume',[py,'-m','pip','freeze'],env)
    write('resume_offline_status.json',dict(stage='installed_runtime_checks_pending',actual_exit_code=0,completed=time.time()))


if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException as exc:
        write('resume_offline_failure.json',dict(actual_exit_code=1,error=repr(exc),at=time.time()))
        raise
    finally:(ROOT/'resume_offline_actual_exit_code').write_text(str(code)+'\n')
