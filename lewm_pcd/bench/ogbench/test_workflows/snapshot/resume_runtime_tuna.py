"""Continue only the owned mirror installation; preserve the retired attempt."""
import json
import os
from pathlib import Path
import time
from prepare_runtime import ROOT, run, write


def main():
    assert (ROOT/'prepare_actual_exit_code').read_text().strip() == '1'
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    py = str(ROOT/'env_runtime/bin/python')
    write('resume_tuna_status.json', dict(stage='installing_same_versions_from_faster_mirror',
        actual_exit_code=None, updated=time.time(), original_lock_unchanged=True))
    # TFP's pinned nightly was verified from Aliyun in the source environment;
    # keep this one direct provider if Tsinghua does not list it.
    run('install_tfp_tuna_resume', [py,'-m','pip','install','--no-deps','--no-cache-dir',
        '--progress-bar','off','--index-url','https://mirrors.aliyun.com/pypi/simple',
        'tfp-nightly==0.26.0.dev20261016'], env)
    run('install_runtime_tuna_resume', [py,'-m','pip','install','--prefer-binary','--no-cache-dir',
        '--progress-bar','off','--timeout','60','--retries','4',
        '--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',
        '--find-links',str(ROOT/'reused_download_wheels'),
        '-r',str(ROOT/'requirements.runtime.lock')], env)
    run('pip_check_tuna_resume', [py,'-m','pip','check'], env)
    run('pip_freeze_tuna_resume', [py,'-m','pip','freeze'], env)
    write('resume_tuna_status.json', dict(stage='installed_runtime_checks_pending',
        actual_exit_code=0, completed=time.time()))


if __name__=='__main__':
    code=1
    try:
        main();code=0
    except BaseException as exc:
        write('resume_tuna_failure.json',dict(error=repr(exc),actual_exit_code=1,at=time.time()))
        raise
    finally:
        (ROOT/'resume_tuna_actual_exit_code').write_text(str(code)+'\n')
