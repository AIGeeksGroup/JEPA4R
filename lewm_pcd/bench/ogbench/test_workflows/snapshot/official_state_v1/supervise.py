"""Bounded state-only smoke; reuse the existing own-process resource supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import prepare_runtime as runtime


def main():
    runtime.ROOT = HERE / 'runtime'
    runtime.ROOT.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for name in ('PYTHONPATH', 'XLA_FLAGS', 'JAX_DEFAULT_MATMUL_PRECISION',
                 'JAX_THREEFRY_PARTITIONABLE', 'LD_LIBRARY_PATH'):
        environment.pop(name, None)
    python = str(ROOT / 'env_runtime/bin/python')
    runtime.write('settings.json', dict(affinity=sorted(os.sched_getaffinity(0)),
        CPU_quota_cores=16, library_threads=4, own_RSS_limit_GiB=4,
        GPU_memory_fraction=.45, preallocate=False, formal_training=False,
        official_state_dim=46, official_goal_dim=46, batch_size=1024,
        fixture_only=True, full_official_dataset_present=False))
    stop = threading.Event()

    def monitor():
        with (runtime.ROOT / 'gpu_samples.jsonl').open('x') as stream:
            while not stop.is_set():
                p = subprocess.run(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                stream.write(json.dumps(dict(at=time.time(), actual_exit_code=p.returncode,
                    GPU_csv=p.stdout.strip())) + '\n')
                stream.flush()
                stop.wait(.5)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        for phase, extra in [('correctness', []), ('fresh_reload', ['--fresh']),
                             ('baseline_benchmark', ['--bench'])]:
            runtime.write('status.json', dict(stage=phase, updated=time.time(),
                actual_exit_code=None, formal_training=False, SR=False))
            output = HERE / 'results' / ('benchmark' if extra == ['--bench'] else 'correctness')
            runtime.run(phase, [python, str(HERE / 'smoke.py'), '--out', str(output),
                               *extra], environment)
        runtime.write('status.json', dict(stage='state_fixture_smoke_completed',
            completed=time.time(), actual_exit_code=0, formal_training=False, SR=False,
            full_official_data_acceptance=False))
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException as exc:
        if runtime.ROOT != ROOT:
            runtime.write('failure.json', dict(error=repr(exc), actual_exit_code=1,
                                             at=time.time()))
        raise
    finally:
        (HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
