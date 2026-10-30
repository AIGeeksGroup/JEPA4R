"""One opt-in FP32 input experiment, after the official baseline has exited."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prepare_runtime as r


def main():
    assert (HERE / 'profile_supervisor_actual_exit_code').read_text().strip() == '0'
    r.ROOT = HERE / 'candidate_runtime'
    r.ROOT.mkdir(exist_ok=False)
    os.sched_setaffinity(0, json.loads((HERE / 'runtime/settings.json').read_text())['affinity'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'XLA_FLAGS',
                'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_THREEFRY_PARTITIONABLE'):
        env.pop(key, None)
    stop = threading.Event()

    def monitor():
        with (r.ROOT / 'GPU.jsonl').open('x') as stream:
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
    out = HERE / 'results/candidate_correctness'
    jobs = [('candidate_exact', 'candidate_check.py', ['--out', str(out)]),
            ('candidate_fresh', 'smoke.py', ['--out', str(out), '--fresh']),
            ('candidate_bench', 'candidate_check.py',
             ['--out', str(HERE / 'results/candidate_benchmark'), '--bench'])]
    try:
        for name, script, args in jobs:
            r.write('status.json', dict(stage=name, actual_exit_code=None, at=time.time()))
            r.run(name, [str(HERE.parent / 'env_runtime/bin/python'), '-B',
                        str(HERE / script), *args], env)
        r.write('status.json', dict(stage='candidate_completed_not_selected',
            actual_exit_code=0, at=time.time(), enabled_by_default=False))
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    finally:
        (HERE / 'candidate_supervisor_actual_exit_code').write_text(str(code) + '\n')
