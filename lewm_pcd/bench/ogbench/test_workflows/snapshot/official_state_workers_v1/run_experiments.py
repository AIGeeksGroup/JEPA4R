"""One serial, bounded full-data comparison; no formal training or SR."""
import hashlib
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
import prepare_runtime as r


def main():
    acceptance = HERE / 'data_prepare/ACCEPTANCE.json'
    stat_freeze = HERE / 'data_prepare/stat-freeze.json'
    accepted = json.loads(acceptance.read_text())
    frozen = json.loads(stat_freeze.read_text())
    assert accepted['stage'] == 'complete' and accepted['actual_exit_code'] == 0
    assert frozen['actual_exit_code'] == 0
    assert frozen['acceptance_sha256'] == hashlib.sha256(acceptance.read_bytes()).hexdigest()
    assert (HERE / 'cpu_actual_exit_code').read_text().strip() == '0'
    cpu = json.loads((HERE / 'cpu_runtime/real_CPU_pytest_exit.json').read_text())
    assert cpu['actual_exit_code'] == 0 and not cpu['residual_members']
    process_list = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    assert not process_list, 'GPU is occupied; do not overlap other experiments'
    r.ROOT = HERE / 'experiment_runtime'
    r.ROOT.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'XLA_FLAGS',
                'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_THREEFRY_PARTITIONABLE'):
        env.pop(key, None)
    python = str(ROOT / 'env_runtime/bin/python')
    common = ['--data-dir', str(ROOT / 'data/official_state'),
              '--data-receipt', str(acceptance), '--data-stat-freeze', str(stat_freeze)]
    correct = HERE / 'results/correctness'
    jobs = [('correctness', 'correctness', 'baseline', correct, []),
            ('fresh', 'fresh', 'baseline', HERE / 'results/fresh',
             ['--reference-dir', str(correct)])]
    for label, mode in [('baseline_A', 'baseline'), ('prefetch1', 'prefetch1'),
                        ('workers2', 'workers2'), ('workers4', 'workers4'),
                        ('baseline_B', 'baseline')]:
        jobs.append((label, 'benchmark', mode, HERE / 'results' / label, []))
    r.write('settings.json', dict(affinity=sorted(os.sched_getaffinity(0)),
        library_threads=4, own_RSS_limit_GiB=4, preallocate=False, GPU_fraction=.45,
        full_data_required=True, batch_size=1024, task_alpha=1.0,
        formal_training=False, SR=False, ahead_batches=1,
        job_order=[job[0] for job in jobs]))
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
    try:
        for label, stage, mode, output, extra in jobs:
            r.write('status.json', dict(stage=label, started=time.time(),
                actual_exit_code=None, formal_training=False, SR=False))
            r.run(label, [python, '-B', str(HERE / 'measure.py'), '--stage', stage,
                '--mode', mode, '--out', str(output), *common, *extra], env)
        r.write('status.json', dict(stage='comparison_complete_selection_pending',
            actual_exit_code=0, completed=time.time(), formal_training=False, SR=False))
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException as exc:
        (HERE / 'experiment_failure.json').write_text(json.dumps(dict(
            actual_exit_code=1, error=repr(exc), at=time.time())) + '\n')
        raise
    finally:
        (HERE / 'experiment_actual_exit_code').write_text(str(code) + '\n')
