"""Longer timing window only; reuse the accepted official state smoke helpers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def worker():
    import smoke as s
    assert s.jax.default_backend() == 'gpu'
    out = HERE / 'results' / 'profile'
    out.mkdir(parents=True, exist_ok=False)
    cfg = s.config()
    s.np.random.seed(0)
    data = s.dataset(out, cfg)
    example = data.sample(1)
    agent = s.GCIQLAgent.create(0, example['observations'], example['actions'], cfg)
    for _ in range(64):
        agent, info = agent.update(data.sample(1024))
    s.jax.block_until_ready((agent, info))
    windows = []
    for number in range(3):
        s.phase(out, 'timed_window', window=number + 1, updates=2048)
        sample_seconds = 0.0
        wall_started = time.time()
        started = time.perf_counter()
        for _ in range(2048):
            sample_started = time.perf_counter()
            batch = data.sample(1024)
            sample_seconds += time.perf_counter() - sample_started
            agent, info = agent.update(batch)
        s.jax.block_until_ready((agent, info))
        elapsed = time.perf_counter() - started
        wall_finished = time.time()
        observed = s.metrics(info)
        assert observed['grad/norm'] > 0
        windows.append(dict(updates=2048, seconds=elapsed, Hz=2048 / elapsed,
            host_sample_seconds=sample_seconds, wall_started=wall_started,
            wall_finished=wall_finished, last_metrics=observed))
    assert int(agent.network.step) == 6209 and int(agent.network.opt_state[0].count) == 6208
    s.write_report(out / 'report.json', dict(actual_exit_code=0, windows=windows,
        warmup_updates=64, timed_updates=6144, total_updates=6208,
        config=cfg.to_dict(), current_dim=46, goal_dim=46, batch_size=1024,
        input_dtype='float32', official_sampler_and_agent_unmodified=True,
        fixture_only=True, fixture_rows=8, artificial_terminal=True,
        full_data_throughput_proven=False, formal_training=False, SR=False))


def supervise():
    sys.path.insert(0, str(HERE.parent))
    import prepare_runtime as r
    r.ROOT = HERE / 'profile_runtime'
    r.ROOT.mkdir(exist_ok=False)
    settings = json.loads((HERE / 'runtime/settings.json').read_text())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key in ('PYTHONPATH', 'XLA_FLAGS', 'JAX_DEFAULT_MATMUL_PRECISION',
                'JAX_THREEFRY_PARTITIONABLE', 'LD_LIBRARY_PATH'):
        environment.pop(key, None)
    os.sched_setaffinity(0, settings['affinity'])
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
        r.run('profile', [str(HERE.parent / 'env_runtime/bin/python'), '-B',
                         str(HERE / 'benchmark_profile.py'), '--worker'], environment)
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    code = 1
    try:
        worker() if '--worker' in sys.argv else supervise()
        code = 0
    finally:
        label = 'profile_worker' if '--worker' in sys.argv else 'profile_supervisor'
        (HERE / (label + '_actual_exit_code')).write_text(str(code) + '\n')
