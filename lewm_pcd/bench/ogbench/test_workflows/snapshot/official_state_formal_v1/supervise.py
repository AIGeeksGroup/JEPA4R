"""Persistent owner-only supervision; official training loop is not copied."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import prepare_runtime as guard


def write(name, value):
    guard.write(name, value)


def main():
    plan = json.loads((HERE / 'launch_plan.json').read_text())
    freeze = json.loads((HERE / 'launch_freeze.json').read_text())
    for relative, expected in freeze['files'].items():
        p = ROOT / relative
        assert p.stat().st_size == expected['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256'], relative
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    guard.ROOT = out
    (out / 'tmp').mkdir()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    environment = dict(os.environ)
    for name in ['PYTHONPATH', 'XLA_FLAGS', 'JAX_DEFAULT_MATMUL_PRECISION',
                 'JAX_THREEFRY_PARTITIONABLE', 'JAX_PLATFORMS']:
        environment.pop(name, None)
    environment.update(plan['environment'])
    environment['TMPDIR'] = str(out / 'tmp')
    process_list = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    assert not process_list, 'GPU is occupied; no overlap authorized'
    write('launch_plan.json', plan)
    write('launch_freeze.json', freeze)
    write('status.json', dict(stage='runtime_check', started=time.time(), actual_exit_code=None))
    with (out / 'runtime_check.log').open('x') as log:
        p = subprocess.run(plan['runtime_check_argv'], stdout=log, stderr=subprocess.STDOUT,
                           env=environment, cwd=plan['cwd'])
    write('runtime_check_exit.json', dict(actual_exit_code=p.returncode, at=time.time()))
    assert p.returncode == 0, 'Runtime check failed; no trainer started'
    before = guard.events()
    with (out / 'train.log').open('x') as log:
        child = subprocess.Popen(plan['argv'], stdout=log, stderr=subprocess.STDOUT,
            env=environment, cwd=plan['cwd'], start_new_session=True)
        write('actual_argv.json', dict(argv=plan['argv'], cwd=plan['cwd'],
            owned_pgid=child.pid, started=time.time(), environment={name: environment.get(name)
                for name in list(plan['environment']) + ['TMPDIR', 'PYTHONPATH', 'XLA_FLAGS',
                    'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_THREEFRY_PARTITIONABLE']},
            process_scope='owned PGID; not proof that detached services share this PGID'))
        write('status.json', dict(stage='official_main_running', started=time.time(),
            trainer_pid=child.pid, actual_exit_code=None, input_mode='prefetch1'))
        while child.poll() is None:
            owned = guard.members(child.pid)
            now = guard.events()
            rss = sum(p['RSS'] for p in owned)
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            with (out / 'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(at=time.time(), owned_members=owned,
                    RSS_sum=rss, RSS_sum_is_conservative_not_PSS=True,
                    memory_events=now, GPU_csv=gpu.stdout.strip())) + '\n')
            if rss > plan['own_RSS_limit_bytes'] or any(
                    now[k] > before[k] for k in ['oom', 'oom_kill']):
                write('resource_stop.json', dict(RSS_sum=rss, events=now,
                    reason='own RSS bound or new cgroup OOM event', at=time.time()))
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(10)
        code = child.wait()
    residual = guard.members(child.pid)
    write('trainer_exit.json', dict(actual_exit_code=code, residual_members=residual,
        completed=time.time()))
    (out / 'trainer_actual_exit_code').write_text(str(code) + '\n')
    assert code == 0 and not residual, 'Trainer failed or owned processes remain'
    write('status.json', dict(stage='official_main_completed_final_acceptance_pending',
        completed=time.time(), actual_exit_code=0))


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException as exc:
        if guard.ROOT != ROOT:
            write('supervisor_failure.json', dict(error=repr(exc), at=time.time(), actual_exit_code=1))
        raise
    finally:
        (HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
