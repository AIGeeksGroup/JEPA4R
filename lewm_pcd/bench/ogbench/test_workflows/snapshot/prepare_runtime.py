"""Install the pinned downstream environment; supervise only our own child group."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CG = Path('/sys/fs/cgroup')


def write(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def events():
    return {a: int(b) for a, b in map(str.split, (CG / 'memory.events').read_text().splitlines())}


def members(group):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                rss = int(next(x.split()[1] for x in (entry / 'status').read_text().splitlines()
                               if x.startswith('VmRSS:'))) * 1024
                found.append(dict(pid=int(entry.name), RSS=rss))
        except (OSError, StopIteration, ProcessLookupError):
            pass
    return found


def run(name, argv, environment):
    before = events()
    with (ROOT / (name + '.log')).open('x') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        write(name + '_argv.json', dict(argv=argv, owned_pgid=child.pid, started=time.time()))
        while child.poll() is None:
            own = members(child.pid)
            now = events()
            rss = sum(p['RSS'] for p in own)
            with (ROOT / 'prepare_resources.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(at=time.time(), phase=name, RSS_sum=rss,
                                             RSS_is_conservative_not_PSS=True, events=now)) + '\n')
            if rss > 4 * 1024**3 or any(now[k] > before[k] for k in ['max', 'oom', 'oom_kill']):
                write(name + '_resource_stop.json', dict(RSS_sum=rss, events=now))
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(2)
        code = child.wait()
    write(name + '_exit.json', dict(actual_exit_code=code, residual_members=members(child.pid),
                                    completed=time.time()))
    if code:
        raise RuntimeError(name + ' failed: ' + str(code))


def main():
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    write('prepare_status.json', dict(stage='creating_environment', actual_exit_code=None,
                                     started=time.time(), CPU_cores=2, own_RSS_limit_GiB=4))
    run('create_environment', [sys.executable, '-m', 'venv', str(ROOT / 'env_runtime')], env)
    py = str(ROOT / 'env_runtime/bin/python')
    write('prepare_status.json', dict(stage='installing_pinned_dependencies', actual_exit_code=None,
                                     updated=time.time(), mirror='https://mirrors.aliyun.com/pypi/simple'))
    run('install_runtime', [py, '-m', 'pip', 'install', '--no-cache-dir', '--prefer-binary',
        '--progress-bar', 'off', '--timeout', '60', '--retries', '4',
        '--index-url', 'https://mirrors.aliyun.com/pypi/simple',
        '-r', str(ROOT / 'requirements.runtime.lock')], env)
    run('pip_check', [py, '-m', 'pip', 'check'], env)
    run('pip_freeze', [py, '-m', 'pip', 'freeze'], env)
    write('prepare_status.json', dict(stage='installed_runtime_checks_pending', actual_exit_code=0,
        completed=time.time(), requirements_sha256=hashlib.sha256(
        (ROOT / 'requirements.runtime.lock').read_bytes()).hexdigest(), formal_training_started=False))


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException as exc:
        write('prepare_failure.json', dict(error=repr(exc), actual_exit_code=1, at=time.time()))
        raise
    finally:
        (ROOT / 'prepare_actual_exit_code').write_text(str(code) + '\n')
