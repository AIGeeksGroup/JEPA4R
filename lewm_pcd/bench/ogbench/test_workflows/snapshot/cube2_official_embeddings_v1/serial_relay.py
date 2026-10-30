"""Execute only the frozen Cube2 embedding phase list; preserve all actual exits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runtime'


def save(name, value):
    p = RUN / name
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, p)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    RUN.mkdir(exist_ok=False)
    plan = json.loads((HERE / 'launch_plan.json').read_text())
    assert plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'
    assert plan['training_updates'] == plan['SR_episodes'] == 0
    for path, expected in plan['source_pins'].items():
        assert sha(path) == expected, path
    spec = importlib.util.spec_from_file_location('_existing_cube_process_guard', plan['process_guard'])
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    save('actual_dispatch.json', dict(pid=os.getpid(), started=time.time(),
        launch_plan_sha256=sha(HERE / 'launch_plan.json'), actual_argv=__import__('sys').argv))
    for step in plan['phases']:
        name = step['name']
        assert '/' not in name and '..' not in name
        env = dict(os.environ, **step.get('environment', {}))
        save('status.json', dict(stage=name, started=time.time(), actual_exit_code=None))
        with (RUN / (name + '.log')).open('x') as log:
            child = subprocess.Popen(step['argv'], cwd=step['cwd'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(name + '_argv.json', dict(argv=step['argv'], cwd=step['cwd'],
                 pid=child.pid, started=time.time()))
            baseline = guard.events()
            while child.poll() is None:
                members, events = guard.members(child.pid), guard.events()
                rss = sum(p['RSS'] for p in members)
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
                with (RUN / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.time(), phase=name, owned_PGID=child.pid,
                        RSS_sum=rss, RSS_sum_is_conservative_not_PSS=True, memory_events=events,
                        GPU_csv=gpu.stdout.strip(), member_count=len(members))) + '\n')
                if rss > plan['own_RSS_limit_bytes'] or any(events[k] > baseline[k] for k in ('oom', 'oom_kill')):
                    save(name + '_resource_stop.json', dict(RSS_sum=rss, events=events, at=time.time()))
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        residual = []
        for _ in range(20):
            residual = guard.members(child.pid)
            if not residual:
                break
            time.sleep(.25)
        save(name + '_exit.json', dict(actual_exit_code=code, residual_owned_members=residual, completed=time.time()))
        assert code == 0 and not residual, f'{name} exited {code}; no automatic fallback or duplicate run'
        for check in step.get('checks', []):
            record = json.loads(Path(check['path']).read_text())
            for key, expected in check['equals'].items():
                assert record[key] == expected, (name, key)
    save('closure.json', dict(actual_exit_code=0, completed=time.time(),
         cube_count=2, data_id='cube-double-play-v0', completed_phases=len(plan['phases']),
         expected_features_per_modality=1101100, training_updates=0, SR_episodes=0))
    save('status.json', dict(stage='all_embedding_phases_completed', actual_exit_code=0, completed=time.time()))


if __name__ == '__main__':
    code = 1
    try:
        main()
        code = 0
    except BaseException:
        if RUN.exists():
            save('failure.json', dict(actual_exit_code=1, error=traceback.format_exc(), at=time.time()))
            save('status.json', dict(stage='failed', actual_exit_code=1, updated=time.time()))
        raise
    finally:
        if RUN.exists():
            (RUN / 'outer_exit_code').write_text(str(code) + '\n')
