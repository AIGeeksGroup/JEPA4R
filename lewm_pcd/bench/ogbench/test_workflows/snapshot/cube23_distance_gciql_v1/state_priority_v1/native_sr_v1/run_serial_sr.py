"""Serial SR only; reuse the existing priority process/resource supervisor."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
PRIORITY, ROOT = HERE.parent, HERE.parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def main(plan_path):
    spec = importlib.util.spec_from_file_location('_frozen_state_priority_supervisor', PRIORITY / 'run_priority.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    guard = old.release_guard()
    plan = load(plan_path)
    assert plan['groups'] == ['G01', 'G04', 'G07'] and plan['training_updates'] == 0
    for path, expected in plan['source_pins'].items():
        assert old.sha(path) == expected, path
    assert load(PRIORITY / 'runtime/status.json')['accepted_groups'] == 3
    for path in (PRIORITY / 'runtime/outer_exit_code', PRIORITY / 'wrapper_exit_code'):
        assert path.read_text().strip() == '0', path
    for group in plan['groups']:
        report = load(PRIORITY / 'acceptance' / group / 'report.json')
        assert report['actual_exit_code'] == 0 and report['Adam_count'] == 1000000
    run = HERE / 'runtime'
    run.mkdir(exist_ok=False)
    state = dict(started=time.time(), actual_exit_code=None, completed_phases=0,
                 groups=plan['groups'], training_updates=0, plan_sha256=old.sha(plan_path))
    child = None
    try:
        for stage in plan['phases']:
            name = stage['name']
            assert '/' not in name and '..' not in name
            folder = run / name
            folder.mkdir()
            state.update(stage=name, group_id=stage.get('group_id'), updated=time.time())
            old.save(run / 'status.json', state)
            env = dict(old.child_environment(cpu=True), **stage.get('environment', {}))
            baseline = guard.events()
            with (folder / 'run.log').open('x') as log:
                child = subprocess.Popen(stage['argv'], cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                old.save(folder / 'actual_argv.json', dict(argv=stage['argv'], cwd=str(ROOT),
                         owned_PGID=child.pid, started=time.time(), environment={key: env.get(key)
                         for key in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'OMP_NUM_THREADS',
                                     'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MUJOCO_GL', 'XLA_FLAGS')}))
                while child.poll() is None:
                    sample = old.resources(guard, child.pid)
                    with (folder / 'resources.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample) + '\n')
                    if sample['RSS_sum'] > 24 * 1024**3 or any(
                            sample['memory_events'][key] > baseline[key] for key in ('oom', 'oom_kill')):
                        old.save(folder / 'resource_stop.json', sample)
                        old.stop_owned(guard, child)
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.wait()
            for _ in range(20):
                if not guard.members(child.pid):
                    break
                time.sleep(.25)
            residual = guard.members(child.pid)
            record = dict(actual_exit_code=code, residual_owned_members=residual, completed=time.time())
            (folder / 'actual_exit_code').write_text(str(code) + '\n')
            old.save(folder / 'exit.json', record)
            assert code == 0 and not residual and not (folder / 'resource_stop.json').exists(), name
            child = None
            for check in stage.get('checks', []):
                report = load(check['path'])
                for key, value in check['equals'].items():
                    assert report[key] == value, (name, key)
            state['completed_phases'] += 1
        state.update(stage='three_final1M_SR_and_audits_complete', actual_exit_code=0, completed=time.time())
    except BaseException:
        if child is not None:
            old.stop_owned(guard, child)
        state.update(stage='failed', actual_exit_code=1, error=traceback.format_exc(), updated=time.time())
        raise
    finally:
        old.save(run / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, training_updates=0)))
    else:
        owned = not (HERE / 'runtime').exists()
        code = 1
        try:
            main(args.plan)
            code = 0
        finally:
            if owned and (HERE / 'runtime').exists():
                with (HERE / 'runtime/outer_exit_code').open('x') as stream:
                    stream.write(str(code) + '\n')
