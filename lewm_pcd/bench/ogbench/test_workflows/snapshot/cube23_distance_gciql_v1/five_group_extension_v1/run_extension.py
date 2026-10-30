"""Serial five-role relay, using the existing owned-process resource guard."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
PRIORITY = HERE.parent / 'state_priority_v1'
ROOT = HERE.parent.parent
GROUPS = ['G10', 'G05', 'G06', 'G08', 'G09']


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = load(HERE / 'launch_plan.json')
    assert plan['groups'] == GROUPS and len(plan['phases']) == 20
    for path, expected in plan['source_pins'].items():
        assert sha(path) == expected, path
    spec = importlib.util.spec_from_file_location('_preserved_priority_supervisor', PRIORITY / 'run_priority.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    guard = old.release_guard()
    assert load(PRIORITY / 'runtime/status.json')['accepted_groups'] == 3
    assert load(PRIORITY / 'native_sr_v1/runtime/status.json')['actual_exit_code'] == 0
    run = HERE / 'runtime'
    run.mkdir(exist_ok=False)
    state = dict(started=time.time(), actual_exit_code=None, completed_phases=0,
        groups=GROUPS, gated_groups=0, reloaded_groups=0, trained_groups=0, accepted_groups=0,
        plan_sha256=sha(HERE / 'launch_plan.json'))
    child = None
    try:
        for index, stage in enumerate(plan['phases']):
            assert stage['group_id'] == GROUPS[index // 4]
            assert stage['action'] == ['gate', 'reload', 'train', 'accept'][index % 4]
            name = stage['name']
            assert '/' not in name and '..' not in name
            folder = run / name
            folder.mkdir()
            state.update(stage=stage['action'], group_id=stage['group_id'], updated=time.time(),
                         phase_directory=str(folder))
            old.save(run / 'status.json', state)
            if not stage['cpu']:
                gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                assert not gpu.stdout.strip(), 'Unrelated GPU process; preserve it and stop this relay'
            env = old.child_environment(stage['cpu'])
            baseline = guard.events()
            with (folder / 'run.log').open('x') as log:
                child = subprocess.Popen(stage['argv'], cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                old.save(folder / 'actual_argv.json', dict(argv=stage['argv'], cwd=str(ROOT),
                    owned_PGID=child.pid, started=time.time(), environment={key: env.get(key)
                    for key in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_FLAGS',
                                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}))
                while child.poll() is None:
                    sample = old.resources(guard, child.pid)
                    for target in (folder / 'resources.jsonl', run / 'resources.jsonl'):
                        with target.open('a') as stream:
                            stream.write(json.dumps(dict(group_id=stage['group_id'], action=stage['action'], **sample)) + '\n')
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
            count_key = {'gate': 'gated_groups', 'reload': 'reloaded_groups',
                         'train': 'trained_groups', 'accept': 'accepted_groups'}[stage['action']]
            state[count_key] += 1
            state['completed_phases'] += 1
        assert state['accepted_groups'] == state['trained_groups'] == 5
        state.update(stage='five_trainings_and_acceptance_complete', actual_exit_code=0, completed=time.time())
    except BaseException:
        if child is not None:
            old.stop_owned(guard, child)
        state.update(stage='failed', actual_exit_code=1, error=traceback.format_exc(), updated=time.time())
        raise
    finally:
        old.save(run / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, groups=GROUPS, formal_training_started=False)))
    else:
        owned = not (HERE / 'runtime').exists()
        code = 1
        try:
            main()
            code = 0
        finally:
            if owned and (HERE / 'runtime').exists():
                with (HERE / 'runtime/outer_exit_code').open('x') as stream:
                    stream.write(str(code) + '\n')
