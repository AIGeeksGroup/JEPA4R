"""G01/G04/G07: bounded gates/reloads first, then serial state training/acceptance."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
CASE, ROOT = HERE.parent, HERE.parent.parent
PY, RUN = ROOT / 'env_runtime/bin/python', HERE / 'runtime'
GROUPS = ['G01', 'G04', 'G07']
GUARD_SHA = '0331473db0b0566d3097c8cd9cf7362a820badc8c8537134ea39388ab79e6063'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def release_guard():
    release = load(HERE / 'release_freeze.json')
    base = Path(release.get('base_dir', str(HERE)))
    assert release['files'] and base.resolve().is_relative_to(ROOT)
    assert release['allowed_groups'] == GROUPS
    for name, item in release['files'].items():
        path = Path(item.get('path', name) if isinstance(item, dict) else name)
        path = path if path.is_absolute() else base / path
        expected = item['sha256'] if isinstance(item, dict) else item
        assert path.resolve().is_relative_to(ROOT) and sha(path) == expected, path
    receipt = release['target_cpu_receipt']
    receipt_path = CASE / 'cpu_input_acceptance_v1/COMPLETION_RECEIPT.json'
    assert Path(receipt['path']) == receipt_path and sha(receipt_path) == receipt['sha256']
    assert load(receipt_path)['actual_exit_code'] == 0
    path = ROOT / 'prepare_runtime.py'
    assert sha(path) == GUARD_SHA
    spec = importlib.util.spec_from_file_location('_priority_owned_process_guard', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


def child_environment(cpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else '0',
               JAX_PLATFORMS='cpu' if cpu else 'cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_THREEFRY_PARTITIONABLE='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    flags = env.get('XLA_FLAGS', '').split()
    setting = [x for x in flags if x.startswith('--xla_gpu_autotune_level=')]
    assert not setting or setting == ['--xla_gpu_autotune_level=0'], 'Conflicting autotune flag'
    if not setting:
        env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') + ' --xla_gpu_autotune_level=0').strip()
    return env


def resources(guard, group):
    members = guard.members(group)
    sample = dict(at=time.time(), owned_PGID=group, owned_members=members,
                  RSS_sum=sum(x['RSS'] for x in members), RSS_is_conservative_not_PSS=True,
                  memory_events=guard.events(), owned_CPU_ticks=0, owned_io={})
    for name in ('cpu.max', 'cpu.stat', 'memory.current', 'memory.max', 'memory.stat', 'memory.pressure'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            sample[name] = path.read_text()
    for member in members:
        try:
            proc = Path('/proc', str(member['pid']))
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            sample['owned_CPU_ticks'] += int(fields[11]) + int(fields[12])
            for line in (proc / 'io').read_text().splitlines():
                key, value = line.split(':')
                sample['owned_io'][key] = sample['owned_io'].get(key, 0) + int(value)
        except (OSError, IndexError, ValueError):
            pass
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        sample.update(GPU_csv=gpu.stdout.strip(), GPU_query_exit=gpu.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        sample['GPU_query_error'] = repr(exc)
    return sample


def stop_owned(guard, child):
    if not guard.members(child.pid):
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while guard.members(child.pid) and time.monotonic() < deadline:
        child.poll()
        time.sleep(.25)
    if guard.members(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def main():
    RUN.mkdir(exist_ok=False)
    state = dict(started=time.time(), actual_exit_code=None, configured_groups=GROUPS,
                 gate_reload_pairs=0, trained_groups=0, accepted_groups=0, selected_input_modes={})
    guard = child = None
    sequence = 0

    def phase(group, action, argv, cpu, report=None):
        nonlocal sequence, child
        sequence += 1
        folder = RUN / f'{sequence:02d}_{action}_{group}'
        folder.mkdir()
        env = child_environment(cpu)
        state.update(stage=action, group_id=group, phase_directory=str(folder), updated=time.time())
        save(RUN / 'status.json', state)
        save(folder / 'status.json', dict(stage='starting', actual_exit_code=None, started=time.time()))
        baseline = guard.events()
        with (folder / 'run.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            save(folder / 'actual_argv.json', dict(argv=argv, cwd=str(ROOT), owned_PGID=child.pid,
                 started=time.time(), environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES',
                 'JAX_PLATFORMS', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                 'JAX_THREEFRY_PARTITIONABLE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}))
            while child.poll() is None:
                sample = resources(guard, child.pid)
                for target in (folder / 'resources.jsonl', RUN / 'resources.jsonl'):
                    with target.open('a') as f:
                        f.write(json.dumps(dict(group_id=group, action=action, **sample)) + '\n')
                if sample['RSS_sum'] > 24 * 1024**3 or any(sample['memory_events'][k] > baseline[k] for k in ('oom', 'oom_kill')):
                    save(folder / 'resource_stop.json', sample)
                    stop_owned(guard, child)
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
        remaining = guard.members(child.pid)
        exit_record = dict(actual_exit_code=code, residual_owned_members=remaining, completed=time.time())
        (folder / 'actual_exit_code').write_text(str(code) + '\n')
        save(folder / 'exit.json', exit_record)
        save(folder / 'status.json', dict(stage='child_exited', **exit_record))
        assert code == 0 and not remaining and not (folder / 'resource_stop.json').exists(), (group, action, code)
        if report:
            assert load(report)['actual_exit_code'] == 0, report
        child = None

    try:
        guard = release_guard()
        roles = load(HERE / 'data_plan.json')['roles']
        assert [r['group_id'] for r in roles] == GROUPS
        for item in roles:
            group, cfg = item['group_id'], item['config_path']
            output = HERE / 'gates' / group / 'reference'
            base = [str(PY), '-B', str(HERE / 'state_boundary.py')]
            phase(group, 'gate', base + ['gate', '--config', cfg, '--output', str(output)], False, output / 'report.json')
            selection = output / 'selection.json'
            picked = load(selection)
            assert picked['actual_exit_code'] == 0 and picked['group_id'] == group
            assert picked['selected_input_mode'] in ('baseline', 'prefetch1')
            state['selected_input_modes'][group] = dict(mode=picked['selected_input_mode'], path=str(selection), sha256=sha(selection))
            report = output.parent / 'reload_report.json'
            phase(group, 'reload', base + ['reload', '--config', cfg, '--output', str(output), '--report', str(report)], True, report)
            state['gate_reload_pairs'] += 1
        assert state['gate_reload_pairs'] == 3
        for item in roles:
            group, cfg = item['group_id'], item['config_path']
            selected = state['selected_input_modes'][group]
            assert sha(selected['path']) == selected['sha256']
            phase(group, 'train', [str(PY), '-B', str(HERE / 'state_runtime.py'), '--config', cfg, '--input-mode', selected['mode']], False)
            output = Path(load(cfg)['output'])
            status = load(output / 'status.json')
            assert status['exit_code'] == 0 and status['completed_updates'] == 1000000
            assert (output / 'trainer_exit_code').read_text().strip() == '0'
            state['trained_groups'] += 1
            report = HERE / 'acceptance' / group / 'report.json'
            phase(group, 'accept', [str(PY), '-B', str(HERE / 'state_boundary.py'), 'accept', '--config', cfg,
                  '--output', str(output), '--report', str(report)], True, report)
            assert load(report)['checkpoint_count'] == 11
            state['accepted_groups'] += 1
        assert state['trained_groups'] == state['accepted_groups'] == 3
        state.update(stage='three_state_trainings_and_acceptance_complete', actual_exit_code=0, completed=time.time())
    except BaseException:
        if guard is not None and child is not None:
            stop_owned(guard, child)
        state.update(stage='failed', actual_exit_code=1, error=traceback.format_exc(), updated=time.time())
        raise
    finally:
        save(RUN / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, executed=False, groups=GROUPS,
                              phases=['gate', 'reload', 'train', 'accept'], release_not_checked=True)))
    else:
        assert not args.plan_only, 'Choose execute or plan-only'
        owned = not RUN.exists()
        code = 1
        try:
            main()
            code = 0
        finally:
            if owned and RUN.exists():
                (RUN / 'outer_exit_code').write_text(str(code) + '\n')
