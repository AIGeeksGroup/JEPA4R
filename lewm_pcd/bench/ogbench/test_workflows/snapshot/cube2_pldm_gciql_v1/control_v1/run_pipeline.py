"""Serial Cube2 PLDM downstream pipeline with persistent phase state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube2_pldm_gciql_v1'
PY = ROOT / 'env_runtime/bin/python'
GPU_LEASE = ROOT / 'cube4_visual_gciql_v1/control_v1/gpu_lease.py'
GROUP = 'P2_cube2_pldm_rgb'
RUNTIME = CASE / 'control_v1/runtime_v1'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def update_status(value):
    path = RUNTIME / 'status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def command(script, *args, gpu=False):
    base = [str(PY), '-B', str(script), *map(str, args)]
    if gpu:
        base = [str(PY), '-B', str(GPU_LEASE), '--', *base]
    return base


def phases():
    config = CASE / 'functional_v1/configs/P2_cube2_pldm_rgb.v1.json'
    plan = CASE / 'cache_v1/producer_plan.json'
    gate = CASE / 'gates' / GROUP
    run = CASE / 'runs' / GROUP
    return [
        ('00_prepare_cache', command(CASE / 'cache_v1/prepare_plan.py', '--case', CASE), False),
        ('01_cache_pilot', command(CASE / 'cache_v1/producer_bridge.py', '--plan', plan, '--execute', '--pilot'), True),
        ('02_cache_full', command(CASE / 'cache_v1/producer_bridge.py', '--plan', plan, '--execute'), True),
        ('03_cache_audit', command(CASE / 'cache_v1/audit_bridge.py', '--cache-dir', CASE / 'cache_v1/cache', '--plan', plan, '--output-dir', CASE / 'cache_v1/independent_audit'), False),
        ('04_prepare_train', command(CASE / 'functional_v1/prepare.py', 'train', '--case', CASE), False),
        ('05_GCIQL_hooks', command(CASE / 'functional_v1/pldm_runtime.py', 'verify-hooks', '--report', CASE / 'functional_v1/HOOK_VERIFY.json'), False),
        ('06_GCIQL_gate', command(CASE / 'functional_v1/pldm_runtime.py', 'gate', '--config', config, '--output', gate), True),
        ('07_GCIQL_reload', command(CASE / 'functional_v1/pldm_runtime.py', 'reload', '--config', config, '--output', gate, '--report', CASE / f'acceptance/{GROUP}_gate_reload.json'), False),
        ('08_GCIQL_train', command(CASE / 'functional_v1/pldm_runtime.py', 'train', '--config', config), True),
        ('09_GCIQL_accept', command(CASE / 'functional_v1/pldm_runtime.py', 'accept', '--config', config, '--output', run, '--report', CASE / f'acceptance/{GROUP}_full_acceptance.json'), False),
        ('10_prepare_native', command(CASE / 'functional_v1/prepare.py', 'native', '--case', CASE), False),
        ('11_native_tiny', command(CASE / 'functional_v1/native_sr.py', '--case', CASE, '--mode', 'tiny', '--output', CASE / 'native_sr_v1/tiny'), True),
        ('12_native_tiny_audit', command(CASE / 'functional_v1/audit_native.py', '--case', CASE, '--mode', 'tiny', '--output', CASE / 'native_sr_v1/tiny'), False),
        ('13_native_full', command(CASE / 'functional_v1/native_sr.py', '--case', CASE, '--mode', 'full', '--output', CASE / 'native_sr_v1/full'), True),
        ('14_native_full_audit', command(CASE / 'functional_v1/audit_native.py', '--case', CASE, '--mode', 'full', '--output', CASE / 'native_sr_v1/full'), False),
    ]


def main():
    global RUNTIME
    plan = phases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--start', choices=[item[0] for item in plan], default=plan[0][0])
    parser.add_argument('--runtime-tag', default='runtime_v1')
    args = parser.parse_args()
    start_index = next(index for index, item in enumerate(plan) if item[0] == args.start)
    selected_plan = plan[start_index:]
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, case=str(CASE), phases=[item[0] for item in selected_plan],
                              no_job_started=True)))
        return 0
    assert args.runtime_tag and all(c.isalnum() or c in '._-' for c in args.runtime_tag)
    RUNTIME = CASE / 'control_v1' / args.runtime_tag
    if RUNTIME.exists(): raise FileExistsError(f'Immutable runtime already exists: {RUNTIME}')
    RUNTIME.mkdir(parents=True)
    started = time.time()
    update_status(dict(stage='starting', start_phase=args.start,
                       started=started, updated=started, failed_stage=None))
    for index, (name, argv, gpu) in enumerate(selected_plan, start=start_index):
        phase = RUNTIME / name
        phase.mkdir()
        write_new(phase / 'actual_argv.json', dict(argv=argv, cwd=str(ROOT), gpu_lease=gpu))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   JAX_THREEFRY_PARTITIONABLE='false',
                   MUJOCO_GL='egl', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        runtime_lib = str(ROOT / 'runtime/lib')
        env['LD_LIBRARY_PATH'] = runtime_lib + (f":{env['LD_LIBRARY_PATH']}" if env.get('LD_LIBRARY_PATH') else '')
        if gpu: env.pop('JAX_PLATFORMS', None)
        else: env['JAX_PLATFORMS'] = 'cpu'
        update_status(dict(stage='running', phase=name, phase_index=index,
                           started=started, updated=time.time(), failed_stage=None))
        with (phase / 'stdout.log').open('xb') as stdout, (phase / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            write_new(phase / 'process.json', dict(pid=process.pid, started=time.time()))
            code = process.wait()
        (phase / 'actual_exit_code').write_text(f'{code}\n')
        update_status(dict(stage='running' if code == 0 else 'failed', phase=name,
                           phase_index=index, phase_exit_code=code, started=started,
                           updated=time.time(), failed_stage=None if code == 0 else name))
        if code != 0:
            (RUNTIME / 'outer_exit_code').write_text(f'{code}\n')
            return code
    completed = time.time()
    update_status(dict(stage='pipeline_complete', started=started, completed=completed,
                       updated=completed, failed_stage=None))
    (RUNTIME / 'outer_exit_code').write_text('0\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
