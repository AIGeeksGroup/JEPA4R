"""Persist the authorized argv, child log and real exit; never infer success from PID."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    launch = Path(plan['launcher_output']); launch.mkdir(parents=True, exist_ok=True)
    lock = (launch / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (launch / 'driver_status.json').exists(): raise FileExistsError('Use a new reviewed driver invocation to resume; do not overwrite prior launch')
    state = dict(stage='preflight', started_at=now(), updated_at=now(), exit_code=None,
                 plan=str(args.plan.resolve()), plan_sha256=sha(args.plan), argv=plan['argv'], driver_pid=os.getpid())
    atomic_json(launch / 'driver_status.json', state)
    code = 1
    try:
        for path, expected in plan['frozen_inputs'].items():
            if sha(path) != expected: raise ValueError('Sealed input changed: ' + path)
        for gate in plan['accepted_gates']:
            gate_exit = Path(gate['exit_code']).read_text().strip()
            if gate_exit != '0' or sha(gate['report']) != gate['report_sha256']: raise ValueError('Gate receipt changed: ' + gate['report'])
        env = os.environ.copy(); env.update(plan['environment'])
        with (launch / 'run.log').open('x') as log:
            child = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='training', child_pid=child.pid, updated_at=now())
            atomic_json(launch / 'driver_status.json', state)
            child_exit = child.wait()
        (launch / 'child_exit_code').write_text(str(child_exit) + '\n')
        state.update(child_exit_code=child_exit, updated_at=now())
        if child_exit != 0: raise RuntimeError('Trainer returned ' + str(child_exit))
        status = json.loads((output / 'status.json').read_text())
        if status['exit_code'] != 0 or status['completed_epochs'] != plan['epochs']: raise ValueError('Trainer full epoch/exit evidence differs')
        if status['global_micro_steps'] != plan['expected_total_microbatches'] or status['global_optimizer_steps'] != plan['expected_total_optimizer_updates']:
            raise ValueError('Trainer full microbatch/update counts differ')
        state.update(stage='training_exit0_pending_independent_reload', trainer_final_status=status,
                     updated_at=now(), exit_code=0)
        code = 0
    except BaseException as exc:
        traceback.print_exc(); state.update(stage='failed', error=repr(exc), updated_at=now(), exit_code=1)
    finally:
        atomic_json(launch / 'driver_status.json', state)
        (launch / 'driver_exit_code').write_text(str(code) + '\n')
    return code


if __name__ == '__main__': raise SystemExit(main())
