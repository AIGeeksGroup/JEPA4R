"""Reuse the frozen owner-only supervisor, with evaluation-only status labels."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'official_state_formal_v1/supervise.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
    'c7594f7335036bbc39c7eab5e764537fd89f3f363e6444a59dbe678b95aaa9ea')
spec = importlib.util.spec_from_file_location('_last3_frozen_supervisor', SOURCE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
supervisor.HERE = HERE
original_write = supervisor.write


def write(name, value):
    if name == 'status.json':
        value = dict(value)
        value['stage'] = {
            'official_main_running': 'last3_evaluation_running',
            'official_main_completed_final_acceptance_pending': 'last3_evaluation_completed',
        }.get(value['stage'], value['stage'])
        value.pop('input_mode', None)
        value['training_updates_in_this_process'] = 0
        value['checkpoint_steps'] = [800000, 900000, 1000000]
    original_write(name, value)


supervisor.write = write
if __name__ == '__main__':
    code = 1
    try:
        supervisor.main()
        code = 0
    except BaseException as exc:
        if supervisor.guard.ROOT != ROOT:
            write('supervisor_failure.json', dict(error=repr(exc), actual_exit_code=1))
        raise
    finally:
        (HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
