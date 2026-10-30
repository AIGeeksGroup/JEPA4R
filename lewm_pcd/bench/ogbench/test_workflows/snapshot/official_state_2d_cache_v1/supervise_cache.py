"""Reuse the frozen process guard; protect the independent CPU SR job."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'official_state_formal_v1/supervise.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
    'c7594f7335036bbc39c7eab5e764537fd89f3f363e6444a59dbe678b95aaa9ea')
spec = importlib.util.spec_from_file_location('_embedding_original_guard', SOURCE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
supervisor.HERE = HERE
original_write = supervisor.write


def write(name, value):
    if name == 'status.json':
        value = dict(value)
        value['stage'] = {
            'official_main_running': 'embedding_pipeline_running',
            'official_main_completed_final_acceptance_pending': 'embedding_pipeline_complete',
        }.get(value['stage'], value['stage'])
        value.pop('input_mode', None)
        value['training_updates_in_this_process'] = 0
        value['modality'] = 'frozen_2d_LeWM_epoch10'
    original_write(name, value)


supervisor.write = write
if __name__ == '__main__':
    code = 1
    try:
        os.nice(10)
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:16][-8:])
        supervisor.main()
        # The relay publishes its checked paths after producer and audit exit0.
        closure = json.loads((HERE / 'runtime/pipeline_closure.json').read_text())
        assert closure['producer_actual_exit_code'] == 0
        assert closure['audit_actual_exit_code'] == 0
        target = Path(closure['cache_dir']) / 'READY.json'
        assert not target.exists()
        closure['ready_for_downstream'] = True
        closure['outer_exit_file_required'] = str(HERE / 'supervisor_actual_exit_code')
        with target.open('x') as stream:
            json.dump(closure, stream, indent=2)
            stream.write('\n')
        code = 0
    except BaseException as exc:
        if supervisor.guard.ROOT != ROOT:
            write('supervisor_failure.json', dict(error=repr(exc), actual_exit_code=1))
        raise
    finally:
        (HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
