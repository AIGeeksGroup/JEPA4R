"""Existing memory/PGID guard, applied to isolated 3D cache phases."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
source = ROOT / 'official_state_formal_v1/supervise.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'c7594f7335036bbc39c7eab5e764537fd89f3f363e6444a59dbe678b95aaa9ea'
spec = importlib.util.spec_from_file_location('_original_owned_supervisor', source)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('preflight_v1', 'full_v1'), required=True)
    args = parser.parse_args()
    old.HERE = HERE / args.profile
    original_write = old.write

    def write(name, value):
        if name == 'status.json':
            value = dict(value)
            value['stage'] = {'official_main_running': args.profile + '_running',
                'official_main_completed_final_acceptance_pending': args.profile + '_completed'}.get(value['stage'], value['stage'])
            value.pop('input_mode', None)
            value.update(modality='frozen_final3D_LeWM_epoch10', training_updates=0, SR_episodes=0)
        original_write(name, value)

    old.write = write
    code = 1
    try:
        os.nice(10)
        old.main()
        if args.profile == 'full_v1':
            ready = json.loads((old.HERE / 'runtime/closure.json').read_text())
            assert ready['actual_exit_code'] == 0 and ready['features'] == 3303300
            ready.update(ready_for_downstream=True,
                outer_exit_file_required=str(old.HERE / 'supervisor_actual_exit_code'))
            with (HERE / 'cache/READY.json').open('x') as stream:
                json.dump(ready, stream, indent=2)
                stream.write('\n')
        code = 0
    except BaseException as error:
        if old.guard.ROOT != ROOT:
            write('supervisor_failure.json', dict(error=repr(error), actual_exit_code=1))
        raise
    finally:
        (old.HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
