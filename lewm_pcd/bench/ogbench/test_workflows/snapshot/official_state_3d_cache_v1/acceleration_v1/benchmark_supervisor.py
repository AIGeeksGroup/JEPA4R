"""Apply the original RSS/PGID guard to the benchmark's single process group."""
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parent
spec = importlib.util.spec_from_file_location('_existing_pcd_supervisor', H / 'supervise.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
old = module.old

if __name__ == '__main__':
    old.HERE = HERE / 'benchmark_v1'
    original_write = old.write

    def write(name, value):
        if name == 'status.json':
            value = dict(value)
            value['stage'] = 'performance_' + value['stage']
            value.pop('input_mode', None)
            value.update(training_updates=0, SR_episodes=0, own_RSS_cap_GiB=24)
        original_write(name, value)

    old.write = write
    code = 1
    try:
        os.nice(10)
        old.main()
        code = 0
    except BaseException as error:
        if old.guard.ROOT != H.parent:
            write('supervisor_failure.json', dict(actual_exit_code=1, error=repr(error)))
        raise
    finally:
        (old.HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
