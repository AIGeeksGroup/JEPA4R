"""Reuse the frozen owned-group resource guard for benchmarks and continuation."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('benchmark_v2', 'resume_v1'), required=True)
    args = parser.parse_args()
    old.HERE = HERE / args.profile
    original_write = old.write

    def write(name, value):
        if name == 'status.json':
            value = dict(value)
            value['stage'] = args.profile + '_' + value['stage']
            value.pop('input_mode', None)
            value.update(training_updates=0, SR_episodes=0, own_RSS_cap_GiB=24)
        original_write(name, value)

    old.write = write
    code = 1
    try:
        os.nice(10)
        old.main()
        if args.profile == 'resume_v1':
            ready = json.loads((old.HERE / 'runtime/closure.json').read_text())
            assert ready['actual_exit_code'] == 0 and ready['features'] == 3303300
            ready.update(ready_for_downstream=True,
                outer_exit_file_required=str(old.HERE / 'supervisor_actual_exit_code'))
            with (H / 'cache/READY.json').open('x') as stream:
                json.dump(ready, stream, indent=2)
                stream.write('\n')
        code = 0
    except BaseException as error:
        if old.guard.ROOT != H.parent:
            write('supervisor_failure.json', dict(actual_exit_code=1, error=repr(error)))
        raise
    finally:
        (old.HERE / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
