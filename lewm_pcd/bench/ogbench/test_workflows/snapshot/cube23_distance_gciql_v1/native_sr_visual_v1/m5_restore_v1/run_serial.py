"""Run corrected Native SR using existing checkpoints; never train or overwrite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit import verify_report

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
BASE = HERE.parent


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    release = json.loads((HERE / 'release_freeze.json').read_text())
    for path, expected in release['target_pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    RUNTIME.mkdir(exist_ok=False)
    plan = json.loads((HERE / 'launch_plan.json').read_text())
    code = 1
    completed = []
    try:
        for stage in plan['stages']:
            folder = RUNTIME / stage['name']
            folder.mkdir()
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            assert max(int(row.strip()) for row in gpu.splitlines()) <= 512, 'GPU still occupied'
            save(folder / 'actual_argv.json', stage)
            status = dict(stage=stage['name'], completed_phases=completed,
                          actual_exit_code=None, updated=time.time())
            save(RUNTIME / 'status.json', status)
            with (folder / 'run.log').open('xb') as log:
                child = subprocess.Popen(stage['argv'], cwd=BASE, env=os.environ.copy(),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                save(folder / 'process.json', dict(pid=child.pid, started=time.time(), argv=child.args))
                while child.poll() is None:
                    output = Path(stage['output']) / 'status.json'
                    if output.exists():
                        status['native'] = json.loads(output.read_text())
                    proc = Path('/proc') / str(child.pid)
                    if proc.exists():
                        status['process_stat'] = (proc / 'stat').read_text()
                        status['process_status'] = (proc / 'status').read_text()
                    status['updated'] = time.time()
                    save(RUNTIME / 'status.json', status)
                    time.sleep(10)
            (folder / 'actual_exit_code').write_text(str(child.returncode) + '\n')
            assert child.returncode == 0, stage['name']
            result = verify_report(BASE, stage['group'], stage['native_mode'], Path(stage['output']),
                                   stage['expected_episodes'], stage['goal_dim'], stage['max_steps'])
            threads = json.loads((Path(stage['output']) / 'runtime_threads.json').read_text())
            assert threads['torch_num_threads'] == 1 and threads['numpy_physics']['BLAS_threads'] == 4
            save(folder / 'audit.json', dict(actual_exit_code=0, summary=result, threads=threads))
            completed.append(stage['name'])
        code = 0
        save(RUNTIME / 'status.json', dict(stage='complete', completed_phases=completed,
                                          actual_exit_code=0, completed=time.time()))
    except BaseException as error:
        save(RUNTIME / 'failure.json', dict(actual_exit_code=1, error=repr(error), completed_phases=completed))
        save(RUNTIME / 'status.json', dict(stage='failed', actual_exit_code=1, error=repr(error), completed_phases=completed))
        raise
    finally:
        (RUNTIME / 'outer_exit_code').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
