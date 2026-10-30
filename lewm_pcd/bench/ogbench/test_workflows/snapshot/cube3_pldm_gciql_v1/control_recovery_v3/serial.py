"""Resume only the uncompleted phases through the frozen original supervisor."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import time
import traceback

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT/'cube3_pldm_gciql_v1'
HERE = CASE/'control_recovery_v3'
SOURCE = CASE/'control_v1/serial.py'
SOURCE_SHA = 'b46b65085a33243f2ed2e76eefffd632102d241e827b99c744110419659b28f2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('prepared_only; no resumed phase started')
        return 0
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('_frozen_P3_serial', SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    runtime = HERE/'runtime_v1'
    old = original.load_supervisor(HERE, runtime)
    owned, code = not runtime.exists(), 1
    try:
        old.main()
        code = 0
    except BaseException:
        if owned and runtime.exists():
            old.save('failure.json', dict(actual_exit_code=1, error=traceback.format_exc(), at=time.time()))
            old.save('status.json', dict(stage='failed', actual_exit_code=1, updated=time.time(), automatic_restart=False))
        raise
    finally:
        if owned and runtime.exists():
            with (runtime/'outer_exit_code').open('x') as stream:
                stream.write(str(code)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
