"""Persist source/config regression only, in process; no torch or subprocess."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    case = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('source_config_v2_test', case / 'test_source_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (args.output / 'source_config_tests.log').open('x') as stream:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    rc = int(not result.wasSuccessful())
    receipt = {'schema': 'config_serialization_v2_source_checks',
               'at_utc': datetime.now(timezone.utc).isoformat(), 'actual_exit_code': rc,
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'scope': 'actual native argparse -> original native config JSON; source contract',
               'unaffected_cpu_math_save_tests': 'reuse frozen v1 evidence; not rerun',
               'torch_imported': 'torch' in sys.modules, 'training_launched': False}
    (args.output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return rc


if __name__ == '__main__':
    sys.exit(main())
