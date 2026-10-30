"""Persist only source/CPU unit test logs and actual exits; never launches training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    case = Path(__file__).resolve().parent
    results = []
    for name in ['test_source_contract.py', 'test_cpu_recovery.py']:
        with (args.output / (name + '.log')).open('x') as stream:
            result = subprocess.run([sys.executable, '-B', str(case / name)], stdout=stream,
                                    stderr=subprocess.STDOUT)
        results.append({'test': name, 'actual_exit_code': result.returncode})
    payload = {'schema': 'recovery_cpu_checks_v1', 'at_utc': datetime.now(timezone.utc).isoformat(),
               'python': sys.executable, 'results': results,
               'training_encoder_constructed': False, 'gpu_used': False,
               'formal_training_started': False}
    (args.output / 'result.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload))
    return int(any(item['actual_exit_code'] != 0 for item in results))


if __name__ == '__main__':
    sys.exit(main())
