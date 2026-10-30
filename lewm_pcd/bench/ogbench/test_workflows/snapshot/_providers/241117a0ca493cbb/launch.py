"""Small process driver using the existing logging/atomic-write helpers."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from train import helpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output')
    parser.add_argument('--launcher-output', required=True)
    parser.add_argument('--total-updates', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--gate-report')
    parser.add_argument('--gate-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); utility = helpers(config)
    output = Path(args.output or config['output'])
    total = args.total_updates or config['total_updates']
    command = [sys.executable, str(Path(__file__).with_name('train.py')), '--config', str(Path(args.config).resolve()),
               '--output', str(output), '--total-updates', str(total)]
    if args.resume: command += ['--resume', args.resume]
    plan = dict(argv=command, output=str(output), total_updates=total, launcher_output=args.launcher_output,
                config_sha256=utility.sha(args.config), evaluation_episodes=0,
                protocol=config['protocol_label'], rng_deviation=config['rng_deviation'])
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    assert args.gate_report and args.gate_sha256, 'Fresh reviewed configured-batch/val/resume GPU gate required'
    assert utility.sha(args.gate_report) == args.gate_sha256
    gate = json.loads(Path(args.gate_report).read_text())
    assert gate['status'] == 'passed' and gate['exit_code'] == 0
    current_code = {p.name: utility.sha(p) for p in Path(__file__).parent.iterdir()
                    if p.name in (('train.py', 'reader.py', 'launch.py', 'test_formal.py', 'config.json', 'gciql.py')
                                  if config.get('input_profile') == 'cached' else
                                  ('train.py', 'reader.py', 'launch.py', 'test_formal.py', 'config.json'))}
    assert gate['code'] == current_code and gate['formal_config_sha256'] == utility.sha(args.config)
    assert not output.exists(), 'New run segment required; preserve old checkpoints/logs'
    launch = Path(args.launcher_output); launch.mkdir(parents=True, exist_ok=False)
    profile = Path(config['runtime_profile'])
    assert utility.sha(profile) == config['runtime_profile_sha256']
    raw_env = subprocess.check_output(['bash', '-ec', 'source "$1"; env -0', 'profile', str(profile)])
    environment = dict(item.split('=', 1) for item in raw_env.decode().split('\0') if item)
    environment.update(JAX_PLATFORMS='cuda,cpu', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    utility.atomic_json(launch / 'plan.json', plan)
    state = dict(stage='starting', started_at=utility.now(), updated_at=utility.now(), exit_code=None, argv=command)
    utility.atomic_json(launch / 'driver_status.json', state)
    code = 1
    try:
        with (launch / 'run.log').open('x') as log:
            child = subprocess.Popen(command, cwd=Path(__file__).parent, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            state.update(stage='training', child_pid=child.pid)
            utility.atomic_json(launch / 'driver_status.json', state)
            actual = child.wait()
        (launch / 'child_exit_code').write_text(str(actual) + '\n')
        assert actual == 0
        trainer = json.loads((output / 'status.json').read_text())
        assert trainer['exit_code'] == 0 and trainer['completed_updates'] == total
        assert trainer['network_step'] == total + 1 and trainer['adam_count'] == total
        assert trainer['validation_count'] == total // config['validation_interval']
        state.update(stage='training_exit0_pending_independent_reload', trainer_status=trainer, exit_code=0)
        code = 0
    except BaseException:
        state.update(stage='failed', error=traceback.format_exc(), exit_code=1)
        raise
    finally:
        state['updated_at'] = utility.now()
        utility.atomic_json(launch / 'driver_status.json', state)
        (launch / 'driver_exit_code').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
