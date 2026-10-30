"""Contract-only wrapper over the same serial process supervisor."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'pldm_cube14_gciql_v1'


def read(p):
    return json.loads(Path(p).read_text())


def validate(plan):
    assert plan['queue'] == 'pldm_downstream' and plan['cube_counts'] == [1, 4] and plan['families'] == ['pldm']
    assert plan['mode'] in ['smoke', 'full']
    smoke = plan['mode'] == 'smoke'
    labels = ['verify_hooks', 'gate', 'reload', 'native_tiny', 'tiny_audit'] if smoke else ['train', 'accept', 'native_full', 'native_audit']
    assert [p['name'] for p in plan['phases']] == [f'pldm_cube{n}_{i:02d}_{label}' for n in (1, 4) for i, label in enumerate(labels, 1)]
    assert (plan['training_updates'], plan['SR_episodes']) == ((0, 0) if smoke else (2000000, 500))
    from prepare import cache_admission
    cache_admission()
    if not smoke:
        runtime = CASE / 'control/smoke/runtime'
        assert (runtime / 'outer_exit_code').read_text().strip() == '0'
        assert (runtime.parent / 'wrapper_exit_code').read_text().strip() == '0'
        assert read(runtime / 'closure.json')['actual_exit_code'] == 0
        for n in (1, 4):
            for i, label in enumerate(['verify_hooks', 'gate', 'reload', 'native_tiny', 'tiny_audit'], 1):
                record = read(runtime / f'pldm_cube{n}_{i:02d}_{label}_exit.json')
                assert record['actual_exit_code'] == 0 and record['residual_owned_members'] == []
            cfg = read(CASE / f'configs/pilot/pldm_cube{n}.json')
            home = Path(cfg['execution_root'])
            for report in [home / f'gates/pldm_cube{n}/report.json', home / f'gates/pldm_cube{n}_reload.json', home / f'native_sr/pldm_cube{n}/independent_audit.json']:
                assert read(report)['actual_exit_code'] == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--queue', choices=['smoke', 'full'], required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / 'foundation_visual_gciql_v1/control'))
    import serial
    serial.validate_plan = validate
    directory = CASE / 'control' / args.queue
    validate(read(directory / 'launch_plan.json'))
    sys.argv = [str(__file__), '--plan-directory', str(directory)] + (['--execute'] if args.execute else [])
    raise SystemExit(serial.main())
