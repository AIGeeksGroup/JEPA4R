"""Only supply the two-Cube phase contract to the existing serial supervisor."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'pldm_cube14_embeddings_v1'
PY = ROOT / 'env_runtime/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def validate(plan):
    assert plan['cube_counts'] == [1, 4] and plan['families'] == ['pldm']
    assert plan['queue'] == 'encoding_only' and plan['training_updates'] == plan['SR_episodes'] == 0
    pilot = plan['mode'] == 'pilot'
    assert plan['mode'] in ('pilot', 'full')
    assert plan['expected_features'] == (4004 if pilot else 6606600)
    suffixes = ['00_pilot'] if pilot else ['01_encode_full', '02_cache_audit', '03_deliver']
    assert [s['name'] for s in plan['phases']] == [f'pldm_cube{n}_{suffix}' for n in (1, 4) for suffix in suffixes]
    for step in plan['phases']:
        assert step['cwd'] == str(ROOT) and step['argv'][0] == str(PY)
        assert not any(x in ' '.join(step['argv']).lower() for x in ['train.py', 'native_sr.py', 'gciql.py'])
    if not pilot:
        runtime = CASE / 'control/pilot/runtime'
        assert (runtime / 'outer_exit_code').read_text().strip() == '0'
        assert (runtime.parent / 'wrapper_exit_code').read_text().strip() == '0'
        assert read(runtime / 'closure.json')['actual_exit_code'] == 0
        for n in (1, 4):
            phase = read(runtime / f'pldm_cube{n}_00_pilot_exit.json')
            assert phase['actual_exit_code'] == 0 and phase['residual_owned_members'] == []
            group_plan = read(CASE / f'groups/pldm_cube{n}/plan.json')
            cache = Path(group_plan['cache_dir'])
            receipt, proof = read(cache / 'pilot_receipt.json'), read(cache / 'batch1_alignment.json')
            assert receipt['actual_exit_code'] == 0 and receipt['not_full_cache'] is True
            assert proof['actual_exit_code'] == 0 and proof['passed'] is True and proof['finite'] is True
            assert proof['encoder_spec_sha256'] == group_plan['encoder_spec_sha256']
            assert proof['batch1_shape'] == [1, 512] and proof['batch32_shape'] == [32, 512]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', choices=['pilot', 'full'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'foundation_visual_gciql_v1/control'))
    import serial
    serial.validate_plan = validate
    directory = CASE / 'control' / args.queue
    validate(read(directory / 'launch_plan.json'))
    sys.argv = [str(__file__), '--plan-directory', str(directory)] + (['--execute'] if args.execute else [])
    raise SystemExit(serial.main())
