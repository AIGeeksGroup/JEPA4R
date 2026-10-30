"""Parameterize the original serial supervisor for a Quad visual role."""
import argparse
import hashlib
import json
from pathlib import Path
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
SOURCE = ROOT / 'cube23_distance_gciql_v1/five_group_extension_v1/run_extension.py'
EXPECTED = 'af4c4526697a49d041db9720a460f06dac1867a94796b32443f564aa99bacd48'


def run(directory):
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    source = raw.decode()
    edits = [
        ("len(plan['phases']) == 20", "len(plan['phases']) == 4*len(GROUPS)"),
        ("state['accepted_groups'] == state['trained_groups'] == 5",
         "state['accepted_groups'] == state['trained_groups'] == len(GROUPS)"),
        ("stage='five_trainings_and_acceptance_complete'", "stage='quad_visual_training_and_acceptance_complete'"),
    ]
    effective = source
    for before, after in edits:
        assert effective.count(before) == 1
        effective = effective.replace(before, after)
    restored = effective
    for before, after in reversed(edits):
        restored = restored.replace(after, before)
    assert restored == source
    original = types.ModuleType('_preserved_quad_visual_supervisor')
    original.__file__ = str(SOURCE)
    exec(compile(effective, str(SOURCE), 'exec'), original.__dict__)
    original.HERE = directory.resolve()
    plan = json.loads((directory / 'launch_plan.json').read_text())
    assert plan['groups'] in (['G11'], ['G12'])
    original.GROUPS = plan['groups']
    code = 1
    owned = not (directory / 'runtime').exists()
    try:
        original.main()
        code = 0
    finally:
        if owned and (directory / 'runtime').exists():
            with (directory / 'runtime/outer_exit_code').open('x') as stream:
                stream.write(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory)
