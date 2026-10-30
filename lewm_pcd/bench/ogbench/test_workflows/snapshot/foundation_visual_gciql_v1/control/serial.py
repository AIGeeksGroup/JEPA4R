"""Reuse the accepted process-group supervisor for a frozen four-Cube phase list."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import types
from barrier import validate_plan, require_smoke

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
SOURCE = ROOT/'cube2_official_embeddings_v1/serial_relay.py'
SOURCE_SHA = 'ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'


def load_supervisor(directory):
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    replacements = [
        ("assert plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'",
         "validate_plan(plan)"),
        ("assert plan['training_updates'] == plan['SR_episodes'] == 0",
         "require_smoke(HERE.parent/'smoke', plan) if plan['queue']=='formal' else None"),
        ("cube_count=2, data_id='cube-double-play-v0'",
         "cube_counts=plan['cube_counts'], families=plan['families']"),
        ('expected_features_per_modality=1101100, training_updates=0, SR_episodes=0',
         "expected_features_per_modality=plan['expected_features'], training_updates=plan['training_updates'], SR_episodes=plan['SR_episodes']"),
        ("stage='all_embedding_phases_completed'", "stage='pipeline_complete'"),
    ]
    source = raw.decode()
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    module = types.ModuleType('_foundation_original_serial')
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    module.HERE, module.RUN = Path(directory), Path(directory)/'runtime'
    module.validate_plan,module.require_smoke=validate_plan,require_smoke
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-directory', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    supervisor = load_supervisor(args.plan_directory)
    if not args.execute:
        print(json.dumps({'prepared_only': True, 'original_supervisor_sha256': SOURCE_SHA}))
        return 0
    owned = not supervisor.RUN.exists()
    code = 1
    try:
        supervisor.main()
        code = 0
    except BaseException:
        if owned and supervisor.RUN.exists():
            supervisor.save('failure.json', dict(actual_exit_code=1, error=traceback.format_exc(), at=time.time()))
            supervisor.save('status.json', dict(stage='failed', actual_exit_code=1, updated=time.time(), automatic_restart=False))
        raise
    finally:
        if owned and supervisor.RUN.exists():
            with (supervisor.RUN/'outer_exit_code').open('x') as out:
                out.write(str(code)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
