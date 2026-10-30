"""Thin Cube3 identity binding over the accepted serial supervisor; default plan only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CONTROL = ROOT/'cube3_pldm_gciql_v1/control_v1'
SOURCE = ROOT/'cube2_official_embeddings_v1/serial_relay.py'
SOURCE_SHA = 'ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'
GUARD_SHA = '0331473db0b0566d3097c8cd9cf7362a820badc8c8537134ea39388ab79e6063'
EDITS = [
    ("plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'",
     "plan['cube_count'] == 3 and plan['data_id'] == 'cube-triple-play-v0'"),
    ("cube_count=2, data_id='cube-double-play-v0'", "cube_count=3, data_id='cube-triple-play-v0'"),
    ('expected_features_per_modality=1101100', 'expected_features_per_modality=3303300'),
    ("plan['training_updates'] == plan['SR_episodes'] == 0",
     "plan['training_updates'] == 1000000 and plan['SR_episodes'] == 250"),
    ('training_updates=0, SR_episodes=0', 'training_updates=1000000, SR_episodes=250'),
    ("stage='all_embedding_phases_completed'", "stage='pipeline_complete'"),
]


def load_supervisor(directory, runtime, source=SOURCE):
    source = Path(source); raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    original = raw.decode(); effective = original
    for before, after in EDITS:
        assert effective.count(before) == 1, before
        effective = effective.replace(before, after)
    restored = effective
    for before, after in reversed(EDITS):
        assert restored.count(after) == 1, after
        restored = restored.replace(after, before)
    assert restored == original
    module = types.ModuleType('_Cube3_PLD512_preserved_serial')
    module.__file__ = str(source)
    exec(compile(effective, str(source), 'exec'), module.__dict__)
    module.HERE, module.RUN = Path(directory), Path(runtime)
    module.identity_binding = dict(source_sha256=SOURCE_SHA,
        effective_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        strict_replacement_count=len(EDITS), reverse_source_equals_original=True)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, control=str(CONTROL),
            plan=str(CONTROL/'launch_plan.json'), runtime=str(CONTROL/'runtime_v1'),
            source_sha256=SOURCE_SHA, no_job_started=True)))
        return 0
    runtime = CONTROL/'runtime_v1'
    old = load_supervisor(CONTROL, runtime)
    owned = not runtime.exists(); code = 1
    try:
        old.main(); code = 0
    except BaseException:
        if owned and runtime.exists():
            old.save('failure.json', dict(actual_exit_code=1, error=traceback.format_exc(), at=time.time()))
            old.save('status.json', dict(stage='failed', actual_exit_code=1, updated=time.time(), automatic_restart=False))
        raise
    finally:
        if owned and runtime.exists():
            with (runtime/'outer_exit_code').open('x') as stream: stream.write(str(code)+'\n')
    return code


if __name__ == '__main__': raise SystemExit(main())
