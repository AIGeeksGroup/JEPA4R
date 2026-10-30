"""Reuse the existing embedding supervisor with Quad identity/count only."""
import hashlib
import json
from pathlib import Path
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
HERE = Path(__file__).resolve().parent / 'PCD'
RUN = ROOT / 'cube4_official_embeddings_v1/pcd_training_camera_v1/runtime_v1'
SOURCE = ROOT / 'cube2_official_embeddings_v1/serial_relay.py'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'
source = raw.decode()
edits = [
    ("plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'",
     "plan['cube_count'] == 4 and plan['data_id'] == 'cube-quadruple-play-v0'"),
    ("cube_count=2, data_id='cube-double-play-v0'", "cube_count=4, data_id='cube-quadruple-play-v0'"),
    ('expected_features_per_modality=1101100', 'expected_features_per_modality=5505500'),
]
effective = source
for before, after in edits:
    assert effective.count(before) == 1
    effective = effective.replace(before, after)
restored = effective
for before, after in reversed(edits):
    restored = restored.replace(after, before)
assert restored == source
original = types.ModuleType('_quad_PCD_original_serial')
original.__file__ = str(SOURCE)
exec(compile(effective, str(SOURCE), 'exec'), original.__dict__)
original.HERE, original.RUN = HERE, RUN

if __name__ == '__main__':
    owned = not RUN.exists()
    code = 1
    try:
        original.main()
        code = 0
    except BaseException as error:
        if owned and RUN.exists():
            original.save('failure.json', dict(actual_exit_code=1, error=repr(error)))
            original.save('status.json', dict(stage='failed', actual_exit_code=1, error=repr(error)))
        raise
    finally:
        if owned and RUN.exists():
            with (RUN / 'outer_exit_code').open('x') as stream:
                stream.write(str(code) + '\n')
