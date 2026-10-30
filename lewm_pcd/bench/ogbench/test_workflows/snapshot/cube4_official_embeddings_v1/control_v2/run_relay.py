"""Reuse the accepted serial supervisor; change only Quad identity/count guards."""
import hashlib
from pathlib import Path
import types
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
SOURCE = ROOT / 'cube2_official_embeddings_v1/serial_relay.py'
EXPECTED = 'ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
text = raw.decode()
changes = [
    ("plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'",
     "plan['cube_count'] == 4 and plan['data_id'] == 'cube-quadruple-play-v0'"),
    ("cube_count=2, data_id='cube-double-play-v0'", "cube_count=4, data_id='cube-quadruple-play-v0'"),
    ('expected_features_per_modality=1101100', 'expected_features_per_modality=5505500'),
]
effective = text
for before, after in changes:
    assert effective.count(before) == 1
    effective = effective.replace(before, after)
restored = effective
for before, after in reversed(changes):
    assert restored.count(after) == 1
    restored = restored.replace(after, before)
assert restored == text
old = types.ModuleType('_quad_preserved_serial_supervisor')
old.__file__ = str(SOURCE)
exec(compile(effective, str(SOURCE), 'exec'), old.__dict__)
old.HERE = HERE
old.RUN = HERE.parent / 'training_camera_v2/runtime_v2'

if __name__ == '__main__':
    code, owned = 1, not old.RUN.exists()
    try:
        old.main()
        code = 0
    except BaseException:
        if owned and old.RUN.exists():
            old.save('failure.json', dict(actual_exit_code=1, error=traceback.format_exc()))
            old.save('status.json', dict(stage='failed', actual_exit_code=1))
        raise
    finally:
        if owned and old.RUN.exists():
            with (old.RUN / 'outer_exit_code').open('x') as stream:
                stream.write(str(code) + '\n')
