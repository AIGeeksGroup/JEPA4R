"""Small shared file guards and scoped bindings; no training or simulator imports."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_output_location(output, baseline):
    """Keep new outputs outside the existing protected source directories."""
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    protected = [baseline / name for name in ('encoders', 'config', 'bench', 'envs', 'scripts', 'rh56_adapter')]
    if baseline.is_relative_to(output) or any(output.is_relative_to(path) for path in protected):
        raise ValueError('RH56 output must stay outside protected repository sources')


def atomic_json(path, value, *, new=False, **options):
    """Keep existing JSON encoding; use exclusive creation for immutable receipts."""
    path = Path(path)
    text = json.dumps(value, indent=2, **options)
    if new:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        temp = path.with_suffix('.tmp.json')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)


@contextmanager
def temporary_bindings(*changes):
    """Restore existing attributes in reverse order, including partial setup failures."""
    with ExitStack() as restore:
        for owner, name, value in changes:
            original = getattr(owner, name)
            restore.callback(setattr, owner, name, original)
            setattr(owner, name, value)
        yield


RUNTIME_HELPERS_SHA256 = file_sha256(__file__)
