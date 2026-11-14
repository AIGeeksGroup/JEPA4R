"""Small helpers shared by the RH56 modules: JSON output and attribute bindings for the length of a block."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def temporary_bindings(*changes):
    """Set (owner, name, value) for the block; the originals come back in reverse order, also after a failed setup."""
    with ExitStack() as restore:
        for owner, name, value in changes:
            original = getattr(owner, name)
            restore.callback(setattr, owner, name, original)
            setattr(owner, name, value)
        yield
