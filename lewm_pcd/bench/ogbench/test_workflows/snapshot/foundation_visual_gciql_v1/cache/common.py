"""Small shared metadata helpers. Importing this module never initializes a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
HERE = ROOT / 'foundation_visual_gciql_v1/cache'
NAMES = {1: 'single', 2: 'double', 3: 'triple', 4: 'quadruple'}
COUNTS = {1: {'train': 1000, 'val': 100}, 2: {'train': 1000, 'val': 100},
          3: {'train': 3000, 'val': 300}, 4: {'train': 5000, 'val': 500}}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                device=s.st_dev, inode=s.st_ino)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Immutable metadata differs: ' + str(path)
        return
    with path.open('x') as f: f.write(data)

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)
