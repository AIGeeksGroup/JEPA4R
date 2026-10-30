"""Bind final accepted files to exact filesystem identities after the real verifier exits zero."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
acceptance_path = ROOT / 'ACCEPTANCE.json'
acceptance = json.loads(acceptance_path.read_text())
assert acceptance['stage'] == 'complete' and acceptance['actual_exit_code'] == 0
assert int((ROOT / 'finalize_actual_exit_code').read_text()) == 0
exits = list(ROOT.glob('finalize_exit_*.json'))
assert len(exits) == 1 and json.loads(exits[0].read_text())['exit_code'] == 0
report = {'acceptance_path': str(acceptance_path),
    'acceptance_sha256': hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
    'real_verifier_exit_path': str(exits[0]),
    'real_verifier_exit_sha256': hashlib.sha256(exits[0].read_bytes()).hexdigest(),
    'datasets': {}, 'frozen': time.time(), 'actual_exit_code': 0,
    'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for label, item in zip(['train', 'validation'], acceptance['datasets']):
    raw, schema = item['download'], item['schema_loader']
    assert schema['actual_exit_code'] == raw['actual_exit_code'] == 0
    path = Path(raw['path'])
    stat = path.stat()
    assert stat.st_size == raw['bytes'] == schema['raw_bytes']
    assert raw['sha256'] == schema['raw_sha256'] and str(path) == schema['path']
    report['datasets'][label] = {'path': str(path), 'raw_sha256': raw['sha256'],
        'accepted_stat': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'device': stat.st_dev, 'inode': stat.st_ino}}
assert len(report['datasets']) == 2
output = ROOT / 'stat-freeze.json'
assert not output.exists(), 'Refuse to replace an existing accepted stat freeze'
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
