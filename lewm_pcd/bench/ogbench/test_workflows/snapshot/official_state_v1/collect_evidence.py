"""Archive closed small evidence only; checkpoints remain remote with SHA pointers."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


assert (HERE / 'supervisor_actual_exit_code').read_text().strip() == '0'
assert (HERE / 'profile_supervisor_actual_exit_code').read_text().strip() == '0'
assert (HERE / 'candidate_supervisor_actual_exit_code').read_text().strip() == '0'
for folder, names in [('runtime', ['correctness', 'fresh_reload', 'baseline_benchmark']),
                      ('profile_runtime', ['profile']),
                      ('candidate_runtime', ['candidate_exact', 'candidate_fresh', 'candidate_bench'])]:
    for name in names:
        record = json.loads((HERE / folder / (name + '_exit.json')).read_text())
        assert record['actual_exit_code'] == 0 and record['residual_members'] == []
before = json.loads((HERE / 'preflight.json').read_text())['official_protected_source']
for name, expected in before.items():
    path = HERE.parent / name
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
checkpoints = {str(p.relative_to(HERE)): dict(bytes=p.stat().st_size, sha256=digest(p),
               copied_locally=False) for p in HERE.glob('results/**/params_2.pkl')}
(HERE / 'checkpoint_pointers.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
paths = [p for p in HERE.rglob('*') if p.is_file() and
         (p.suffix in ('.py', '.sh', '.md', '.json', '.jsonl', '.log') or
          p.name.endswith('exit_code') or p.name == 'first_mismatch.npz')]
paths = sorted(p for p in paths if p.name not in ('evidence_receipt.json', 'evidence_manifest.json'))
manifest = {}
target = HERE / 'closed_state_smoke_evidence_v1.tar.gz'
assert not target.exists()
with tarfile.open(target, 'w:gz') as archive:
    for path in paths:
        old_stat = path.stat()
        content = path.read_bytes()
        current = path.stat()
        assert (old_stat.st_size, old_stat.st_mtime_ns, old_stat.st_ino) == (
            current.st_size, current.st_mtime_ns, current.st_ino)
        name = path.relative_to(HERE).as_posix()
        manifest[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        archive.addfile(archive.gettarinfo(str(path), arcname=name), io.BytesIO(content))
with gzip.open(target, 'rb') as stream:
    while stream.read(1024 * 1024):
        pass
with tarfile.open(target, 'r:gz') as archive:
    for member in archive:
        content = archive.extractfile(member).read()
        assert len(content) == manifest[member.name]['bytes']
        assert hashlib.sha256(content).hexdigest() == manifest[member.name]['sha256']
(HERE / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
receipt = dict(actual_exit_code=0, completed=time.time(), archive=target.name,
    bytes=target.stat().st_size, sha256=digest(target), members=len(manifest),
    gzip_CRC=True, all_member_SHA_and_length=True, source_before_after_stat_exact=True,
    official_and_existing_visual_sources_unchanged=True,
    diagnostic_checkpoints_remote_only=True, formal_training=False, SR=False)
(HERE / 'evidence_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
