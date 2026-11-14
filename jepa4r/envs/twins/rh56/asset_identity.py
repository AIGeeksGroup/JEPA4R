"""Check the external native RH56 bundle against its file list before importing it."""
import hashlib
import json
from pathlib import Path,PurePosixPath


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def _relative(value):
    name=PurePosixPath(value)
    if not value or name.is_absolute() or '..' in name.parts or chr(92) in value or ':' in value:
        raise ValueError('bundle manifest paths must be safe relative POSIX paths')
    return name


def verify_rh56_bundle(bundle_root,*,manifest_path=None):
    manifest = Path(
        manifest_path or Path(__file__).resolve().parents[4] / 'configs/policy/bc/relocate/rh56_native_bundle_v1.json'
    ).resolve()
    payload=json.loads(manifest.read_text(encoding='utf-8'))
    files=payload['files']
    if not files or len(files)!=payload['file_count']:raise ValueError('bundle file list count mismatch')
    root=Path(bundle_root).expanduser().resolve();seen=set();checked=[]
    for row in files:
        relative=_relative(row['relative_path']);name=relative.as_posix()
        if name in seen:raise ValueError('duplicate trusted bundle member')
        seen.add(name);path=root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root):raise ValueError('bundle member escapes the selected root')
        if not path.is_file():raise FileNotFoundError(path)
        before=path.stat()
        if before.st_size!=row['bytes']:
            raise ValueError('RH56 bundle member size mismatch: '+name)
        digest=_sha(path)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('RH56 bundle member changed while verifying: '+name)
        checked.append(dict(relative_path=name,bytes=row['bytes'],sha256=digest))
    if not {'env.py','ppo.py','experts/relocate_expert.pt'}<=seen:
        raise ValueError('bundle file list lacks required native expert members')
    canonical=json.dumps(sorted(checked,key=lambda row:row['relative_path']),sort_keys=True,separators=(',',':'))
    return dict(passed=True,bundle_root=str(root),verified_files=len(checked),manifest_path=str(manifest),
        manifest_sha256=_sha(manifest),content_identity_sha256=hashlib.sha256(canonical.encode()).hexdigest())
