"""Unpack one CortexBench demo zip ($STABLEWM_HOME/datasets/cortexbench/downloads/<name>.zip, URLs in README.md) into
$STABLEWM_HOME/datasets/cortexbench/<name>/, taking each <name>/*.pickle once: the MetaWorld zip (and
maybe others) also holds a second, identical copy under checkpoint/maksymets/eaif/datasets/mujoco_vil_datasets/.
Files are written to _partial_<name>/ next to it and the directory is renamed into place at the end, so a BC job
never sees a half-written pickle; each written file is fsynced and dropped from the page cache (container cap).

    source env.sh && $PY unpack_suite.py dmc-expert-v1.0
"""
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(os.environ['JEPA4R_DEMOS_HOME'])  # data/policy/<env> (holds downloads/<name>.zip)
name = sys.argv[1]
z = HERE / 'downloads' / f'{name}.zip'
assert z.exists() and not (HERE / 'downloads' / f'{name}.zip.aria2').exists(), f'{z} missing or still downloading'
data = HERE
final, partial = data / name, data / f'_partial_{name}'
assert not final.exists(), f'{final} exists already'
shutil.rmtree(partial, ignore_errors=True)
(partial / name).mkdir(parents=True)
with zipfile.ZipFile(z) as zf:
    members = {}
    for m in zf.infolist():  # prefer the top-level <name>/ copy; fall back to any .../<name>/<file>
        parts = m.filename.rstrip('/').split('/')
        if m.is_dir() or len(parts) < 2 or parts[-2] != name:
            continue
        if parts[-1] not in members or len(parts) == 2:
            members[parts[-1]] = m
    print(time.strftime('%F_%T'), f'{len(members)} files from {len(zf.infolist())} members:',
          ', '.join(f'{k} ({m.file_size / 2**30:.2f} GiB)' for k, m in sorted(members.items())), flush=True)
    for base, m in sorted(members.items()):
        out = partial / name / base
        with zf.open(m) as src, open(out, 'wb') as dst:  # zf.open checks the CRC at the end of the read
            shutil.copyfileobj(src, dst, 64 << 20)
            dst.flush()
            os.fsync(dst.fileno())
            os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        print(time.strftime('%F_%T'), 'wrote', base, flush=True)
os.rename(partial / name, final)
partial.rmdir()
print(time.strftime('%F_%T'), 'unpacked', name, '->', final, flush=True)
