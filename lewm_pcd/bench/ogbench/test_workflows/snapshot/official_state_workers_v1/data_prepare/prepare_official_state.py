"""Prepare unmodified official 3Cube NPZ files; CPU-only, bounded Range buffers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import zipfile

CASE = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
ROOT = CASE / 'official_state_workers_v1/data_prepare'
DATA = CASE / 'data/official_state'
SOURCE = CASE / 'source/ogbench'
COMMIT = 'a977253ce5b88434f4e4f531c50047cc95993a51'
EXPECTED_UTILS_SHA = 'f503a3a0497f91ce02958abae5770f3e08c9d6a3a740687c828c996fedb24e9b'
SPECS = [('cube-triple-play-v0', 1003544255), ('cube-triple-play-v0-val', 100613267)]
URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench/'
CHUNK = 1024 ** 2
BUFFER = 256 * 1024
WORKERS = 8


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(BUFFER), b''):
            h.update(b)
    return h.hexdigest()


def resources():
    return {'memory_current': int(Path('/sys/fs/cgroup/memory.current').read_text()),
            'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
            'memory_events': Path('/sys/fs/cgroup/memory.events').read_text(),
            'disk_free': os.statvfs(DATA).f_bavail * os.statvfs(DATA).f_frsize}


STARTED = time.time()
STATE = {'stage': 'initializing', 'started': STARTED, 'actual_exit_code': None,
         'completed_files': [], 'failed_count': 0}


def status(**values):
    STATE.update(values)
    STATE['updated'] = time.time()
    STATE['resources'] = resources()
    save(ROOT / 'status.json', STATE)


def download(name, size):
    url = URL + name + '.npz'
    target = DATA / (name + '.npz')
    receipt_path = ROOT / 'receipts' / (name + '.download.json')
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=120) as r:
        headers = dict(r.headers.items())
        size_now = int(r.headers['Content-Length'])
        etag = r.headers['ETag']
        assert r.status == 200 and size_now == size and r.geturl() == url
    save(ROOT / 'headers' / (name + '.json'), {'url': url, 'headers': headers, 'observed': time.time()})
    if target.exists():
        old = json.loads(receipt_path.read_text())
        assert old['url'] == url and old['etag'] == etag
        assert target.stat().st_size == size and sha(target) == old['sha256']
        print('REUSE_VERIFIED', name, flush=True)
        return old
    chunks_dir = ROOT / 'chunks' / name
    chunks_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = chunks_dir / 'origin.json'
    origin = {'url': url, 'size': size, 'etag': etag, 'chunk_bytes': CHUNK}
    if ledger_path.exists():
        assert json.loads(ledger_path.read_text()) == origin, 'Origin changed; refuse stale resume'
    else:
        save(ledger_path, origin)
    completed = 0
    last_bytes, last_time = 0, time.time()
    def one(start):
        end = min(size, start + CHUNK) - 1
        expected = end - start + 1
        part = chunks_dir / (str(start) + '.part')
        rec = chunks_dir / (str(start) + '.json')
        if rec.exists() and part.exists():
            p = json.loads(rec.read_text())
            assert p['etag'] == etag and p['start'] == start and p['end'] == end
            assert part.stat().st_size == expected and sha(part) == p['sha256']
            return p
        errors = []
        for attempt in range(6):
            tmp = chunks_dir / (str(start) + '.incomplete')
            try:
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={start}-{end}', 'If-Match': etag, 'Accept-Encoding': 'identity'})
                h, count = hashlib.sha256(), 0
                with urllib.request.urlopen(request, timeout=120) as r, tmp.open('wb') as f:
                    assert r.status == 206
                    assert r.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    assert r.headers['ETag'] == etag
                    for b in iter(lambda: r.read(BUFFER), b''):
                        count += len(b)
                        assert count <= expected
                        f.write(b)
                        h.update(b)
                    f.flush()
                    os.fsync(f.fileno())
                assert count == expected
                os.replace(tmp, part)
                result = {'start': start, 'end': end, 'bytes': count, 'etag': etag,
                          'sha256': h.hexdigest(), 'attempt': attempt + 1,
                          'content_range': f'bytes {start}-{end}/{size}', 'errors_before_success': errors}
                save(rec, result)
                return result
            except Exception as e:
                errors.append(repr(e))
                if attempt == 5:
                    raise RuntimeError(f'{name} range {start}-{end}: {errors}') from e
                time.sleep(min(2 ** attempt, 16))
    started = time.time()
    status(stage='downloading', dataset=name, expected_bytes=size, completed_bytes=0,
           streams=WORKERS, buffer_bytes_per_stream=BUFFER, started_file=started)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, x) for x in range(0, size, CHUNK)]
        for job in as_completed(futures):
            completed += job.result()['bytes']
            now = time.time()
            speed = (completed - last_bytes) / max(now - last_time, 0.001)
            if now - last_time >= 15 or completed == size:
                status(completed_bytes=completed, bytes_per_sec_window=speed,
                       eta_seconds=(size - completed) / speed if speed else None)
                print('DOWNLOAD', name, completed, '/', size, 'window_Bps', round(speed), flush=True)
                last_bytes, last_time = completed, now
    assert completed == size
    status(stage='assembling', completed_bytes=completed)
    assembled = DATA / (name + '.npz.assembling')
    with assembled.open('wb') as dst:
        for start in range(0, size, CHUNK):
            part = chunks_dir / (str(start) + '.part')
            rec = json.loads((chunks_dir / (str(start) + '.json')).read_text())
            assert sha(part) == rec['sha256']
            with part.open('rb') as src:
                for b in iter(lambda: src.read(BUFFER), b''):
                    dst.write(b)
        dst.flush()
        os.fsync(dst.fileno())
    assert assembled.stat().st_size == size
    status(stage='full_sha_zipcrc')
    digest = sha(assembled)
    with zipfile.ZipFile(assembled) as z:
        assert z.testzip() is None
        members = [{'name': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size,
                    'crc32': x.CRC} for x in z.infolist()]
    assert not target.exists(), 'Refuse overwrite of existing final NPZ'
    os.link(assembled, target)
    assembled.unlink()
    rec = {'url': url, 'path': str(target), 'bytes': size, 'etag': etag, 'sha256': digest,
           'headers': headers, 'all_ranges_verified': True, 'all_zip_crc_verified': True,
           'zip_members': members, 'completed': time.time(), 'actual_exit_code': 0}
    save(receipt_path, rec)
    print('RAW_ACCEPTED', name, size, digest, flush=True)
    return rec


def schema_and_loader(name, download_receipt):
    import numpy as np
    sys.path.insert(0, str(SOURCE))
    from ogbench.utils import load_dataset
    path = DATA / (name + '.npz')
    status(stage='full_schema_finite', dataset=name)
    fields = {}
    with np.load(path, allow_pickle=False) as z:
        assert {'observations', 'actions', 'terminals', 'qpos', 'qvel'} <= set(z.files)
        rows = None
        for key in z.files:
            a = z[key]
            assert a.dtype.kind in 'buif', (key, str(a.dtype))
            rows = len(a) if rows is None else rows
            assert len(a) == rows
            for start in range(0, len(a), 65536):
                assert np.isfinite(a[start:start + 65536]).all(), (key, start)
            fields[key] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                           'finite_all': True, 'min': float(a.min()), 'max': float(a.max())}
            del a
        raw_t = z['terminals'].astype(np.float32)
    assert raw_t.ndim == 1 and raw_t[-1] == 1 and np.isin(raw_t, [0, 1]).all()
    ends = np.flatnonzero(raw_t)
    lengths = np.diff(np.concatenate([[-1], ends]))
    assert lengths.min() >= 2
    status(stage='official_compact_loader', rows=rows)
    compact = load_dataset(str(path), compact_dataset=True, add_info=True)
    assert set(compact) == set(fields) | {'valids'}
    assert np.array_equal(compact['valids'], 1.0 - raw_t)
    expected_t = np.minimum(raw_t + np.concatenate([raw_t[1:], [1.0]]), 1.0).astype(np.float32)
    assert np.array_equal(compact['terminals'], expected_t)
    assert int(compact['valids'].sum()) == rows - len(ends)
    assert compact['valids'][-1] == 0 and 'next_observations' not in compact
    for key, value in compact.items():
        assert len(value) == rows
        for start in range(0, rows, 65536):
            assert np.isfinite(value[start:start + 65536]).all()
    with np.load(path, allow_pickle=False) as z:
        for key in fields:
            if key != 'terminals':
                assert np.array_equal(compact[key], z[key])
    rec = {'path': str(path), 'raw_sha256': download_receipt['sha256'], 'raw_bytes': path.stat().st_size,
           'fields': fields, 'rows': rows, 'episodes': len(ends),
           'episode_lengths': {'min': int(lengths.min()), 'max': int(lengths.max()),
                               'unique': np.unique(lengths).tolist()},
           'raw_last_terminal': float(raw_t[-1]), 'compact_valid_rows': int(compact['valids'].sum()),
           'compact_terminal_count': int(compact['terminals'].sum()),
           'all_raw_fields_finite': True, 'all_compact_fields_finite': True,
           'compact_rows_exactly_match_raw': True, 'compact_terminal_semantics_exact': True,
           'official_loader': 'ogbench.utils.load_dataset(compact_dataset=True, add_info=True)',
           'official_commit_from_migration_provenance': COMMIT, 'utils_sha256': sha(SOURCE / 'ogbench/utils.py'),
           'numpy_version': np.__version__, 'completed': time.time(), 'actual_exit_code': 0}
    save(ROOT / 'receipts' / (name + '.schema_loader.json'), rec)
    del compact, raw_t, expected_t
    gc.collect()
    print('SCHEMA_LOADER_ACCEPTED', name, rows, len(ends), flush=True)
    return rec


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    assert sha(SOURCE / 'ogbench/utils.py') == EXPECTED_UTILS_SHA
    save(ROOT / 'actual_argv.json', {'argv': [sys.executable] + sys.argv,
         'pid': os.getpid(), 'started': STARTED, 'cwd': os.getcwd(),
         'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'OMP_NUM_THREADS']},
         'producer_sha256': sha(__file__), 'source_utils_sha256': EXPECTED_UTILS_SHA,
         'config': {'specs': SPECS, 'workers': WORKERS, 'chunk_bytes': CHUNK, 'buffer_bytes': BUFFER}})
    try:
        accepted = []
        for name, size in SPECS:
            raw = download(name, size)
            schema = schema_and_loader(name, raw)
            accepted.append({'download': raw, 'schema_loader': schema})
            status(completed_files=[x['download']['path'] for x in accepted])
        save(ROOT / 'ACCEPTANCE.json', {'stage': 'complete', 'actual_exit_code': 0,
             'started': STARTED, 'completed': time.time(), 'datasets': accepted,
             'producer_sha256': sha(__file__), 'resources': resources(),
             'scope': 'official state raw train+val preparation; no training/SR/GPU computation'})
        status(stage='complete', actual_exit_code=0)
    except BaseException as e:
        status(stage='failed', actual_exit_code=1, failed_count=1, error=repr(e), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
