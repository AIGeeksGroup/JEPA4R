"""Package only closed Cube2 2D metadata; never read vectors/raw/model weights."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube2_official_embeddings_v1'
OUT = CASE / 'closed_2d_metadata_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rawstat(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def j(path):
    return json.loads(path.read_text())


def newjson(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    plan_path = CASE / 'workspace/official_state_2d_cache_v1/producer_plan.json'
    plan = j(plan_path)
    cache = CASE / '2d/cache'
    audit = j(CASE / '2d/independent_audit/report.json')
    assert audit['actual_exit_code'] == 0
    assert (audit['episodes_verified'], audit['raw_rows_verified'], audit['main_rows_verified']) == (1100, 1101100, 1100000)
    assert audit['all_new_latent_SHA_shape_dtype_finite'] and audit['all_raw_ID_and_true_terminal_next_contracts']
    assert audit['tracked_stats_stable'] and audit['plan_sha256'] == sha(plan_path)
    assert audit['manifest_sha256'] == sha(cache / 'cache_manifest.json')
    assert audit['producer_seal_sha256'] == sha(cache / 'cache_seal.json')
    paths = [plan_path, CASE / 'data_acceptance.json', CASE / 'stat-freeze.json',
             CASE / 'source_arrays/source_arrays.json',
             ROOT / 'checkpoints/cube2/origin_manifest.json']
    for rel in ['protocol.json', 'preprocess_selection.json', 'pilot_receipt.json',
                'pilot_missing_control_mocap_witness.json', 'cache_manifest.json',
                'cache_seal.json', 'status.json']:
        paths.append(cache / rel)
    for rel in ['status.json', 'progress.jsonl', 'report.json', 'actual_exit_code']:
        paths.append(CASE / '2d/independent_audit' / rel)
    for phase in ['2d_pilot', '2d_full', '2d_independent_audit']:
        exit_ = j(CASE / 'runtime_v2' / (phase + '_exit.json'))
        assert exit_['actual_exit_code'] == 0 and exit_['residual_owned_members'] == []
        for suffix in ['.log', '_argv.json', '_exit.json']:
            paths.append(CASE / 'runtime_v2' / (phase + suffix))
    for rel, expected in plan['source_pins'].items():
        path = CASE / 'workspace' / rel
        assert sha(path) == expected
        paths.append(path)
    receipt_count = 0
    for split, count in [('train', 1000), ('val', 100)]:
        rows = audit['receipt_bindings'][split]
        assert len(rows) == count and [x['episode'] for x in rows] == list(range(count))
        for item in rows:
            path = cache / split / f"episode_{item['episode']:06d}" / 'receipt.json'
            assert sha(path) == item['receipt_sha256']
            assert item['raw_row_start'] == item['episode'] * 1001
            assert item['terminal_raw_id'] == item['episode'] * 1001 + 1000
            paths.append(path)
            receipt_count += 1
    paths = sorted(set(paths))
    ledger = []
    for path in paths:
        assert path.exists() and path.is_file() and path.suffix not in ['.npy', '.npz', '.pt', '.ckpt']
        assert path.resolve().is_relative_to(ROOT)
        before = rawstat(path)
        h = sha(path)
        assert rawstat(path) == before
        ledger.append(dict(path=str(path), member=str(path.relative_to(ROOT)),
                           bytes=before['size'], sha256=h, before=before))
    member_manifest = dict(scope='closed Cube2 2D metadata only; full2D accepted, pair3D failure separate',
        files=ledger, episodes=1100, raw_rows=1101100, receipt_count=receipt_count,
        original_vectors_read=False, old_raw_read=False, checkpoint_contents_read=False,
        encoder_executed=False, READY_written=False, producer_source_sha256=sha(__file__))
    newjson(OUT / 'member_manifest.json', member_manifest)
    archive = OUT / 'cube2_2d_closed_metadata.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', dereference=True) as tar:
            for item in ledger:
                tar.add(item['path'], arcname=item['member'], recursive=False)
            tar.add(OUT / 'member_manifest.json', arcname='member_manifest.json')
    for item in ledger:
        path = Path(item['path'])
        assert rawstat(path) == item['before'] and sha(path) == item['sha256']
    expected = {x['member']: x for x in ledger}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(ledger) + 1
        for member in members:
            assert member.isfile()
            b = tar.extractfile(member).read()
            if member.name == 'member_manifest.json':
                assert hashlib.sha256(b).hexdigest() == sha(OUT / 'member_manifest.json')
            else:
                item = expected[member.name]
                assert len(b) == item['bytes'] and hashlib.sha256(b).hexdigest() == item['sha256']
    receipt = dict(actual_exit_code=0, archive_path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(ledger) + 1, source_files=len(ledger),
        member_manifest_sha256=sha(OUT / 'member_manifest.json'), source_before_after_SHA_stat_exact=True,
        all_member_SHA_size_exact=True, completed=time.time(), archive_read_all_members=True,
        vectors_or_original_NPZ_or_checkpoints_included=False, twoD_independent_acceptance=True,
        full_pair_complete=False, READY_not_written=True)
    newjson(OUT / 'receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
