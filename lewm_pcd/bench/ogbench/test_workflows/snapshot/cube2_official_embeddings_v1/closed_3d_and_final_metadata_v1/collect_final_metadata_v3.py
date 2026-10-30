"""Closed3D/final evidence only; execution requires an explicit post-closure flag."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import tarfile
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube2_official_embeddings_v1'
OUT = CASE / 'closed_3d_and_final_metadata_v1'
PHASES = ('released_profile_prepare', '3d_released_pilot',
          '3d_released_full', '3d_independent_audit')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def j(path):
    return json.loads(Path(path).read_text())


def rawstat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def accepted_report(mode):
    ready = j(CASE / mode / 'cache/READY.json')
    expected = CASE / mode / ('independent_audit' if mode == '2d' else 'independent_audit_v2') / 'report.json'
    path = Path(ready['audit_report'])
    assert path == expected and path.resolve() == expected
    assert ready['audit_sha256'] == sha(path)
    return path


def closure():
    assert (CASE / 'wrapper_v4_exit_code').read_text().strip() == '0'
    assert (CASE / 'runtime_v4/outer_exit_code').read_text().strip() == '0'
    assert j(CASE / 'runtime_v4/closure.json')['actual_exit_code'] == 0
    r = j(CASE / 'runtime_v4/audit_exit.json')
    assert r['actual_exit_code'] == 0 and not r['residual_owned_members']
    for name in ('2d_pilot', '2d_full', '2d_independent_audit',
                 '3d_reference_gate', '3d_fresh_gate', '3d_gpu_preprocessing_gate'):
        r = j(CASE / 'runtime_v2' / (name + '_exit.json'))
        assert r['actual_exit_code'] == 0 and not r['residual_owned_members']
    for name in PHASES[:-1]:
        r = j(CASE / 'runtime_v3' / (name + '_exit.json'))
        assert r['actual_exit_code'] == 0 and not r['residual_owned_members']
    old = j(CASE / 'runtime_v3/3d_independent_audit_exit.json')
    assert old['actual_exit_code'] == 1 and not old['residual_owned_members']
    assert (CASE / 'wrapper_v3_exit_code').read_text().strip() == '1'
    assert (CASE / 'runtime_v3/outer_exit_code').read_text().strip() == '1'
    for mode in ('2d', '3d'):
        ready = j(CASE / mode / 'cache/READY.json')
        assert ready['actual_exit_code'] == 0 and ready['ready_for_downstream']
        assert ready['cube_count'] == 2 and ready['features'] == 1101100
        report_path = accepted_report(mode)
        assert (report_path.parent / 'actual_exit_code').read_text().strip() == '0'
        report = j(report_path)
        assert report['actual_exit_code'] == 0 and report['episodes_verified'] == 1100
        assert report['raw_rows_verified'] == 1101100 and report['main_rows_verified'] == 1100000
        assert report['latent_dim'] == 192 and report['dtype'] == 'float32'
        assert ready['manifest_sha256'] == sha(CASE / mode / 'cache/cache_manifest.json')
    both = j(CASE / 'BOTH_CACHES_READY.json')
    assert both['actual_exit_code'] == 0 and both['features_per_modality'] == 1101100
    assert set(both['modes']) == {'2d', '3d'} and both['cube_count'] == 2
    original = j(CASE / '3d/GPU_preprocess_report.json')
    assert original['accepted'] is False and original['actual_exit_code'] == 0
    assert original['prepared_bitexact'] and original['latent_bitexact']
    assert (CASE / 'wrapper_v2_exit_code').read_text().strip() == '1'
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-root-closure', action='store_true')
    args = parser.parse_args()
    if not args.execute_after_root_closure:
        print(json.dumps(dict(plan_only=True, executed=False, no_remote_or_GPU_calls=True)))
        return
    original = closure()
    assert not (OUT / 'cube2_3d_final_closed_metadata.tar.gz').exists()
    allow = j(OUT / 'allowlist_v3.json')
    paths = {CASE / rel for rel in allow['fixed_case_files']}
    planpath = CASE / 'workspace/official_state_3d_cache_v1/producer_plan.json'
    plan = j(planpath)
    reportpath = accepted_report('3d')
    audit = j(reportpath)
    assert audit['actual_exit_code'] == 0 and audit['episodes_verified'] == 1100
    assert audit['raw_rows_verified'] == 1101100 and audit['main_rows_verified'] == 1100000
    assert audit['all_new_latent_SHA_shape_dtype_finite']
    assert audit['all_canonical_ID_and_terminal_next_contracts']
    assert audit['manifest_sha256'] == sha(CASE / '3d/cache/cache_manifest.json')
    assert audit['producer_seal_sha256'] == sha(CASE / '3d/cache/cache_seal.json')
    references = {}
    for pins, base in [(plan['source_pins'], CASE / 'workspace'),
                       (j(CASE / 'control_v3/launch_plan.json')['source_pins'], ROOT)]:
        for rel, expected in pins.items():
            path = Path(rel) if Path(rel).is_absolute() else base / rel
            if path.suffix in ('.npy', '.npz', '.pt', '.ckpt', '.h5', '.hdf5', '.pth'):
                references[str(path)] = expected
                continue
            assert sha(path) == expected
            paths.add(path)
    manifest = j(CASE / '3d/cache/cache_manifest.json')
    count = 0
    for split, n in [('train', 1000), ('val', 100)]:
        entries = manifest['splits'][split]
        assert len(entries) == n and [x['episode'] for x in entries] == list(range(n))
        bindings = audit['receipt_bindings'][split]
        assert len(bindings) == n
        for entry, binding in zip(entries, bindings):
            assert entry['episode'] == binding['episode']
            receipt = Path(entry['receipt_path'])
            assert receipt == CASE / '3d/cache' / split / f"episode_{entry['episode']:06d}" / 'receipt.json'
            assert sha(receipt) == entry['receipt_sha256'] == binding['receipt_sha256']
            paths.add(receipt)
            count += 1
    assert count == 1100
    prior = j(CASE / 'closed_2d_metadata_v1/receipt.json')
    assert prior['sha256'] == allow['prior_2D_backup_external_reference']['sha256']
    assert prior['bytes'] == 544890 and prior['members'] == 1144
    files = []
    for path in sorted(paths):
        assert path.resolve().is_relative_to(ROOT) and path.is_file()
        assert path.suffix not in allow['forbidden_suffixes']
        s = rawstat(path)
        h = sha(path)
        assert rawstat(path) == s
        files.append(dict(path=str(path), member=str(path.relative_to(ROOT)),
                          bytes=s['size'], sha256=h, before=s))
    proof = dict(actual_exit_code=0, files=files, original_model_SHA_references_only=references,
                 prior_2D_backup_external_reference=allow['prior_2D_backup_external_reference'],
                 episodes=1100, raw_rows=1101100, source_before_after_required=True,
                 model_or_vector_contents_read=False, original_failed_5percent_gate_preserved=True,
                 original_report_SHA=sha(CASE / '3d/GPU_preprocess_report.json'),
                 original_report_accepted=original['accepted'], helper_sha256=sha(__file__))
    write_new(OUT / 'member_manifest.json', proof)
    archive = OUT / 'cube2_3d_final_closed_metadata.tar.gz'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', dereference=True) as tar:
            for item in files:
                tar.add(item['path'], arcname=item['member'], recursive=False)
            tar.add(OUT / 'member_manifest.json', arcname='member_manifest.json')
    for item in files:
        assert rawstat(Path(item['path'])) == item['before']
        assert sha(item['path']) == item['sha256']
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1024 * 1024):
            pass
    expected = {x['member']: x for x in files}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files) + 1
        for member in members:
            assert member.isfile()
            b = tar.extractfile(member).read()
            if member.name == 'member_manifest.json':
                assert hashlib.sha256(b).hexdigest() == sha(OUT / 'member_manifest.json')
            else:
                item = expected[member.name]
                assert len(b) == item['bytes'] and hashlib.sha256(b).hexdigest() == item['sha256']
    record = dict(actual_exit_code=0, archive_path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), members=len(files) + 1, source_files=len(files),
        member_manifest_sha256=sha(OUT / 'member_manifest.json'),
        source_before_after_SHA_stat_exact=True, all_member_SHA_size_exact=True,
        gzip_CRC_full_decompression=True, pair_complete_READY_and_wrapper0=True,
        original_failed_performance_report_preserved=True, prior_2D_archive_not_copied=True,
        original_raw_vector_model_contents_not_read=True, completed=time.time())
    write_new(OUT / 'receipt.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
