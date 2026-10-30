"""Closed 3D cache metadata only; reuse the accepted SR archive primitives."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import time

D = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
H = D / 'official_state_3d_cache_v1'
A, C = H / 'acceleration_v1', H / 'cache'
R = A / 'resume_v1/runtime'
HERE = H / 'closed_metadata_backup_20261018_v1'
HELPER = D / 'official_state_last3_v1/build_last3_evidence.py'
HELPER_SHA = 'e6402da6df5471352023381fd0f288fc473d170c8068ab297a531961f4f8f6a1'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location('_closed_metadata_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
need, st, read, rb, sha, write_new = (helper.need, helper.source_stat,
    helper.read_json, helper.read_bytes, helper.sha, helper.write_new)


def closure():
    ready = read(C / 'READY.json')
    manifest, seal = read(C / 'cache_manifest.json'), read(C / 'cache_seal.json')
    feature, runtime = read(Path(ready['feature_report_path'])), read(Path(ready['runtime_report_path']))
    for name in ('full_exit.json', 'independent_audit_exit.json', 'trainer_exit.json'):
        value = read(R / name)
        need(value['actual_exit_code'] == 0 and value['residual_members'] == [], 'Closed child: ' + name)
    for path in (A / 'resume_v1/supervisor_actual_exit_code', R / 'pipeline_actual_exit_code',
                 R / 'trainer_actual_exit_code', R / 'audit_v1/features/actual_exit_code',
                 R / 'audit_v1/runtime_audit_actual_exit_code'):
        need(path.read_text().strip() == '0', 'Actual exit file: ' + str(path))
    need(Path(ready['outer_exit_file_required']) == A / 'resume_v1/supervisor_actual_exit_code', 'READY outer pointer')
    need(ready['ready_for_downstream'] and ready['actual_exit_code'] == 0 and
         ready['features'] == 3303300 and ready['latent_dim'] == 192 and ready['dtype'] == 'float32', 'READY coverage')
    need(feature['actual_exit_code'] == runtime['actual_exit_code'] == 0 and
         feature['episodes_verified'] == 3300 and feature['raw_rows_verified'] == 3303300 and
         feature['all_new_latent_SHA_shape_dtype_finite'] and feature['tracked_stats_stable'], 'Audited coverage')
    need(runtime['legacy_base_workers2_episodes'] == 199 and runtime['resource_profile_episodes'] == 3101 and
         runtime['all_episode_receipt_SHA_and_execution_profile_bindings'] and
         runtime['actual_renderer_workers'] == 16 and runtime['actual_prefetch_jobs'] == 32, 'Mixed profile audit')
    bindings = (('feature_report_sha256', Path(ready['feature_report_path'])),
                ('runtime_report_sha256', Path(ready['runtime_report_path'])),
                ('manifest_sha256', C / 'cache_manifest.json'), ('producer_seal_sha256', C / 'cache_seal.json'),
                ('runtime_profiles_manifest_sha256', C / 'runtime_profiles_manifest.json'))
    for key, path in bindings:
        need(ready[key] == sha(rb(path)), 'READY binding: ' + key)
    need(feature['manifest_sha256'] == seal['manifest_sha256'] == ready['manifest_sha256'] and
         feature['producer_seal_sha256'] == ready['producer_seal_sha256'], 'Report/seal manifest binding')
    need({s: len(v) for s, v in manifest['splits'].items()} == {'train': 3000, 'val': 300}, 'Split coverage')
    return ready, manifest, feature, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-signal', required=True)
    args = parser.parse_args()
    need(args.closed_signal.strip(), 'Explicit closed signal')
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    out = HERE / 'remote_package_v1'
    need(not out.exists(), 'Fresh output only')
    ready, manifest, feature, runtime = closure()
    plan, paths = read(H / 'producer_plan.json'), {}

    def add(path):
        path = Path(path)
        need(path.suffix not in {'.npy', '.npz', '.pkl', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.gz'},
             'Refuse raw/vector/model/archive contents: ' + str(path))
        st(path)
        paths[path.relative_to(D).as_posix()] = path

    # All historical source/gate/benchmark/runtime text is closed; current Root monitors are separate.
    roots = ('acceleration_v1', 'full_v1', 'preflight_v1', 'gate_v1', 'reports', 'versions')
    allowed = {'.py', '.json', '.jsonl', '.log', '.diff', '.md', '.sh', '.txt'}
    for directory in roots:
        for path in sorted((H / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and (
                path.suffix in allowed or path.name.endswith('exit_code')):
                add(path)
    for path in H.iterdir():
        if path.is_file() and path.suffix in allowed and not path.name.startswith(('root_status', 'monitor_handoff')):
            add(path)
    for name in ('report.json', 'actual_exit_code'):
        add(H / 'instance_restart_20261018_v1/postrestart_integrity_v1' / name)
    for name in ('READY.json', 'cache_manifest.json', 'cache_seal.json', 'protocol.json',
                 'runtime_profiles_manifest.json', 'pilot_receipt.json', 'status.json'):
        add(C / name)
    for split, count in (('train', 3000), ('val', 300)):
        entries, bound = manifest['splits'][split], feature['receipt_bindings'][split]
        need(len(bound) == count, 'Audit binding count')
        for episode, (item, audited) in enumerate(zip(entries, bound)):
            path = C / split / f'episode_{episode:06d}/receipt.json'
            content = rb(path)
            value = json.loads(content)
            need(Path(item['receipt_path']) == path and item['receipt_sha256'] == audited['receipt_sha256'] == sha(content),
                 'Receipt/manifest/auditor SHA binding')
            need(value['split'] == split and value['episode'] == episode and
                 value['raw_row_start'] == episode * 1001 and value['raw_row_count'] == 1001 and
                 value['terminal_raw_id'] == episode * 1001 + 1000 and
                 value['latent']['shape'] == [1001, 192] and value['latent']['dtype'] == 'float32' and
                 value['latent']['sha256'] == audited['latent_sha256'], 'Canonical receipt descriptor')
            add(path)
    for key in ('data_acceptance', 'source_stat_freeze'):
        need(sha(rb(Path(plan[key]))) == plan[key + '_sha256'], 'Data metadata binding')
        add(plan[key])
    for name, digest in plan['source_pins'].items():
        need(sha(rb(D / name)) == digest, 'Base source pin: ' + name)
        add(D / name)
    launch = read(A / 'resume_v1/launch_freeze.json')
    for name, entry in launch['files'].items():
        need(sha(rb(D / name)) == entry['sha256'], 'Final launch source pin: ' + name)
        add(D / name)
    arrays = Path(plan['shared_state_root']) / 'source_arrays.json'
    need(sha(rb(arrays)) == feature['source_arrays_receipt_sha256'], 'Source arrays receipt binding')
    add(arrays)
    add(Path(plan['shared_state_root']) / 'extraction_ownership.json')
    for path in (Path(__file__), HELPER):
        add(path)
    paths = dict(sorted(paths.items()))
    before = {name: st(path) for name, path in paths.items()}
    out.mkdir(exist_ok=False)
    policy = dict(metadata_only=True, latent_contents_read=False, raw_NPZ_contents_read=False,
        source_array_contents_read=False, model_contents_read=False, old_2D_or_SR_package_included=False,
        feature_vectors_remain_in_approved_remote_Temp=True, full_LOCAL_vectors_backup_claimed=False,
        accepted_feature_report_SHA=ready['feature_report_sha256'], accepted_runtime_report_SHA=ready['runtime_report_sha256'],
        checkpoint_reference=dict(path=plan['checkpoint'], sha256=plan['checkpoint_sha256'],
                                  accepted_stat=feature['checkpoint_stat']),
        legacy_199_profile_preserved=True, prior_failed_and_controlled_stop_exit_codes_preserved=True,
        helper_sha256=HELPER_SHA, closed_signal=args.closed_signal)
    write_new(out / 'package_policy.json', policy)
    paths['_backup/package_policy.json'] = out / 'package_policy.json'
    before['_backup/package_policy.json'] = st(out / 'package_policy.json')
    target = out / 'closed_3D_cache_metadata_v1.tar.gz'
    payload = {}
    with target.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
        for name, path in paths.items():
            need(st(path) == before[name], 'Source changed before read: ' + name)
            content = rb(path)
            payload[name] = dict(bytes=len(content), sha256=sha(content), source_stat=before[name])
            archive.addfile(archive.gettarinfo(str(path), arcname=name), io.BytesIO(content))
            need(st(path) == before[name], 'Source changed during read: ' + name)
        content = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
        info = tarfile.TarInfo('_backup/evidence_manifest.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    need(all(st(p) == before[n] for n, p in paths.items()), 'Final metadata source changed')
    closure()
    with gzip.open(target, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024**2), b''):
            pass
    expected = dict(payload)
    expected['_backup/evidence_manifest.json'] = dict(bytes=len(content), sha256=sha(content))
    seen = set()
    with tarfile.open(target, 'r:gz') as archive:
        for member in archive:
            need(member.isfile() and member.name in expected and member.name not in seen, 'Unexpected archive member')
            value, item = archive.extractfile(member).read(), expected[member.name]
            need(len(value) == member.size == item['bytes'] and sha(value) == item['sha256'], 'Member SHA/length')
            seen.add(member.name)
    need(seen == set(expected) and all(st(p) == before[n] for n, p in paths.items()), 'Final member/source check')
    receipt = dict(actual_exit_code=0, completed=time.time(), archive=target.name, bytes=target.stat().st_size,
        sha256=helper.digest_file(target), members=len(seen), manifest_member='_backup/evidence_manifest.json',
        manifest_sha256=sha(content), gzip_CRC_EOF=True, all_member_SHA_length_verified=True,
        source_before_after_stat_exact=True, episode_receipts=3300, feature_rows_bound=3303300,
        feature_dimension=192, feature_dtype='float32', metadata_only=True,
        latent_raw_array_model_contents_read=False, accepted_feature_report_sha256=ready['feature_report_sha256'],
        accepted_runtime_report_sha256=ready['runtime_report_sha256'],
        actual_builder_exit_requires_external_wait=True, local_backup_requires_actual_copy_and_full_verification=True)
    write_new(out / 'evidence_receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
