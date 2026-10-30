"""Closed 2D metadata only; reuse verified SR packer primitives, never read vectors."""
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
H = D / 'official_state_2d_cache_v1'
C, R, A = H / 'cache', H / 'runtime', H / 'aux_audit_recovery_v2'
HERE = H / 'aux_metadata_backup_v1'
HELPER = D / 'official_state_last3_v1/build_last3_evidence.py'
HELPER_SHA = 'e6402da6df5471352023381fd0f288fc473d170c8068ab297a531961f4f8f6a1'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location('_closed_metadata_pack_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
need, st, read, read_bytes, sha, write_new = (helper.need, helper.source_stat,
    helper.read_json, helper.read_bytes, helper.sha, helper.write_new)


def closure():
    report_path = A / 'runtime/audit/report.json'
    report, manifest, ready = read(report_path), read(C / 'cache_manifest.json'), read(C / 'READY.json')
    full = read(R / 'full_exit.json')
    need(full['actual_exit_code'] == 0 and full['residual_members'] == [], 'Full producer exit')
    need((H / 'supervisor_actual_exit_code').read_text().strip() == '1', 'Preserve original supervisor failure')
    need((A / 'recovery_actual_exit_code').read_text().strip() == '0', 'Recovery outer exit')
    need((A / 'runtime/audit/actual_exit_code').read_text().strip() == '0', 'Audit exit file')
    need(read(A / 'runtime/audit_exit.json')['actual_exit_code'] == report['actual_exit_code'] == 0, 'Actual audit exit')
    need(report['episodes_verified'] == 3300 and report['raw_rows_verified'] == 3303300 and
         report['latent_dim'] == 192 and report['dtype'] == 'float32', 'Audit dimensions/coverage')
    need(report['all_new_latent_SHA_shape_dtype_finite'] and report['tracked_stats_stable'], 'Feature audit scope')
    need({k: len(v) for k, v in manifest['splits'].items()} == {'train': 3000, 'val': 300}, 'Manifest coverage')
    need(ready['ready_for_downstream'] and ready['features'] == 3303300, 'READY coverage')
    need(ready['audit_report_sha256'] == sha(read_bytes(report_path)) and
         ready['manifest_sha256'] == report['manifest_sha256'] == sha(read_bytes(C / 'cache_manifest.json')) and
         ready['producer_seal_sha256'] == report['producer_seal_sha256'] == sha(read_bytes(C / 'cache_seal.json')),
         'READY/report/manifest/seal bindings')
    need(ready['recovery_source_freeze_sha256'] == sha(read_bytes(A / 'freeze.json')) ==
         'fd277b287a54c7eb4962698abee395a4465609ae0ad26c6ebead3447d0a67ec1', 'Recovery freeze')
    need(Path(ready['outer_exit_file_required']) == A / 'recovery_actual_exit_code', 'Actual READY outer pointer')
    freeze = read(A / 'freeze.json')
    for name, digest in freeze['pins'].items():
        need(sha(read_bytes(H / name)) == digest, 'Recovery pin: ' + name)
    return report, manifest, ready, full


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--closed-signal', required=True)
    args = p.parse_args()
    need(args.closed_signal.strip(), 'Explicit closed signal')
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    out = HERE / 'closed_metadata_v1'
    need(not out.exists(), 'Fresh output only; never overwrite')
    report, manifest, ready, full = closure()
    plan, paths = read(H / 'producer_plan.json'), {}

    def add(path, required=True):
        path = Path(path)
        if not path.exists() and not required:
            return
        need(path.suffix not in {'.npy', '.npz', '.pkl', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.gz'},
             'Refuse feature/raw/model/archive payload: ' + str(path))
        st(path)
        paths[path.relative_to(D).as_posix()] = path

    for name in ('produce.py', 'render_adapter.py', 'producer_plan.json', 'audit_cache.py', 'AUDIT_FREEZE.json',
                 'source_freeze_v1.json', 'source_freeze_v2.json', 'launch_plan.json', 'launch_freeze.json',
                 'relay_cache.py', 'supervise_cache.py', 'run_cache.sh', 'run_producer_bounded.py',
                 'supervisor_actual_exit_code', 'negative_branch_v2.diff'):
        add(H / name)
    for name in ('README.md', 'ROOT_PLAN.md', 'PRODUCER_REVIEW.json', 'AUDIT_README.md',
                 'CPU_exit.json', 'CPU_stdout.log', 'CPU_stderr.log', 'nonfinite_v2_exit.json',
                 'nonfinite_v2_stderr.log', 'AUDIT_TEST_GREEN.json', 'AUDIT_TEST_GREEN.log',
                 'AUDIT_TEST_RED.json', 'dispatch_receipt.json', 'dispatch.log', 'ROOT_AFTERFULL_FAILURE_READBACK.json'):
        add(H / name, False)
    for name in ('launch_plan.json', 'launch_freeze.json', 'actual_argv.json', 'status.json',
                 'runtime_check.log', 'runtime_check_exit.json', 'pilot.log', 'pilot_exit.json', 'pilot_argv.json',
                 'full.log', 'full_exit.json', 'full_argv.json', 'resources.jsonl', 'trainer_exit.json',
                 'trainer_actual_exit_code', 'pipeline_actual_exit_code', 'pipeline_status.json', 'pipeline_failure.json',
                 'independent_audit.log', 'independent_audit_exit.json', 'supervisor_failure.json',
                 'cache_audit_v1/failure.json', 'cache_audit_v1/status.json', 'cache_audit_v1/actual_exit_code'):
        add(R / name)
    for name in ('audit_gate_metadata.py', 'recover_closed_cache.py', 'freeze.json', 'static_review.json',
                 'test_real_gate.py', 'recovery_actual_exit_code', 'runtime/actual_argv.json',
                 'runtime/audit_exit.json', 'runtime/audit.log', 'runtime/closure.json',
                 'runtime/audit/report.json', 'runtime/audit/status.json', 'runtime/audit/actual_exit_code',
                 'runtime/audit/progress.jsonl'):
        add(A / name)
    for name in ('Root_actual_audit_readback.json', 'dispatch_receipt.json', 'dispatch.log'):
        add(A / name, False)
    for name in ('protocol.json', 'preprocess_selection.json', 'pilot_receipt.json',
                 'pilot_missing_control_mocap_witness.json', 'cache_manifest.json', 'cache_seal.json', 'READY.json'):
        add(C / name)
    for split, count in (('train', 3000), ('val', 300)):
        entries = manifest['splits'][split]
        bound = report['receipt_bindings'][split]
        need(len(bound) == count and [x['episode'] for x in entries] == list(range(count)), 'EP order')
        for episode, (item, audited) in enumerate(zip(entries, bound)):
            path = C / split / f'episode_{episode:06d}/receipt.json'
            content = read_bytes(path)
            need(Path(item['receipt_path']) == path and item['receipt_sha256'] == audited['receipt_sha256'] == sha(content),
                 'Actual receipt/manifest/audit SHA binding')
            value = json.loads(content)
            need(value['episode'] == episode and value['split'] == split and value['raw_row_start'] == episode * 1001 and
                 value['raw_row_count'] == 1001 and value['terminal_raw_id'] == episode * 1001 + 1000,
                 'Receipt canonical raw IDs')
            need(value['latent']['shape'] == [1001, 192] and value['latent']['dtype'] == 'float32' and
                 value['latent']['sha256'] == audited['latent_sha256'], 'Bound feature descriptor')
            add(path)
    for key in ('data_acceptance', 'source_stat_freeze'):
        need(sha(read_bytes(Path(plan[key]))) == plan[key + '_sha256'], 'Data metadata identity')
        add(plan[key])
    for name, digest in plan['source_pins'].items():
        need(sha(read_bytes(D / name)) == digest, 'Source pin: ' + name)
        add(D / name)
    arrays_path = Path(plan['shared_state_root']) / 'source_arrays.json'
    need(sha(read_bytes(arrays_path)) == report['source_arrays_receipt_sha256'], 'Source-array receipt identity')
    add(arrays_path)
    add(Path(plan['shared_state_root']) / 'extraction_ownership.json')
    for path in (Path(__file__), HERE / 'CLOSED_READBACK.json', HELPER):
        add(path)
    paths = dict(sorted(paths.items()))
    before = {name: st(path) for name, path in paths.items()}
    out.mkdir(exist_ok=False)
    write_new(out / 'package_policy.json', dict(metadata_only=True, new_latent_files_read=False,
        raw_NPZ_read_or_hashed=False, source_array_NPY_read=False, model_contents_read=False,
        old_SR_package_included=False, expected_episode_receipts=3300,
        original_supervisor_failure_preserved=True, successful_recovery_outer_exit=0,
        feature_vectors_remain_in_approved_remote_Temp=True, final_LOCAL_vectors_backup_claimed=False,
        helper_sha256=HELPER_SHA, closed_signal=args.closed_signal))
    paths['_backup/package_policy.json'] = out / 'package_policy.json'
    before['_backup/package_policy.json'] = st(out / 'package_policy.json')
    target = out / 'closed_2D_cache_metadata_v1.tar.gz'
    payload = {}
    with target.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
        for name, path in paths.items():
            need(st(path) == before[name], 'Source changed before read: ' + name)
            content = read_bytes(path)
            payload[name] = dict(bytes=len(content), sha256=sha(content), source_stat=before[name])
            archive.addfile(archive.gettarinfo(str(path), arcname=name), io.BytesIO(content))
            need(st(path) == before[name], 'Source changed during read: ' + name)
        manifest_bytes = (json.dumps(payload, indent=2, allow_nan=False) + chr(10)).encode()
        info = tarfile.TarInfo('_backup/evidence_manifest.json');info.size=len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    need(all(st(path) == before[name] for name, path in paths.items()), 'Metadata changed during packaging')
    closure()  # Only small metadata; no latent/array/model contents.
    with gzip.open(target, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024**2), b''):
            pass
    expected = dict(payload)
    expected['_backup/evidence_manifest.json'] = dict(bytes=len(manifest_bytes), sha256=sha(manifest_bytes))
    seen = set()
    with tarfile.open(target, 'r:gz') as archive:
        for member in archive:
            need(member.isfile() and member.name in expected and member.name not in seen, 'Unexpected tar member')
            content = archive.extractfile(member).read();item=expected[member.name]
            need(len(content) == member.size == item['bytes'] and sha(content) == item['sha256'], 'Member SHA/size')
            seen.add(member.name)
    need(seen == set(expected) and all(st(path) == before[name] for name, path in paths.items()), 'Final source/member set')
    receipt = dict(actual_exit_code=0, completed=time.time(), archive=target.name, bytes=target.stat().st_size,
        sha256=helper.digest_file(target), members=len(seen), manifest_member='_backup/evidence_manifest.json',
        manifest_sha256=sha(manifest_bytes), gzip_CRC_EOF=True, all_member_SHA_length_verified=True,
        source_before_after_stat_exact=True, episode_receipts=3300, feature_rows_bound=3303300,
        feature_dimension=192, feature_dtype='float32', metadata_only=True, latent_files_read=False,
        raw_or_source_array_or_model_contents_read=False, old_SR_archive_copied=False,
        original_supervisor_exit=1, successful_recovery_outer_exit=0,
        actual_builder_exit_requires_external_wait=True, local_backup_requires_real_SCP_and_project_verification=True)
    write_new(out / 'evidence_receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
