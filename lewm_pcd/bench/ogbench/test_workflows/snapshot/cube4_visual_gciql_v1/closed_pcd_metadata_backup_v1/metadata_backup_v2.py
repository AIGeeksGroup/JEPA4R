"""Closed Quad PCD metadata only. Default plan; no cache payload or active G12 reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from parameter_bridge import load, ASSET, ARCHIVE

base = load('pack')
Q = base.Q
PCD = base.ROOT/'cube4_official_embeddings_v1/pcd_training_camera_v1'
MODEL = '95e7980f665a0ad888a0723499f5d0771c8cf797065b1f1b39c3eae39fd951d8'
RECOVERY_SHA = 'e62099b13b11e5b81d3034a13187e531e7aadd708f69223f81bacc4bcf9f6e30'
BINDING_SHA = 'bb47c70f0ccb998e786703e6febb4975b6438ba8da57f064c691d712ddcde693'
OLD = ('00_wait_G11_acceptance', '01_reference', '02_fresh', '03_GPU_preprocess')
NEW = ('04_profile', '05_pilot', '06_full', '07_audit')
SUFFIXES = {'', '.json', '.py', '.log', '.jsonl', '.sh', '.yaml', '.yml', '.txt', '.xml', '.toml', '.diff'}


def verify_metadata(manifest, seal, audit, receipt, exits, codes):
    assert all(x['actual_exit_code'] == 0 for x in (manifest, seal, audit, receipt))
    assert len(exits) == 8 and all(x['actual_exit_code'] == 0 and x['residual_owned_members'] == [] for x in exits)
    assert codes == ['1', '1', '0', '0', '0']  # Original runtime/wrapper, recovered runtime/wrapper/audit.
    assert receipt['cube_count'] == 4 and receipt['data_id'] == 'cube-quadruple-play-v0'
    assert (receipt['PCD_episodes'], receipt['PCD_rows']) == (5500, 5505500)
    assert receipt['chain']['original_runtime_v1_actual_exit_code'] == 1
    assert receipt['chain']['original_GPU_gate_accepted'] is False and receipt['chain']['logical_completed_phases'] == 8
    assert manifest['checkpoint_sha256'] == audit['checkpoint_sha256'] == MODEL
    assert (audit['episodes_verified'], audit['raw_rows_verified'], audit['latent_dim'], audit['dtype']) == (5500, 5505500, 192, 'float32')
    assert audit['all_new_latent_SHA_shape_dtype_finite'] and audit['all_canonical_ID_and_terminal_next_contracts']
    assert (seal['train_rows'], seal['val_rows'], seal['full_raw_ID_coverage']) == (5005000, 500500, True)
    assert [len(manifest['splits'][split]) for split in ('train', 'val')] == [5000, 500]


def closed_plan_phases(version, launch):
    wanted = OLD if version == 1 else NEW
    phases = [phase for phase in launch['phases'] if phase['name'] in wanted]
    assert [phase['name'] for phase in phases] == list(wanted)
    return phases


def select(q, _group, u, out):
    pcd, cache = PCD, PCD/'3d/cache'
    plan_path = pcd/'workspace/official_state_3d_cache_v1/producer_plan.json'
    manifest_path, seal_path = cache/'cache_manifest.json', cache/'cache_seal.json'
    audit_path = pcd/'3d/independent_audit/report.json'
    receipt_path = pcd/'recovery_receipt_v2.json'
    manifest, seal, audit, receipt = (u.read(p) for p in (manifest_path, seal_path, audit_path, receipt_path))
    exits = [u.read(pcd/f'runtime_v{version}'/f'{name}_exit.json') for version, names in ((1, OLD), (2, NEW)) for name in names]
    code_paths = [pcd/'runtime_v1/outer_exit_code', q/'control_v1/PCD/wrapper_exit_code',
                  pcd/'runtime_v2/outer_exit_code', q/'control_v2/PCD/wrapper_exit_code', pcd/'3d/independent_audit/actual_exit_code']
    verify_metadata(manifest, seal, audit, receipt, exits, [p.read_text().strip() for p in code_paths])
    assert audit['manifest_sha256'] == seal['manifest_sha256'] == u.sha(manifest_path)
    assert audit['producer_seal_sha256'] == u.sha(seal_path) and audit['plan_sha256'] == u.sha(plan_path)
    assert audit['protocol_sha256'] == manifest['protocol_sha256'] == seal['protocol_sha256']
    assert u.read(cache/'protocol.json') == manifest['protocol']
    assert audit['protocol_file_sha256'] == u.sha(cache/'protocol.json')
    recovery_path = q/'functional_v2/recovery.py'
    assert u.sha(recovery_path) == RECOVERY_SHA
    spec = importlib.util.spec_from_file_location('_frozen_PCD_recovery_metadata', recovery_path)
    recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)
    binding_path = q/'control_v2/RECOVERY_BINDING_final_v2.json'
    binding = recovery.load_binding(q, pcd, binding_path, BINDING_SHA, {})
    # Reuse the already frozen metadata closure chain; never call its prepare/dispatcher.
    recovery.checked_receipt(q, pcd, binding, binding_path, BINDING_SHA,
        lambda target, expected_sha=None, json_value=True: recovery.stable(target, {}, expected_sha, json_value))
    paths = {manifest_path, seal_path, audit_path, receipt_path, plan_path, binding_path, recovery_path,
             cache/'protocol.json', cache/'pilot_receipt.json', cache/'status.json', cache/'runtime_profiles_manifest.json',
             pcd/'3d/independent_audit/status.json', *code_paths, Path(__file__), Path(base.__file__), Path(u.__file__),
             Path(base.__file__).with_name('copy_verified.py'),
             Path(__file__).with_name('parameter_bridge.py'), Path(__file__).with_name('copy_metadata.py'),
             Path(__file__).with_name('SOURCE_FREEZE_v2.json'), Path(__file__).with_name('HANDOFF_v2.json')}
    ledger = []
    for split, count in (('train', 5000), ('val', 500)):
        entries, accepted = manifest['splits'][split], audit['receipt_bindings'][split]
        assert len(accepted) == count and [x['episode'] for x in entries] == list(range(count))
        for item, accepted_item in zip(entries, accepted):
            ep = item['episode']; path = cache/split/f'episode_{ep:06d}/receipt.json'
            assert Path(item['receipt_path']) == path and u.sha(path) == item['receipt_sha256'] == accepted_item['receipt_sha256']
            record = u.read(path)
            assert record['actual_exit_code'] == 0 and record['status'] == 'episode_committed'
            assert (record['split'], record['episode'], record['raw_row_start'], record['raw_row_count'], record['main_rows'], record['terminal_raw_id']) == (split, ep, ep*1001, 1001, 1000, ep*1001+1000)
            assert record['checkpoint_sha256'] == MODEL and record['protocol_sha256'] == manifest['protocol_sha256']
            assert record['latent']['shape'] == [1001, 192] and record['latent']['dtype'] == 'float32'
            assert record['latent']['sha256'] == accepted_item['latent_sha256']
            paths.add(path)
            ledger.append(dict(split=split, episode=ep, receipt=str(path), receipt_sha256=item['receipt_sha256'],
                declared_latent_sha256=record['latent']['sha256'], latent_payload_included=False))
    for version, names in ((1, OLD), (2, NEW)):
        folder = pcd/f'runtime_v{version}'
        paths.update(folder/n for n in ('status.json', 'actual_dispatch.json', 'resources.jsonl', 'outer_exit_code'))
        for name in names: paths.update(folder/(name+suffix) for suffix in ('_argv.json', '_exit.json', '.log'))
    paths.update((pcd/'runtime_v1/failure.json', pcd/'runtime_v2/closure.json'))
    for version in (1, 2):
        control = q/f'control_v{version}/PCD'
        launch = u.read(control/'launch_plan.json')
        paths.update(control/n for n in ('launch_plan.json', 'wrapper_exit_code'))
        for path, digest in launch['source_pins'].items():
            assert Path(path).suffix in SUFFIXES and u.sha(path) == digest
            paths.add(Path(path))
        for phase in closed_plan_phases(version, launch):
            for check in phase.get('checks', []):
                target = Path(check['path'])
                assert target.suffix in SUFFIXES
                paths.add(target)
            argv = phase['argv']
            if '--runtime-dir' in argv:
                folder = Path(argv[argv.index('--runtime-dir')+1])
                paths.update(folder/n for n in ('actual_launch.json', 'effective_source.py'))
    paths.update(q/rel for rel in ('control_v1/PCD/release_freeze.json', 'control_v2/PCD/RELEASE_v2.json',
        'control_v2/PCD/dispatch_receipt.json', 'functional_v2/RECOVERY_SOURCE_FREEZE_v1.json',
        'functional_v2/RECOVERY_HANDOFF_v1.json', 'control_v2/G12_wait_arm_v2/RELEASE_v2.json',
        'control_v2/G12_wait_arm_v2/dispatch_receipt.json'))
    paths.add(Path(manifest['source_arrays_receipt']['path']))  # JSON receipt only, never qpos/qvel NPY.
    for path in paths:
        assert path.suffix in SUFFIXES and path.stat().st_size <= 32*1024*1024
        assert '/runs/' not in path.as_posix() and '/control_v2/G12/runtime/' not in path.as_posix()
        assert '__pycache__' not in path.parts
    u.new('PCD_receipt_SHA_ledger.json', dict(actual_exit_code=0, asset_kind=ASSET, episodes=5500, raw_rows=5505500,
        receipts=ledger, only_metadata_payload=True, vector_SHA_scope='original_completed_independent_audit; no new vector reads'))
    u.new('closure_readback.json', dict(actual_exit_code=0, asset_kind=ASSET, logical_completed_phases=8,
        actual_old_four_exits=exits[:4], actual_new_four_exits=exits[4:], old_overall_failure_retained=True,
        source_manifest_sha256=u.sha(manifest_path), source_seal_sha256=u.sha(seal_path), full_audit_sha256=u.sha(audit_path),
        recovery_receipt_sha256=u.sha(receipt_path), receipt_count=5500, raw_rows=5505500, checkpoint_sha256=MODEL,
        only_metadata_no_vector_raw_WM_model_training_or_SR=True, parameter_bridge=base.parameter_bridge_proof))
    paths.update((out/'PCD_receipt_SHA_ledger.json', out/'closure_readback.json'))
    return pcd, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--build-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.q, args.group, args.reference = Q, ASSET, base.REFERENCE
    args.output = Q/'closed_pcd_metadata_backup_v1/completed_v1'
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True, asset_kind=ASSET, archive=ARCHIVE, output=str(args.output), bridge=base.parameter_bridge_proof)))
        return 0
    if args.build_child:
        code = 1
        try:
            base.select = select; base.build(args); code = 0
        finally:
            with (args.output/'builder_actual_exit_code').open('x') as stream: stream.write(str(code)+'\n')
        return code
    args.output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--build-child']
    with (args.output/'builder_actual_argv.json').open('x') as stream:
        json.dump(dict(argv=argv, started=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), parameter_bridge=base.parameter_bridge_proof), stream, indent=2)
    with (args.output/'builder_stdout.log').open('xb') as stdout, (args.output/'builder_stderr.log').open('xb') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr)
    with (args.output/'builder_outer_exit_code').open('x') as stream: stream.write(str(result.returncode)+'\n')
    if result.returncode == 0: print((args.output/'receipt.json').read_text())
    return result.returncode


if __name__ == '__main__': raise SystemExit(main())
