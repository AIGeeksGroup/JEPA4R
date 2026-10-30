"""G12 recovery namespace selector; reuse frozen v2 packing and copy verification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

V2_SHA = 'bd4a240d9a39d0d7140f873de62ce3f760651ae2223598a68882f680f30e69d4'
RECOVERY_SHA = 'e62099b13b11e5b81d3034a13187e531e7aadd708f69223f81bacc4bcf9f6e30'
BINDING_SHA = 'bb47c70f0ccb998e786703e6febb4975b6438ba8da57f064c691d712ddcde693'
HERE = Path(__file__).resolve().parent


def load_v2(path=None):
    path = Path(path or HERE.parent / 'closed_backup_preparation_v2/closed_role.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V2_SHA
    spec = importlib.util.spec_from_file_location('_frozen_Quad_backup_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Never run its v2 CLI or G11 selector.
    return module


base = load_v2()


def locations(q):
    return dict(config=q/'functional_v2/configs/G12_cube4_pcd.v3.json',
        role=q/'control_v2/G12', run=q/'runs/G12_cube4_pcd',
        report=q/'acceptance/G12_full_acceptance.json',
        prepared=q/'functional_v2/G12_CONFIG_PREPARED_v3.json',
        binding=q/'control_v2/RECOVERY_BINDING_final_v2.json',
        release=q/'control_v2/G12_wait_arm_v2/RELEASE_v2.json',
        dispatcher=q/'control_v2/after_cache_G12_runtime_v2',
        pcd=q.parent/'cube4_official_embeddings_v1/pcd_training_camera_v1')


def verify_recovery(prepared, receipt, dispatcher, phase_exits, code_texts):
    assert prepared['actual_exit_code'] == receipt['actual_exit_code'] == 0
    assert prepared['PCD_recovery_chain_actual0'] is True
    assert receipt['publisher_sha256'] == RECOVERY_SHA and receipt['binding_sha256'] == BINDING_SHA
    assert receipt['cube_count'] == 4 and receipt['data_id'] == 'cube-quadruple-play-v0'
    assert (receipt['PCD_episodes'], receipt['PCD_rows']) == (5500, 5505500)
    chain = receipt['chain']
    assert chain['original_runtime_v1_actual_exit_code'] == 1 and chain['original_GPU_gate_accepted'] is False
    assert chain['logical_completed_phases'] == 8 and chain['new_runtime_completed_phases'] == 4
    assert dispatcher['actual_exit_code'] == 0 and dispatcher['stage'] == 'complete'
    assert len(phase_exits) == 3 and all(x['actual_exit_code'] == 0 for x in phase_exits)
    assert code_texts == ['0', '0', '1', '1', '1', '1']


def select(q, group, u, out):
    """Only namespace/provenance changes; original closure and final-only ledger checks."""
    assert group == 'G12'
    at = locations(q)
    config, role, run, report = (at[k] for k in ('config', 'role', 'run', 'report'))
    cfg, accepted, status = u.read(config), u.read(report), u.read(run/'status.json')
    assert Path(cfg['output']) == run
    phases = [role/'runtime'/f'{i:02d}_{action}_G12' for i, action in enumerate(('gate', 'reload', 'train', 'accept'), 1)]
    exits = [u.read(p/'exit.json') for p in phases]
    base.verify_metadata(cfg, accepted, status, u.read(role/'runtime/status.json'), exits,
        [p.read_text().strip() for p in (role/'runtime/outer_exit_code', role/'wrapper_exit_code')])
    for p in phases:
        assert (p/'actual_exit_code').read_text().strip() == '0'
    assert (run/'trainer_exit_code').read_text().strip() == '0'
    resolved = u.read(run/'resolved.json')
    assert resolved['binding']['config'] == cfg and resolved['resumed_from'] is None and resolved['initial_counter'] == 0
    assert resolved['initial_network_step'] == 1 and resolved['initial_adam_count'] == 0
    assert u.read(run/'final_state_fingerprint.json') == accepted['state']

    pcd, dispatch = at['pcd'], at['dispatcher']
    receipt_path = pcd/'recovery_receipt_v2.json'
    prepared, receipt = u.read(at['prepared']), u.read(receipt_path)
    publisher_phases = [dispatch/n for n in ('00_publish_recovery', '01_prepare_G12', '02_run_G12')]
    historical = [pcd/'runtime_v1/outer_exit_code', q/'control_v1/PCD/wrapper_exit_code',
                  q/'control_v1/after_cache_G12_runtime_v1/outer_exit_code', q/'control_v1/after_cache_G12_arm_v1/wrapper_exit_code']
    verify_recovery(prepared, receipt, u.read(dispatch/'status.json'), [u.read(p/'exit.json') for p in publisher_phases],
        [p.read_text().strip() for p in (dispatch/'outer_exit_code', q/'control_v2/G12_wait_arm_v2/wrapper_exit_code', *historical)])
    assert prepared['config_path'] == str(config) and prepared['config_sha256'] == u.sha(config)
    assert prepared['metadata_input_files'][str(receipt_path)]['sha256'] == u.sha(receipt_path)
    assert receipt['binding_path'] == str(at['binding']) and u.sha(at['binding']) == BINDING_SHA
    plan, release = u.read(role/'launch_plan.json'), u.read(at['release'])
    assert plan['groups'] == ['G12'] and len(plan['phases']) == 4
    for phase in plan['phases']:
        assert phase['argv'][phase['argv'].index('--config')+1] == str(config)
    assert plan['activation_lineage']['prepared_config_sha256'] == u.sha(at['prepared'])
    assert plan['activation_lineage']['PCD_closure_path'] == str(receipt_path)
    assert plan['activation_lineage']['PCD_closure_sha256'] == u.sha(receipt_path)
    assert release['CLI'][release['CLI'].index('--binding')+1] == str(at['binding'])
    assert release['CLI'][release['CLI'].index('--binding-sha256')+1] == BINDING_SHA

    paths = {config, report, role/'launch_plan.json', role/'runtime/status.json', role/'runtime/outer_exit_code',
             role/'wrapper_exit_code', at['prepared'], receipt_path, at['binding'], at['release'],
             dispatch/'status.json', dispatch/'outer_exit_code', Path(__file__), Path(base.__file__), Path(u.__file__), *historical}
    paths.update(run/n for n in ('train.csv', 'resolved.json', 'status.json', 'trainer_exit_code', 'final_state_fingerprint.json', 'final_val_probe.npz'))
    for p in phases:
        paths.update(p/n for n in ('actual_argv.json', 'run.log', 'resources.jsonl', 'actual_exit_code', 'exit.json'))
    for p in publisher_phases:
        paths.update(p/n for n in ('actual_argv.json', 'run.log', 'actual_exit_code', 'exit.json'))
    paths.update(q/rel for rel in ('control_v1/run_role.py', 'control_v2/run_G12.sh',
        'control_v2/after_cache_G12.py', 'control_v2/G12_wait_arm_v2/run.sh',
        'control_v2/G12_wait_arm_v2/wrapper_exit_code', 'control_v2/G12_wait_arm_v2/dispatch_receipt.json',
        'control_v2/G12_target_CPU_deploy_v2.json',
        'functional_v2/recovery.py', 'functional_v2/RECOVERY_SOURCE_FREEZE_v1.json', 'functional_v2/RECOVERY_HANDOFF_v1.json',
        'functional_v1/SOURCE_FREEZE_v1.json', 'functional_v1/HANDOFF_v1.json',
        'closed_backup_preparation_v3/SOURCE_FREEZE_v1.json', 'closed_backup_preparation_v3/HANDOFF_v1.json'))
    paths.update(pcd/rel for rel in ('runtime_v1/status.json', '3d/GPU_preprocess_report.json', 'performance_release_v2/release_decision.json'))
    for p, expected in {**resolved['binding']['code'], **release['source_pins']}.items():
        assert u.sha(p) == expected
        paths.add(Path(p))

    ledger = []
    for entry in accepted['checkpoint_transactions']:
        counter = entry['counter']; cp = run/'checkpoints'/f'update_{counter:09d}'; manifest = cp/'manifest.json'
        assert u.sha(manifest) == entry['manifest_sha256']
        item = u.read(manifest); assert item['counter'] == counter
        assert (cp/f'params_{counter}.pkl').is_file() and (cp/'run_state.pkl').is_file()
        paths.add(manifest)
        ledger.append(dict(counter=counter, manifest=str(manifest), manifest_sha256=entry['manifest_sha256'],
            policy=str(cp/f'params_{counter}.pkl'), policy_sha256=item['agent_sha256'],
            sidecar=str(cp/'run_state.pkl'), sidecar_sha256=item['sidecar_sha256'],
            policy_stat=u.stat(cp/f'params_{counter}.pkl'), sidecar_stat=u.stat(cp/'run_state.pkl'), included=counter == 1000000))
        if counter == 1000000:
            assert str(cp) == accepted['final_checkpoint'] and u.sha(manifest) == accepted['final_manifest_sha256']
            assert u.sha(cp/f'params_{counter}.pkl') == item['agent_sha256'] and u.sha(cp/'run_state.pkl') == item['sidecar_sha256']
            paths.update((cp/f'params_{counter}.pkl', cp/'run_state.pkl'))
    u.new('checkpoint_SHA_ledger.json', dict(actual_exit_code=0, group_id=group, checkpoints=ledger,
        only_final_policy_and_sidecar_included=True, all11_policies_and_sidecars_retained_remote=True))
    u.new('closure_readback.json', dict(actual_exit_code=0, group_id=group, CPU_acceptance_sha256=u.sha(report),
        cfg_sha256=u.sha(config), four_phase_exits=exits, reference_IO_source_sha256=base.REFERENCE_SHA,
        final_model_sha256=ledger[-1]['policy_sha256'], full153_state_exact=True, Adam_count=1000000,
        network_step=1000001, CSV_validation_records=200, checkpoint_transactions=11,
        actual_current_dim=211, actual_goal_dim=204, no_NN_or_training_or_SR=True,
        selector='recovery_functional_v2_config_v3_control_v2', original_failed_exits_retained=True,
        recovery_receipt_sha256=u.sha(receipt_path), generic_v2_pack_sha256=V2_SHA))
    paths.update((out/'checkpoint_SHA_ledger.json', out/'closure_readback.json'))
    return run, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['G12'], default='G12')
    parser.add_argument('--q', type=Path, default=base.Q)
    parser.add_argument('--reference', type=Path, default=base.REFERENCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--build-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); args.q = args.q.resolve()
    args.output = (args.output or args.q/'closed_training_backup_v2/G12/completed_v1').resolve()
    assert args.q == base.Q.resolve() and args.output == args.q/'closed_training_backup_v2/G12/completed_v1'
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True, group='G12', output=str(args.output),
            config=str(locations(args.q)['config']), pack_source_sha256=V2_SHA)))
        return 0
    if args.build_child:
        code = 1
        try:
            base.select = select  # Explicit selector argument via the imported module; packing is unchanged.
            base.build(args); code = 0
        finally:
            with (args.output/'builder_actual_exit_code').open('x') as stream: stream.write(str(code)+'\n')
        return code
    args.output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--group', 'G12', '--q', str(args.q),
            '--reference', str(args.reference), '--output', str(args.output), '--build-child']
    with (args.output/'builder_actual_argv.json').open('x') as stream:
        json.dump(dict(argv=argv, started=time.time(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            generic_pack_source_sha256=V2_SHA), stream, indent=2)
    with (args.output/'builder_stdout.log').open('xb') as stdout, (args.output/'builder_stderr.log').open('xb') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr)
    with (args.output/'builder_outer_exit_code').open('x') as stream: stream.write(str(result.returncode)+'\n')
    if result.returncode == 0: print((args.output/'receipt.json').read_text())
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
