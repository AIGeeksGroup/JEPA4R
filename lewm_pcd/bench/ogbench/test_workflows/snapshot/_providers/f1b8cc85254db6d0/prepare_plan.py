"""Derive recovery metadata from the exact prior recipe; never launches a process."""
import hashlib
import json
from pathlib import Path

case = Path(__file__).resolve().parent
reference = case.parent / 'double_cube_lejepa_match_3cube_20261026_v1'
plan = json.loads((reference / 'plan.json').read_text())
remote_case = '/root/autodl-tmp/workspace/users/lewm/coordination/' + case.name
remote_reference = '/root/autodl-tmp/workspace/users/lewm/coordination/' + reference.name
parent = plan['output_root'] + '/checkpoints/lejepa_ogb_double_front500_seed0_w24pf2_10ep/weights_epoch_7.pt'
name = 'lejepa_ogb_double_front500_seed0_w24pf2_resume7_to10'
plan.update(schema='double_cube_tiny_weight_only_recovery_v1',
            preparation_status='source_prepared_not_launched_root_gates_pending',
            reference_case=remote_reference,
            reference_freeze_sha256=hashlib.sha256((reference / 'freeze.json').read_bytes()).hexdigest(),
            output_root=remote_case + '/runtime_home', verified_export_root=remote_case + '/verified_exports',
            run_name=name,
            source_functional_diff_lines='7 exact main anchors; all other native main text reversible byte-for-byte')
plan['argv'] = [plan['argv'][0], '-B', remote_case + '/resume_entry.py', '--case', remote_case, '--',
                '--task', 'ogb_double_front500', '--dataset', plan['data']['train_vds'],
                '--name', name, '--workers', '24', '--epochs', '10',
                '--resume-model', parent, '--start-epoch', '7',
                '--verified-export-root', plan['verified_export_root']]
plan['environment']['STABLEWM_HOME'] = plan['output_root']
plan['environment']['PYTHONPATH'] = remote_reference + '/dependencies'
plan['recipe'].update(checkpoint_input=parent, optimizer_resume=False, exact_resume=False)
plan['recovery'] = {
    'mode': 'model_weights_only; explicitly accepted by user; not exact resume',
    'parent_model': parent,
    'parent_sha256': '0d7c48b1faa9ecbadba46c0ae2a8ca9662a7bae2768a4c03749250d94756fabb',
    'parent_bytes': 43663359, 'parent_tensor_count': 216, 'parent_floating_tensor_count': 214,
    'start_epoch': 7, 'total_epochs': 10, 'new_epochs': 3, 'batches_per_epoch': 19531,
    'scheduler_logical_position_before': 136717, 'expected_lr_before': .00125,
    'scheduler_method': 'replay original SequentialLR.step 136717 times without optimizer.step',
    'first_new_nominal_batch_one_based': 136718, 'last_nominal_batch_one_based': 195310,
    'adam': 'fresh AdamW; initial state empty; first actual update step1',
    'grad_scaler': 'fresh native enabled GradScaler; scaler state not restored',
    'rng': 'native torch.manual_seed(0) only; Python/NumPy new-process state; loader RNG rebuilt; no old RNG restored',
    'retained_model_lineage': 'parent valid export7 plus newly computed/exported epochs8,9,10',
    'historical_computation': 'old run really computed epochs8,9,10 but failed export storage; preserve old source/logs/zero-length exports',
    'nominal_counter_caveat': 'scheduler follows native nominal batches even if GradScaler skips; nominal batches are not successful Adam update counts'}
plan['storage'] = {
    'primary': 'instance disk runtime_home; temporary storage, external durable backup required',
    'stage': 'instance disk verified_exports; retain all new 8/9/10 weights even if publication/backup fails',
    'shared_mirror_enabled': False,
    'shared_mirror_reason': '2026-10-27 live readback shared200GiB still full; no shared write authorized here',
    'external_backup': 'pending Root/user-selected Windows or expanded shared location and independent receipt',
    'expected_three_exports_bytes': 130990077,
    'expected_storage_budget_bytes_per_copy_with_one_partial_and_small_logs': 200000000,
    'minimum_free_bytes_each_filesystem': 536870912,
    'publication': 'exclusive partial files; CRC/CPU torchload/three keys/finite/tensor bytes/SHA; atomic hardlink refuses overwrite; fresh primary readback before saved'}
plan['budget'].update(total_original_scheduler_budget=195310, nominal_new_batches=58593,
                      first_new_nominal_batch_one_based=136718, last_nominal_batch_one_based=195310)
plan['allowed_differences_from_active_reference'] += [
    'explicit default-off resume-model/start-epoch and local verified-export-root flags',
    'load parent backbone/head/proj strict and compare every tensor byte before first forward',
    'native scheduler-only position replay; rebuilt optimizer/scaler/RNG, not exact resume',
    'skip seven completed retained-lineage epochs; execute native range(7,10)',
    'first optimizer submission read-only hooks with fresh Adam step1 and logical scheduler136718 receipt',
    'stage and primary verified export; max-steps smoke additionally exports a clearly named smoke model',
    'primary instance output isolated from full shared filesystem; external backup remains separately pending']
plan['not_yet_executed'] = ['fresh target preflight by Root', 'real-data two-batch resumed GPU smoke',
                            'fresh native Encoder strict reload by Root', 'formal new three epochs',
                            'all new export acceptance', 'external durable backup acceptance']
plan['new_training_started'] = False
plan['new_GPU_forward_or_update_executed'] = False
with (case / 'plan.json').open('x') as stream:
    json.dump(plan, stream, indent=2)
    stream.write('\n')
