"""Publish P2 configs only from closed PLDM cache and completed final1M metadata."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from pldm_inputs import CHECKPOINT_SHA, DIM, FAMILY

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
GROUP = 'P2_cube2_pldm_rgb'
COUNTERS = [1] + list(range(100000, 1000001, 100000))
ASSET = ROOT/'assets/baselines/cube2/cube2_pldm_impala512_epoch10_20261021_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def read(path, records, digest=None):
    path = Path(path)
    before = stat(path)
    blob = path.read_bytes()
    assert stat(path) == before
    actual = hashlib.sha256(blob).hexdigest()
    assert digest is None or actual == digest, str(path)
    records[str(path)] = dict(sha256=actual, stat=before)
    return json.loads(blob)


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.pending_{os.getpid()}')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def source_pins(case):
    functional = Path(str(case)).joinpath('functional_v1')
    freeze = json.loads(functional.joinpath('SOURCE_FREEZE_V1.json').read_bytes())
    pins = {str(functional/name): digest for name, digest in freeze['files'].items()}
    for path, digest in pins.items():
        assert sha(path) == digest, path
    return pins


def check_cache_metadata(plan, manifest, seal, audit, selection):
    assert manifest['actual_exit_code'] == seal['actual_exit_code'] == audit['actual_exit_code'] == selection['actual_exit_code'] == 0
    assert plan['checkpoint_sha256'] == manifest['checkpoint_sha256'] == CHECKPOINT_SHA
    assert plan['split_episodes'] == {'train':1000, 'val':100} and plan['raw_rows_per_episode'] == 1001
    assert manifest['protocol']['encoder_family'] == FAMILY and manifest['protocol']['latent_dim'] == DIM
    assert (audit['episodes_verified'], audit['raw_rows_verified'], audit['main_rows_verified'], audit['latent_dim'], audit['dtype']) == (1100,1101100,1100000,DIM,'float32')
    assert audit['all_new_latent_SHA_shape_dtype_finite'] and audit['all_raw_ID_and_true_terminal_next_contracts'] and audit['tracked_stats_stable']
    assert audit['batch1_alignment_passed'] is True and len(audit['batch1_alignment_sha256']) == 64
    assert (seal['train_rows'], seal['val_rows'], seal['full_raw_ID_coverage']) == (1001000,100100,True)
    device = selection['selected_device']
    assert device in ('cpu', 'cuda') and manifest['protocol']['pixel_preprocess_device'] == device
    if device == 'cuda':
        assert selection['numeric_passed'] and selection['cuda_faster_by_required_gain']
    for split, count in plan['split_episodes'].items():
        assert [entry['episode'] for entry in manifest['splits'][split]] == list(range(count))


def closed_cache(case, records):
    home = case/'cache_v1'
    plan_path = home/'producer_plan.json'
    plan = read(plan_path, records)
    cache = home/'cache'
    assert Path(plan['cache_dir']).resolve() == cache.resolve()
    manifest_path, seal_path = cache/'cache_manifest.json', cache/'cache_seal.json'
    manifest, seal = read(manifest_path, records), read(seal_path, records)
    audit_path = home/'independent_audit/report.json'
    assert (audit_path.parent/'actual_exit_code').read_text().strip() == '0'
    audit = read(audit_path, records)
    selection_path = cache/'preprocess_selection.json'
    selection = read(selection_path, records)
    check_cache_metadata(plan, manifest, seal, audit, selection)
    assert audit['plan_sha256'] == records[str(plan_path)]['sha256']
    assert audit['manifest_sha256'] == seal['manifest_sha256'] == records[str(manifest_path)]['sha256']
    assert audit['producer_seal_sha256'] == records[str(seal_path)]['sha256']
    assert audit['protocol_sha256'] == seal['protocol_sha256'] == manifest['protocol_sha256']
    protocol = read(cache/'protocol.json', records)
    assert protocol == manifest['protocol']
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() == manifest['protocol_sha256']
    assert protocol['preprocessing_selection_sha256'] == records[str(selection_path)]['sha256']
    ref = manifest['source_arrays_receipt']
    arrays = read(ref['path'], records, ref['sha256'])
    assert Path(ref['path']).resolve() == (home/'source_arrays/source_arrays.json').resolve()
    assert ref['sha256'] == plan['source_arrays_view_sha256'] == audit['source_arrays_receipt_sha256']
    return dict(plan=plan, cache=str(cache), manifest_sha256=records[str(manifest_path)]['sha256'],
        manifest_path=str(manifest_path), seal_path=str(seal_path), audit_path=str(audit_path),
        selection_path=str(selection_path), selection_sha256=records[str(selection_path)]['sha256'],
        selected_device=selection['selected_device'], arrays=arrays, arrays_ref=ref)


def config_from_cache(template, facts, case):
    cfg = copy.deepcopy(template)
    assert cfg['executable'] is False and cfg['group_id'] == GROUP
    cfg.update(executable=True, status='closed_new_PLDM_cache_accepted_pending_real_GCIQL_gate', output=str(case/'runs'/GROUP))
    cfg.pop('disabled_reason', None)
    cfg['feature_loader_arguments'] = {}
    for split, count in facts['plan']['split_episodes'].items():
        source = facts['arrays']['splits'][split]
        bound = cfg['data_binding']['splits'][split]
        assert source['source_npz'] == bound['path'] and source['rows'] == count*1001
        bound['accepted_stat'] = source['accepted_stat']
        cfg['feature_loader_arguments'][split] = dict(cache_dir=facts['cache'], split=split,
            manifest_sha256=facts['manifest_sha256'], checkpoint_sha256=CHECKPOINT_SHA,
            source_npz=source['source_npz'], expected_rows=count*1001, encoder_family=FAMILY, feature_dim=DIM)
    plan = facts['plan']
    cfg['data_binding']['data_acceptance'] = dict(path=plan['data_acceptance'], sha256=plan['data_acceptance_sha256'])
    cfg['data_binding']['source_stat_freeze'] = dict(path=plan['source_stat_freeze'], sha256=plan['source_stat_freeze_sha256'])
    cfg['data_binding']['source_arrays_receipt'] = facts['arrays_ref']
    cfg['encoder_binding'].update(selected_preprocess_device=facts['selected_device'],
        preprocessing_selection=dict(path=facts['selection_path'], sha256=facts['selection_sha256']),
        source_pins={str(Path(plan['workspace'])/name):digest for name,digest in plan['source_pins'].items()})
    cfg['cache_binding'] = dict(cube_count=2, data_id='cube-double-play-v0', encoder_family=FAMILY,
        latent_dim=DIM, dtype='float32', layout='raw1001_rows_per_episode_single_npy',
        split_episodes=dict(train=1000,val=100), canonical_total_rows=1101100,
        manifest=dict(path=facts['manifest_path'],sha256=facts['manifest_sha256']),
        accepted_audit=dict(path=facts['audit_path']), checkpoint_sha256=CHECKPOINT_SHA,
        no_LeWM_feature_or_RGB_source_rebind=True)
    return cfg


def prepare_train(case):
    pins = source_pins(case)
    records = {}
    template = read(case/'functional_v1/configs/P2_cube2_pldm_rgb.TEMPLATE.json', records)
    facts = closed_cache(case, records)
    cfg = config_from_cache(template, facts, case)
    for split in ('train','val'):
        source = cfg['data_binding']['splits'][split]
        assert stat(source['path']) == source['accepted_stat']
    for path, item in records.items():
        assert stat(path) == item['stat']
    output = case/'functional_v1/configs/P2_cube2_pldm_rgb.v1.json'
    receipt = case/'functional_v1/TRAIN_CONFIG_PREPARED_v1.json'
    assert not output.exists() and not receipt.exists()
    publish(output, cfg)
    result = dict(actual_exit_code=0, group_id=GROUP, config_path=str(output), config_sha256=sha(output),
        current_dim=531, goal_dim=518, encoder_family=FAMILY, latent_dim=DIM,
        cache_episodes=1100, cache_raw_rows=1101100, selected_preprocess_device=facts['selected_device'],
        metadata_input_files=records, source_pins=pins, NN_or_training_started=False, completed=time.time())
    publish(receipt, result)
    return result


def prepare_native(case):
    pins = source_pins(case)
    records = {}
    config_path = case/'functional_v1/configs/P2_cube2_pldm_rgb.v1.json'
    cfg = read(config_path, records)
    prepared = read(case/'functional_v1/TRAIN_CONFIG_PREPARED_v1.json', records)
    assert prepared['actual_exit_code'] == 0 and prepared['config_sha256'] == sha(config_path)
    report_path = case/'acceptance/P2_cube2_pldm_rgb_full_acceptance.json'
    accepted = read(report_path, records)
    run = case/'runs'/GROUP
    status, resolved = read(run/'status.json', records), read(run/'resolved.json', records)
    assert (run/'trainer_exit_code').read_text().strip() == '0'
    assert status['exit_code'] == 0 and status['completed_updates'] == status['adam_count'] == 1000000
    assert status['network_step'] == 1000001 and status['validation_count'] == 200
    assert accepted['actual_exit_code'] == 0 and accepted['role'] == cfg['role']
    assert accepted['full153_state_exact'] and accepted['state']['leaves'] == 153 and accepted['finite_actions']
    assert (accepted['Adam_count'],accepted['network_step'],accepted['CSV_validation_records']) == (1000000,1000001,200)
    assert (accepted['actual_current_dim'],accepted['actual_goal_dim'],accepted['synthetic_zero_padding']) == (531,518,False)
    assert [entry['counter'] for entry in accepted['checkpoint_transactions']] == COUNTERS
    assert resolved['binding']['config'] == cfg and resolved['resumed_from'] is None and resolved['initial_counter'] == 0
    assert resolved['initial_network_step'] == 1 and resolved['initial_adam_count'] == 0
    assert read(run/'final_state_fingerprint.json', records) == accepted['state']
    for entry in accepted['checkpoint_transactions']:
        manifest = run/'checkpoints'/f"update_{entry['counter']:09d}"/'manifest.json'
        read(manifest, records, entry['manifest_sha256'])
    final = run/'checkpoints/update_001000000'
    assert accepted['final_checkpoint'] == str(final) and sha(final/'manifest.json') == accepted['final_manifest_sha256']
    gate = case/'gates'/GROUP/'next_batch.npz'
    group = dict(cube_count=2,state_dim=37,current_dim=531,goal_dim=518,max_episode_steps=500,mode='rgb',
        wm=str(ASSET/'weights/weights_epoch_10.pt'),wm_sha256=CHECKPOINT_SHA,
        training=str(config_path),training_sha256=sha(config_path),acceptance=str(report_path),acceptance_sha256=sha(report_path),
        gate=str(gate),gate_sha256=sha(gate),final=str(final),manifest_sha256=accepted['final_manifest_sha256'],
        encoder=cfg['encoder_binding'])
    pins.update(resolved['binding']['code'])
    pins.update(cfg['encoder_binding']['source_pins'])
    pins.update({path:item['sha256'] for path,item in records.items()})
    pins.update({str(gate):group['gate_sha256'],cfg['encoder_binding']['preprocessing_selection']['path']:cfg['encoder_binding']['preprocessing_selection']['sha256']})
    for path, digest in pins.items():
        assert sha(path) == digest, path
    output = case/'functional_v1/NATIVE_BINDING_v1.json'
    receipt = case/'functional_v1/NATIVE_PREPARED_v1.json'
    assert not output.exists() and not receipt.exists()
    binding = dict(actual_exit_code=0,group_id=GROUP,final_counter=1000000,group=group,source_pins=pins,
        current_dim=531,goal_dim=518,encoder_family=FAMILY,latent_dim=DIM,
        new_training_updates=0,Native_not_yet_run=True)
    publish(output,binding)
    result = dict(actual_exit_code=0,group_id=GROUP,binding_path=str(output),binding_sha256=sha(output),
        final_counter=1000000,current_dim=531,goal_dim=518,encoder_family=FAMILY,latent_dim=DIM,
        full153_CPU_acceptance=True,checkpoint_transactions=11,completed=time.time())
    publish(receipt,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','native'))
    parser.add_argument('--case',type=Path,required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    assert case.name == 'cube2_pldm_gciql_v1'
    print(json.dumps((prepare_train if args.phase == 'train' else prepare_native)(case)))
