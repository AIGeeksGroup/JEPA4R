"""CPU-only audit of new per-episode latents; never read old NPZ/model contents."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


if not __debug__:
    raise RuntimeError('Cache audit requires assertions enabled; remove -O/PYTHONOPTIMIZE')


EXPECTED_EPISODES = {'train': 1000, 'val': 100}
ROWS, DIM = 1001, 192
PROTOCOL_KEYS = ('checkpoint_sha256', 'data_acceptance_sha256',
    'source_stat_freeze_sha256', 'source_pins', 'camera', 'renderer_workers',
    'encoder_batch', 'torch_cpu_threads', 'split_episodes', 'raw_rows_per_episode')
PRODUCER = 'official_state_2d_cache_v1/produce.py'
ADAPTER = 'official_state_2d_cache_v1/render_adapter.py'
GATE = dict(pixel_atol=1e-6, pixel_rtol=0., latent_atol=1e-5, latent_rtol=1e-5,
            minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20,
            scope='original preprocessing plus frozen encode; rendering excluded; CUDA tail synchronized')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def camera_check(actual, planned):
    for key, value in dict(source_name='front', render_name='front_pixels', FOV=45.,
            width=224, height=224, offsamples=4, ob_type='states', visualize_info=False,
            pixel_transparent_arm=False, pullback=0, crop=None).items():
        assert actual[key] == value, 'camera: ' + key
        if key in planned:
            assert planned[key] == value, 'plan camera: ' + key
    assert isinstance(actual['source_id'], int) and actual['source_id'] >= 0
    assert isinstance(actual['render_id'], int) and actual['render_id'] >= 0
    assert actual['source_id'] != actual['render_id'], 'Named camera slots must be distinct'
    for key in ('source_position', 'position'):
        np.testing.assert_allclose(actual[key], [1.287, 0., .509], rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual['source_quaternion'], planned['base_quaternion'], rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual['actual_quaternion'], planned['actual_quaternion'], rtol=0., atol=1e-12)
    alpha = actual['arm_material_alpha']
    assert isinstance(alpha, dict) and alpha and all(value == 1. for value in alpha.values()), 'arm alpha'


def audit(cache_dir, plan_path, progress=None):
    """Verify the fixed complete cache; optional callback writes audit-owned status."""
    assert not Path(cache_dir).is_symlink(), 'Cache root must not be a symlink'
    cache_dir, plan_path = Path(cache_dir).resolve(), Path(plan_path).resolve()
    tracked = {}

    def track(path):
        path = Path(path).resolve()
        current = stat(path)
        assert tracked.setdefault(path, current) == current, 'File changed during audit: ' + str(path)
        return current

    def read(path, expected_sha=None):
        path = Path(path)
        before = track(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert expected_sha is None or digest == expected_sha, 'JSON/source SHA: ' + str(path)
        assert stat(path) == before, 'File changed while reading: ' + str(path)
        return json.loads(content), digest

    plan, plan_sha = read(plan_path)
    assert plan['split_episodes'] == EXPECTED_EPISODES, 'Require full official1000/100 Double cache'
    assert plan['raw_rows_per_episode'] == ROWS and plan['encoder_batch'] == 32
    assert Path(plan['cache_dir']).resolve() == cache_dir, 'Plan cache_dir differs from audited cache'
    workspace = Path(plan['workspace']).resolve()
    for relative, expected_sha in plan['source_pins'].items():
        relative_path = Path(relative)
        assert not relative_path.is_absolute() and '..' not in relative_path.parts, relative
        path = workspace / relative_path
        assert path.suffix not in {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.h5', '.hdf5'}, 'Source pin is data/model'
        before = track(path)
        assert sha(path) == expected_sha and stat(path) == before, 'Source pin: ' + relative
    base_protocol = {key: plan[key] for key in PROTOCOL_KEYS}
    base_protocol.update(producer_source_sha256=plan['source_pins'][PRODUCER],
        render_adapter_source_sha256=plan['source_pins'][ADAPTER], latent_dim=DIM, dtype='float32',
        pixel_preprocess_device=plan.get('pixel_preprocess_device', 'cpu'),
        preprocess_gate=plan.get('preprocess_gate', {}))
    base_sha = canonical_sha(base_protocol)
    manifest, manifest_sha = read(cache_dir / 'cache_manifest.json')
    assert manifest['base_protocol'] == base_protocol and manifest['base_contract_sha256'] == base_sha
    assert manifest['actual_exit_code'] == 0
    policy, selection_sha = base_protocol['pixel_preprocess_device'], None
    assert policy in ('cpu', 'cuda', 'auto_validated'), 'Unsupported preprocessing policy'
    selected_device = 'cpu'
    if policy != 'cpu':
        assert plan['preprocess_gate'] == GATE, 'Unapproved numeric/timing thresholds'
        selection_path = cache_dir / 'preprocess_selection.json'
        selection, selection_sha = read(selection_path)
        binding = manifest['preprocessing_selection']
        assert Path(binding['path']).resolve() == selection_path.resolve() and binding['sha256'] == selection_sha
        assert selection['base_contract_sha256'] == base_sha and selection['actual_exit_code'] == 0
        assert selection['thresholds'] == GATE and selection['batch'] == 32 and selection['CPU_path_finite'] is True
        selected_device = selection['selected_device']
        assert selected_device in ('cpu', 'cuda')
        if selected_device == 'cuda':
            assert selection['numeric_passed'] is True and selection['CUDA_path_finite'] is True
            assert selection['cuda_faster_by_required_gain'] is True
            assert selection['cuda_gain'] >= GATE['minimum_gain']
    else:
        assert manifest['preprocessing_selection'] is None
    protocol = dict(base_protocol, pixel_preprocess_device=selected_device,
                    preprocessing_selection_sha256=selection_sha)
    protocol_sha = canonical_sha(protocol)
    stored_protocol, protocol_file_sha = read(cache_dir / 'protocol.json')
    assert stored_protocol == protocol, 'Protocol payload differs from selected preprocessing identity'
    assert manifest['protocol'] == protocol and manifest['protocol_sha256'] == protocol_sha
    assert set(manifest['splits']) == set(EXPECTED_EPISODES)
    seal, seal_sha = read(cache_dir / 'cache_seal.json')
    assert seal['ready_for_downstream'] is False, 'Producer seal must remain pending independent closure'
    assert seal['actual_exit_code'] == 0 and seal['manifest_sha256'] == manifest_sha
    assert seal['protocol_sha256'] == protocol_sha
    accepted, _ = read(plan['data_acceptance'], plan['data_acceptance_sha256'])
    frozen, _ = read(plan['source_stat_freeze'], plan['source_stat_freeze_sha256'])
    assert accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2 and accepted['actual_exit_code'] == frozen['actual_exit_code'] == 0
    assert frozen['acceptance_sha256'] == plan['data_acceptance_sha256']
    arrays_path = Path(plan['shared_state_root']) / 'source_arrays.json'
    arrays, arrays_sha = read(arrays_path)
    arrays_contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
        source_stat_freeze_sha256=plan['source_stat_freeze_sha256'], split_episodes=plan['split_episodes'],
        raw_rows_per_episode=ROWS, qpos_width=28, qvel_width=26,
        extractor_source_sha256=plan['source_pins'][PRODUCER])
    arrays_protocol = canonical_sha(arrays_contract)
    assert arrays['contract'] == arrays_contract and arrays['protocol_sha256'] == arrays_protocol
    assert manifest['source_arrays_protocol_sha256'] == arrays_protocol
    assert manifest['source_arrays_receipt']['sha256'] == arrays_sha
    assert Path(manifest['source_arrays_receipt']['path']).resolve() == arrays_path.resolve()
    assert set(arrays['splits']) == set(EXPECTED_EPISODES)
    identities = dict(producer_source_sha256=protocol['producer_source_sha256'],
        render_adapter_source_sha256=protocol['render_adapter_source_sha256'],
        protocol_sha256=protocol_sha, checkpoint_sha256=plan['checkpoint_sha256'],
        preprocess_source_sha256=plan['source_pins']['source/lewm_pcd/utils.py'],
        checkpoint_encoder_sha256=plan['source_pins']['checkpoint_encoder.py'],
        camera_contract_sha256=canonical_sha(plan['camera']))
    assert all(is_sha(value) for value in identities.values())
    completed, latent_bytes, receipt_bindings = 0, 0, {}
    for split, count in EXPECTED_EPISODES.items():
        source = arrays['splits'][split]
        old = frozen['datasets']['validation' if split == 'val' else 'train']
        assert Path(source['source_npz']).resolve() == Path(old['path']).resolve()
        assert source['accepted_stat'] == old['accepted_stat'] == track(source['source_npz']), 'Old NPZ stat'
        assert source['rows'] == count * ROWS
        for key, width in (('qpos', 28), ('qvel', 26)):
            item = source[key]
            assert item['shape'] == [count * ROWS, width] and item['dtype'] == 'float32'
            actual_stat = track(item['path'])
            expected_stat = item['stat']
            assert {'size', 'mtime_ns', 'inode'} <= expected_stat.keys() <= actual_stat.keys()
            assert expected_stat == {key: actual_stat[key] for key in expected_stat}, 'Source-array stat'
            assert is_sha(item['sha256']), 'Source-array content identity'
        entries = manifest['splits'][split]
        assert [entry['episode'] for entry in entries] == list(range(count)), 'Manifest EP order/coverage'
        assert not (cache_dir / split).is_symlink(), 'Split path must not be a symlink'
        expected_names = {f'episode_{episode:06d}' for episode in range(count)}
        actual_names = {path.name for path in (cache_dir / split).iterdir()
                        if re.fullmatch(r'episode_\d{6}', path.name)}
        assert actual_names == expected_names, 'Unexpected/missing episode path'
        receipt_bindings[split] = []
        for episode, entry in enumerate(entries):
            directory = cache_dir / split / f'episode_{episode:06d}'
            receipt_path, latent_path = directory / 'receipt.json', directory / 'latent.npy'
            assert not directory.is_symlink() and not receipt_path.is_symlink() and not latent_path.is_symlink()
            assert cache_dir in directory.resolve().parents, 'Episode escaped the fixed cache root'
            assert Path(entry['receipt_path']).resolve() == receipt_path.resolve()
            receipt, receipt_sha = read(receipt_path, entry['receipt_sha256'])
            assert receipt['status'] == 'episode_committed' and receipt['actual_exit_code'] == 0
            start = episode * ROWS
            expected = dict(split=split, episode=episode, raw_row_start=start, raw_row_count=ROWS,
                            main_rows=1000, terminal_raw_id=start + 1000)
            assert all(receipt[key] == value for key, value in expected.items()), 'EP/raw ID contract'
            ids_sha = hashlib.sha256(np.arange(start, start + ROWS, dtype='<i8').tobytes()).hexdigest()
            assert receipt['canonical_raw_ids_sha256'] == ids_sha, 'Canonical raw ID order'
            assert all(receipt[key] == value for key, value in identities.items()), 'EP identity/provenance'
            assert receipt['pixel_preprocess_device'] == selected_device
            assert receipt['preprocessing_selection_sha256'] == selection_sha
            assert Path(receipt['source_arrays_receipt']['path']).resolve() == arrays_path.resolve()
            assert receipt['source_arrays_receipt']['sha256'] == arrays_sha
            assert Path(receipt['source_npz']['path']).resolve() == Path(source['source_npz']).resolve()
            assert receipt['source_npz']['accepted_stat'] == source['accepted_stat']
            camera_check(receipt['camera'], plan['camera'])
            geometry = receipt['render_geometry']
            assert all(geometry[key] == value for key, value in dict(qpos_max_abs_error=0.,
                qvel_max_abs_error=0., nq=28, nv=26, nu=7).items()), 'Geometry witness'
            assert is_sha(receipt['RGB_sha256']), 'RGB digest format'
            assert receipt['encoder_rows_valid'] == ROWS and receipt['padding_rows'] == 23
            assert receipt['encoder_rows_executed'] == 1024
            latent = receipt['latent']
            assert Path(latent['path']).resolve() == latent_path.resolve()
            before = track(latent_path)
            assert latent['bytes'] == before['size'] and latent['shape'] == [ROWS, DIM]
            assert latent['dtype'] == 'float32'
            assert sha(latent_path) == latent['sha256'], 'latent SHA: ' + str(latent_path)
            value = np.load(latent_path, allow_pickle=False, mmap_mode='r')
            assert value.shape == (ROWS, DIM) and value.dtype == np.dtype('float32'), 'latent shape/dtype'
            assert value.offset + value.nbytes == before['size'], 'latent NPY length/trailing bytes'
            assert np.isfinite(value).all(), 'Nonfinite latent: ' + str(latent_path)
            del value
            assert stat(latent_path) == before, 'Latent changed while auditing'
            latent_bytes += before['size']
            completed += 1
            receipt_bindings[split].append(dict(episode=episode, receipt_sha256=receipt_sha,
                latent_sha256=latent['sha256'], raw_row_start=start, terminal_raw_id=start + 1000))
            if progress is not None and (completed % 100 == 0 or episode == count - 1):
                progress(dict(stage='auditing', split=split, episode=episode, completed_episodes=completed,
                              target_episodes=sum(EXPECTED_EPISODES.values())))
    for path, before in tracked.items():
        assert stat(path) == before, 'Tracked source/evidence changed during complete audit: ' + str(path)
    return dict(stage='new_feature_cache_audit_pass', actual_exit_code=0,
        plan_sha256=plan_sha, protocol_sha256=protocol_sha, protocol_file_sha256=protocol_file_sha,
        base_contract_sha256=base_sha, pixel_preprocess_device=selected_device,
        preprocessing_selection_sha256=selection_sha, source_arrays_protocol_sha256=arrays_protocol,
        manifest_sha256=manifest_sha, producer_seal_sha256=seal_sha,
        source_arrays_receipt_sha256=arrays_sha, episodes_verified=completed,
        raw_rows_verified=completed * ROWS, main_rows_verified=completed * 1000,
        latent_dim=DIM, dtype='float32', total_latent_NPY_bytes=latent_bytes,
        receipt_bindings=receipt_bindings, all_new_latent_SHA_shape_dtype_finite=True,
        all_raw_ID_and_true_terminal_next_contracts=True, tracked_stats_stable=True,
        old_NPZ_contents_read=False, source_arrays_contents_rehashed=False,
        model_checkpoint_contents_read=False, RGB_re_rendered=False, encoder_executed=False,
        RGB_digest_and_geometry_witness_scope='producer receipts checked; not independently re-rendered',
        ready_for_downstream=False, final_READY_owner='Root after actual outer exits and independent audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.cache_dir / 'outer_audit_v1'
    output.mkdir(parents=True, exist_ok=False)
    started, code = time.time(), 1

    def write_new(name, value):
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def progress(fields):
        record = dict(started=started, updated=time.time(), actual_exit_code=None)
        record.update(fields)
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output / 'status.json')
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')

    try:
        progress(dict(stage='starting', completed_episodes=0))
        report = audit(args.cache_dir, args.plan, progress)
        report.update(auditor_sha256=sha(__file__), started=started, completed=time.time())
        write_new('report.json', report)
        code = 0
        progress(dict(stage='audit_complete', completed_episodes=report['episodes_verified'], actual_exit_code=0))
        print(json.dumps(dict(actual_exit_code=0, report=str(output / 'report.json'),
                              episodes_verified=report['episodes_verified'], ready_for_downstream=False)))
    except BaseException as exc:
        write_new('failure.json', dict(stage='audit_failed', error=repr(exc), actual_exit_code=1,
                                      started=started, completed=time.time()))
        progress(dict(stage='audit_failed', error=repr(exc), actual_exit_code=1))
        raise
    finally:
        (output / 'actual_exit_code').write_text(str(code) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
