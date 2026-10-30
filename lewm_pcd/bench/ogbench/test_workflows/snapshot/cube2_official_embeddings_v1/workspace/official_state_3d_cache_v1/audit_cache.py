"""CPU-only new 3D feature audit; old raw/model arrays remain stat-only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np

from common import episode_seed, object_sha, protocol_payload

if not __debug__:
    raise RuntimeError('Cache audit requires assertions enabled; remove -O/PYTHONOPTIMIZE')

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = '75eb5137c30cac3628f0e781c7d2eb4a96b29eb7689a7fcae6ce1509dc303adc'
EXPECTED_EPISODES = {'train': 1000, 'val': 100}
ROWS, DIM = 1001, 192


def original_helpers():
    path = HERE.parent / 'official_state_2d_cache_v1/audit_cache.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA
    spec = importlib.util.spec_from_file_location('_original_2d_audit_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = original_helpers()
sha, stat, is_sha, camera_check = OLD.sha, OLD.stat, OLD.is_sha, OLD.camera_check


def rng_sha(rng):
    # This is the original provider's JSON encoding, not the compact protocol hash.
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def capture_check(bounds, seed, pins):
    assert set(bounds) == {'0', '999', '1000'}, 'PCD capture boundary coverage'
    for index in (0, 999, 1000):
        item = bounds[str(index)]
        assert item['phase'] == 'current' and item['capture_index'] == index
        assert item['sampling_seed'] == seed and item['shape'] == [1200, 6]
        assert item['core_source_sha256'] == pins['source/lewm_pcd/envs/pcd_wrapper.py']
        assert item['provider_source_sha256'] == pins['external/pcd_geometry.py']
        assert item['geometry_offsamples'] == 0 and item['no_feature_cache'] is True
        for key in ('rng_before_sha256', 'rng_after_sha256', 'cloud_sha256',
                    'RGB_AA4_sha256', 'source_UV_sha256', 'source_geom_sha256'):
            assert is_sha(item[key]), 'PCD capture digest: ' + key
        assert set(item['source_pose_sha256']) == {'qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat'}
        assert all(is_sha(value) for value in item['source_pose_sha256'].values())
        assert np.isfinite(item['source_time'])
        sample = item['sampling']
        assert sample['raw_foreground_pool'] >= sample['foreground_cropped_pool'] >= 0
        cube, other, voxels = (sample[key] for key in
                              ('cube_unique_pool', 'other_unique_pool', 'voxel_unique_pool'))
        assert cube >= 0 and other >= 0 and cube + other == voxels > 0
        assert 1 <= sample['actual_selected_unique_pool_indices'] <= min(1200, voxels)
        assert sample['repeated_samples'] == 1200 - sample['actual_selected_unique_pool_indices']
        assert sample['sampled_cube_points'] == (min(300, cube) if other else 1200)
        if sample['fallback_used']:
            assert sample['foreground_cropped_pool'] == 0 and sample['raw_foreground_pool'] > 0
            assert sample['fallback_reason'] == 'original_crop_empty_use_actual_foreground'
            assert voxels <= sample['raw_foreground_pool']
        else:
            assert sample['fallback_reason'] is None and voxels <= sample['foreground_cropped_pool']
    # Independently verify the first private stream transition from the declared seed.
    rng, first = np.random.default_rng(seed), bounds['0']
    assert rng_sha(rng) == first['rng_before_sha256'], 'EP seed/private RNG start'
    cube, other = (first['sampling'][key] for key in ('cube_unique_pool', 'other_unique_pool'))
    if other == 0:
        selected = rng.choice(np.arange(cube), 1200, replace=cube < 1200)
    else:
        n_cube = min(300, cube)
        selected = np.concatenate([rng.choice(np.arange(cube), n_cube, replace=False),
            rng.choice(np.arange(other), 1200 - n_cube, replace=other < 1200 - n_cube)])
    rng.permutation(selected)
    assert rng_sha(rng) == first['rng_after_sha256'], 'EP first private RNG transition'
    assert bounds['999']['rng_after_sha256'] == bounds['1000']['rng_before_sha256'], 'Terminal RNG adjacency'


def audit(cache_dir, plan_path, progress=None):
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
        assert stat(path) == before
        return json.loads(content), digest

    plan, plan_sha = read(plan_path)
    assert plan['split_episodes'] == EXPECTED_EPISODES
    assert plan['raw_rows_per_episode'] == ROWS and plan['encoder_batch'] == 32 and plan['renderer_workers'] == 2
    assert Path(plan['cache_dir']).resolve() == cache_dir
    assert plan['pcd'] == dict(points=1200, channels=6, crop_box=[[.1,-.5,-.02],[.75,.5,.6]],
        voxel_size=.005, cube_fraction=.25, foreground_only=True, geometry_offsamples=0,
        sampling_namespace='double_official_pcd_epoch10_v1',
        sampling_policy='SHA256([namespace,split,episode]) first4bytes little uint32; continuous1001 current captures')
    assert plan['preprocess'] == dict(device='cpu', center=[.425,0.,.2], scale=4., use_rgb=True, original_factory=True)
    assert plan['precision'] == dict(dtype='float32', AMP=False, change_original_settings=False)
    workspace, pins = Path(plan['workspace']).resolve(), plan['source_pins']
    required = {'checkpoint_encoder.py', 'source/lewm_pcd/utils.py',
        'source/lewm_pcd/envs/pcd_wrapper.py', 'external/native_pcd_frame.py', 'external/pcd_geometry.py',
        'official_state_2d_cache_v1/produce.py', 'official_state_3d_cache_v1/common.py',
        'official_state_3d_cache_v1/gate.py', 'official_state_3d_cache_v1/pcd_renderer.py',
        'official_state_3d_cache_v1/produce.py'}
    assert required <= pins.keys()
    for relative, digest in pins.items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        path = workspace / path
        assert path.suffix not in {'.npz','.npy','.pt','.pth','.ckpt','.h5','.hdf5'}
        before = track(path)
        assert sha(path) == digest and stat(path) == before, 'Source pin: ' + relative
    checkpoint = Path(plan['checkpoint']).resolve()
    checkpoint_stat = track(checkpoint)  # Original model bytes are not re-read.
    config_path = checkpoint.parent / 'config.json'
    assert config_path.relative_to(workspace).as_posix() in pins
    config, _ = read(config_path)
    assert config['_target_'] == 'jepa.JEPA' and config['obs_key'] == 'pcd'
    enc = config['encoder']
    assert enc['_target_'] == 'encoders.DexArtPointNet' and enc['in_dim'] == 6
    assert enc['out_dim'] == DIM and enc['chunk_points'] == 1200
    assert config['projector']['input_dim'] == config['projector']['output_dim'] == DIM
    gate_dir = Path(plan['gate_dir'])
    reference, reference_sha = read(gate_dir / 'reference_report.json')
    fresh, fresh_sha = read(gate_dir / 'fresh_report.json')
    for report in (reference, fresh):
        assert report['actual_exit_code'] == 0 and report['finite'] is True
        assert report['checkpoint_sha256'] == plan['checkpoint_sha256'] and report['plan_sha256'] == plan_sha
        assert report['strict_all_tensors_equal'] and report['encoder_projector_only']
        assert report['updates'] == report['SR_episodes'] == 0
        assert report['raw_ids'] == list(range(29)) + [999,1000,1001]
        assert report['input_shape'] == [32,1200,6] and report['output_shape'] == [32,DIM]
        assert report['dtype'] == 'float32' and report['AMP'] is False
    assert fresh['fresh_reload_bitexact'] and fresh['max_abs_difference'] == 0.
    assert fresh['frozen_model_state_sha256'] == reference['frozen_model_state_sha256']
    assert fresh['fixture_sha256'] == reference['fixture_sha256']
    protocol = protocol_payload(plan)
    protocol_sha = object_sha(protocol)
    stored_protocol, protocol_file_sha = read(cache_dir / 'protocol.json')
    assert stored_protocol == protocol
    manifest, manifest_sha = read(cache_dir / 'cache_manifest.json')
    assert manifest['actual_exit_code'] == 0 and manifest['protocol'] == protocol
    assert manifest['protocol_sha256'] == protocol_sha and manifest['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert manifest['source_encoder_state_sha256'] == fresh['frozen_model_state_sha256']
    assert manifest['training_updates'] == manifest['SR_episodes'] == manifest['world_model_dynamics_calls'] == 0
    assert manifest['full_raw_RGB_or_PCD_persisted'] is False
    seal, seal_sha = read(cache_dir / 'cache_seal.json')
    assert seal['actual_exit_code'] == 0 and seal['ready_for_downstream'] is False
    assert seal['manifest_sha256'] == manifest_sha and seal['protocol_sha256'] == protocol_sha
    assert seal['train_rows'] == EXPECTED_EPISODES['train'] * ROWS
    assert seal['val_rows'] == EXPECTED_EPISODES['val'] * ROWS and seal['full_raw_ID_coverage'] is True
    accepted, _ = read(plan['data_acceptance'], plan['data_acceptance_sha256'])
    frozen, _ = read(plan['source_stat_freeze'], plan['source_stat_freeze_sha256'])
    assert accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2 and accepted['actual_exit_code'] == frozen['actual_exit_code'] == 0
    assert frozen['acceptance_sha256'] == plan['data_acceptance_sha256']
    arrays_path = Path(plan['shared_state_root']) / 'source_arrays.json'
    arrays, arrays_sha = read(arrays_path, plan['source_arrays_receipt_sha256'])
    old_plan, _ = read(workspace / 'official_state_2d_cache_v1/producer_plan.json')
    old_producer = old_plan['source_pins']['official_state_2d_cache_v1/produce.py']
    assert old_producer == pins['official_state_2d_cache_v1/produce.py']
    contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
        source_stat_freeze_sha256=plan['source_stat_freeze_sha256'], split_episodes=EXPECTED_EPISODES,
        raw_rows_per_episode=ROWS, qpos_width=28, qvel_width=26, extractor_source_sha256=old_producer)
    assert arrays['contract'] == contract and arrays['protocol_sha256'] == object_sha(contract)
    arrays_binding = dict(path=str(arrays_path), sha256=arrays_sha)
    assert manifest['source_arrays_receipt'] == arrays_binding
    assert set(arrays['splits']) == set(manifest['splits']) == set(EXPECTED_EPISODES)
    completed, total_bytes, seed_set, bindings = 0, 0, set(), {}
    for split, count in EXPECTED_EPISODES.items():
        source = arrays['splits'][split]
        prior = frozen['datasets']['validation' if split == 'val' else 'train']
        assert Path(source['source_npz']).resolve() == Path(prior['path']).resolve()
        assert source['accepted_stat'] == prior['accepted_stat'] == track(source['source_npz'])
        assert source['rows'] == count * ROWS
        for key, width in [('qpos',28), ('qvel',26)]:
            item = source[key]
            assert item['shape'] == [count*ROWS,width] and item['dtype'] == 'float32'
            actual_stat = track(item['path'])
            assert all(actual_stat[k] == v for k,v in item['stat'].items())
        entries = manifest['splits'][split]
        assert len(entries) == count and [entry['episode'] for entry in entries] == list(range(count))
        assert not (cache_dir/split).is_symlink()
        names = {path.name for path in (cache_dir/split).iterdir() if re.fullmatch(r'episode_\d{6}',path.name)}
        assert names == {f'episode_{episode:06d}' for episode in range(count)}
        bindings[split] = []
        for episode, entry in enumerate(entries):
            directory = cache_dir/split/f'episode_{episode:06d}'
            receipt_path, latent_path = directory/'receipt.json', directory/'latent.npy'
            assert not any(path.is_symlink() for path in (directory,receipt_path,latent_path))
            assert cache_dir in directory.resolve().parents
            assert Path(entry['receipt_path']).resolve() == receipt_path.resolve()
            receipt, receipt_sha = read(receipt_path, entry['receipt_sha256'])
            assert receipt['status'] == 'episode_committed' and receipt['actual_exit_code'] == 0
            start = episode*ROWS
            for key, expected in dict(split=split,episode=episode,raw_row_start=start,raw_row_count=ROWS,
                    main_rows=1000,terminal_raw_id=start+1000).items():
                assert receipt[key] == expected, 'EP/raw ID: ' + key
            ids = np.arange(start,start+ROWS,dtype='<i8')
            assert receipt['canonical_raw_ids_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest()
            assert receipt['protocol_sha256'] == protocol_sha and receipt['checkpoint_sha256'] == plan['checkpoint_sha256']
            assert receipt['source_arrays_receipt'] == arrays_binding
            assert Path(receipt['source_npz']['path']).resolve() == Path(source['source_npz']).resolve()
            assert receipt['source_npz']['accepted_stat'] == source['accepted_stat']
            seed = episode_seed(split,episode)
            assert receipt['sampling_seed'] == seed and seed not in seed_set
            seed_set.add(seed)
            capture_check(receipt['capture_boundaries'],seed,pins)
            for key in ('capture_receipt_stream_sha256','RGB_sha256','PCD_xyz_sha256','PCD_rgb_sha256'):
                assert is_sha(receipt[key]), key
            camera_check(receipt['camera'],plan['camera'])
            geometry = receipt['render_geometry']
            assert all(geometry[k] == v for k,v in dict(qpos_max_abs_error=0.,qvel_max_abs_error=0.,nq=28,nv=26,nu=7).items())
            assert receipt['encoder_rows_valid'] == ROWS and receipt['padding_rows'] == 23
            assert receipt['encoder_rows_executed'] == 1024
            latent = receipt['latent']
            assert Path(latent['path']).resolve() == latent_path.resolve()
            before = track(latent_path)
            assert latent['bytes'] == before['size'] and latent['shape'] == [ROWS,DIM] and latent['dtype'] == 'float32'
            assert sha(latent_path) == latent['sha256'], 'latent SHA: ' + str(latent_path)
            value = np.load(latent_path,allow_pickle=False,mmap_mode='r')
            assert value.shape == (ROWS,DIM) and value.dtype == np.float32
            assert value.offset + value.nbytes == before['size'] and np.isfinite(value).all()
            del value
            assert stat(latent_path) == before
            completed += 1
            total_bytes += before['size']
            bindings[split].append(dict(episode=episode,receipt_sha256=receipt_sha,latent_sha256=latent['sha256'],
                raw_row_start=start,terminal_raw_id=start+1000,sampling_seed=seed))
            if progress is not None and (completed%100 == 0 or episode == count-1):
                progress(dict(stage='auditing',completed_episodes=completed,split=split,episode=episode))
    pilot, pilot_sha = read(cache_dir/'pilot_receipt.json')
    assert pilot['actual_exit_code'] == 0 and pilot['source_encoder_frozen'] is True
    assert pilot['protocol_sha256'] == protocol_sha and pilot['source_arrays_receipt'] == arrays_binding
    assert pilot['completed_episodes'] == 2 and pilot['completed_rows'] == 2002
    boundary_bindings = {}
    for split in EXPECTED_EPISODES:
        assert pilot['splits'][split] == [manifest['splits'][split][0]], 'Pilot EP0 receipt changed'
        path = cache_dir/f'pilot_{split}_boundary.npz'
        before = track(path)
        boundary_bindings[split] = dict(sha256=sha(path),bytes=before['size'])
        with np.load(path,allow_pickle=False) as boundary:
            assert np.array_equal(boundary['raw_ids'],[999,1000,1001])
            points,colors,latents = (boundary[key] for key in ('pcd_xyz','pcd_rgb','latents'))
            assert points.shape == (3,1200,3) and points.dtype == np.float32 and np.isfinite(points).all()
            assert colors.shape == points.shape and colors.dtype == np.uint8
            assert latents.shape == (3,DIM) and latents.dtype == np.float32 and np.isfinite(latents).all()
            for i,(episode,row) in enumerate(((0,999),(0,1000),(1,0))):
                value=np.load(cache_dir/split/f'episode_{episode:06d}/latent.npy',allow_pickle=False,mmap_mode='r')
                assert np.array_equal(latents[i],value[row]), 'Pilot/full boundary latent mismatch'
                receipt,_=read(cache_dir/split/f'episode_{episode:06d}/receipt.json')
                cloud=np.concatenate((points[i],colors[i].astype(np.float32)),axis=1)
                assert hashlib.sha256(cloud.tobytes()).hexdigest() == receipt['capture_boundaries'][str(row)]['cloud_sha256'], 'Pilot/full boundary cloud mismatch'
                del value
        assert stat(path) == before
    for path, before in tracked.items():
        assert stat(path) == before, 'File changed during complete audit: ' + str(path)
    return dict(stage='new_3d_feature_cache_audit_pass',actual_exit_code=0,plan_sha256=plan_sha,
        protocol_sha256=protocol_sha,protocol_file_sha256=protocol_file_sha,manifest_sha256=manifest_sha,
        producer_seal_sha256=seal_sha,source_arrays_receipt_sha256=arrays_sha,
        source_arrays_true_extractor_sha256=old_producer,checkpoint_sha256=plan['checkpoint_sha256'],
        checkpoint_stat=checkpoint_stat,reference_gate_report_sha256=reference_sha,fresh_gate_report_sha256=fresh_sha,
        pilot_receipt_sha256=pilot_sha,pilot_boundary_bindings=boundary_bindings,pilot_full_999_1000_1001_cloud_and_latent_byteexact=True,
        episodes_verified=completed,raw_rows_verified=completed*ROWS,main_rows_verified=completed*1000,
        latent_dim=DIM,dtype='float32',total_latent_NPY_bytes=total_bytes,receipt_bindings=bindings,
        all_new_latent_SHA_shape_dtype_finite=True,all_canonical_ID_and_terminal_next_contracts=True,
        private_EP_seed_and_first_RNG_transition_and_terminal_adjacency_verified=True,tracked_stats_stable=True,
        old_NPZ_contents_read=False,source_arrays_contents_rehashed=False,model_checkpoint_contents_read=False,
        PCD_re_rendered=False,encoder_executed=False,ready_for_downstream=False,
        geometry_RNG_scope='Producer boundary receipts/source pins checked; no unpersisted per-frame cloud or full RNG replay claimed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started,code=time.time(),1

    def write_new(name,value):
        with (args.output_dir/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

    def progress(fields):
        record=dict(started=started,updated=time.time(),actual_exit_code=None,**fields)
        temporary=args.output_dir/'status.json.tmp'
        temporary.write_text(json.dumps(record)+'\n',encoding='utf-8')
        temporary.replace(args.output_dir/'status.json')

    try:
        progress(dict(stage='starting'))
        report=audit(args.cache_dir,args.plan,progress)
        report.update(auditor_sha256=sha(__file__),original_helpers_sha256=ORIGINAL_SHA,
                      started=started,completed=time.time())
        write_new('report.json',report)
        code=0
        progress(dict(stage='audit_complete',completed_episodes=report['episodes_verified']))
        print(json.dumps(dict(actual_exit_code=0,episodes_verified=report['episodes_verified'],report=str(args.output_dir/'report.json'))))
    except BaseException as exc:
        write_new('failure.json',dict(actual_exit_code=1,error=repr(exc),started=started,completed=time.time()))
        progress(dict(stage='failed',error=repr(exc)))
        raise
    finally:
        (args.output_dir/'actual_exit_code').write_text(str(code)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
