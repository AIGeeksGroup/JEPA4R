"""One complete small fixture and one real feature-byte corruption case."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from common import episode_seed, object_sha, protocol_payload

HERE = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fixture(root):
    counts, cache, shared = {'train': 2, 'val': 2}, root/'cache', root/'shared'
    source_names = ['official_state_2d_cache_v1/produce.py', 'source/lewm_pcd/utils.py',
        'checkpoint_encoder.py', 'source/lewm_pcd/envs/pcd_wrapper.py',
        'external/native_pcd_frame.py', 'external/pcd_geometry.py',
        'official_state_3d_cache_v1/common.py', 'official_state_3d_cache_v1/gate.py',
        'official_state_3d_cache_v1/pcd_renderer.py', 'official_state_3d_cache_v1/produce.py']
    pins = {}
    for name in source_names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture source ' + name)
        pins[name] = sha(path)
    checkpoint = root/'checkpoints/3d_epoch10/weights_epoch_10.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'OLD MODEL CONTENT MUST NOT BE READ')
    config = dict(_target_='jepa.JEPA', obs_key='pcd', encoder=dict(
        _target_='encoders.DexArtPointNet', in_dim=6, out_dim=192, chunk_points=1200),
        projector=dict(input_dim=192, output_dim=192))
    write(checkpoint.parent/'config.json', config)
    (checkpoint.parent/'config.yaml').write_text('# small frozen model config')
    for name in ('config.json','config.yaml'):
        relative = 'checkpoints/3d_epoch10/' + name
        pins[relative] = sha(root/relative)
    raw = {}
    for split in counts:
        path = root/f'{split}.npz'
        path.write_bytes(b'OLD NPZ CONTENT MUST NOT BE READ')
        raw[split] = dict(path=str(path), accepted_stat=stat(path))
    acceptance, frozen = root/'accepted.json', root/'stat-freeze.json'
    write(acceptance, dict(stage='complete', actual_exit_code=0))
    write(frozen, dict(actual_exit_code=0, acceptance_sha256=sha(acceptance),
        datasets={'train':raw['train'], 'validation':raw['val']}))
    camera = dict(source_name='front', render_name='front_pixels', position=[1.287,0.,.509],
        base_quaternion=[.5792028135240382,.4056157058175116,.4056157058175116,.5792028135240384],
        actual_quaternion=[.5609588321928806,.4304941214288525,.4304941214288525,.5609588321928807],
        FOV=45., width=224, height=224, offsamples=4, ob_type='states', visualize_info=False,
        pixel_transparent_arm=False, pullback=0, crop=None)
    plan = dict(workspace=str(root), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        data_acceptance=str(acceptance), data_acceptance_sha256=sha(acceptance),
        source_stat_freeze=str(frozen), source_stat_freeze_sha256=sha(frozen),
        source_pins=pins, shared_state_root=str(shared), cache_dir=str(cache), gate_dir=str(root/'gate'),
        camera=camera, split_episodes=counts, raw_rows_per_episode=1001, encoder_batch=32,
        renderer_workers=2, pcd=dict(points=1200, channels=6,
        crop_box=[[-.25,-.5,-.02],[.75,.5,.9]], voxel_size=.005, cube_fraction=.25,
        foreground_only=True, geometry_offsamples=0, sampling_namespace='triple_official_pcd_epoch10_v1',
        sampling_policy='SHA256([namespace,split,episode]) first4bytes little uint32; continuous1001 current captures'),
        preprocess=dict(device='cpu',center=[.425,0.,.2],scale=4.,use_rgb=True,original_factory=True),
        precision=dict(dtype='float32',AMP=False,change_original_settings=False))
    write(root/'official_state_2d_cache_v1/producer_plan.json', dict(source_pins={
        'official_state_2d_cache_v1/produce.py':pins['official_state_2d_cache_v1/produce.py']}))
    arrays_contract = dict(data_acceptance_sha256=sha(acceptance), source_stat_freeze_sha256=sha(frozen),
        split_episodes=counts, raw_rows_per_episode=1001, qpos_width=35, qvel_width=32,
        extractor_source_sha256=pins['official_state_2d_cache_v1/produce.py'])
    arrays = dict(contract=arrays_contract,protocol_sha256=object_sha(arrays_contract),splits={})
    for split,count in counts.items():
        item = dict(source_npz=raw[split]['path'],accepted_stat=raw[split]['accepted_stat'],rows=count*1001)
        for key,width in [('qpos',35),('qvel',32)]:
            path=shared/f'{split}_{key}.npy';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'OLD SOURCE ARRAY CONTENT MUST NOT BE READ')
            item[key]=dict(path=str(path),shape=[count*1001,width],dtype='float32',stat=stat(path),sha256=sha(path))
        arrays['splits'][split]=item
    arrays_path=shared/'source_arrays.json';write(arrays_path,arrays)
    plan['source_arrays_receipt_sha256']=sha(arrays_path)
    plan_path=root/'plan.json';write(plan_path,plan)
    frozen_tensor='f'*64
    reference=dict(actual_exit_code=0,checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha(plan_path),
        strict_all_tensors_equal=True,frozen_model_state_sha256=frozen_tensor,encoder_projector_only=True,
        updates=0,SR_episodes=0,raw_ids=list(range(29))+[999,1000,1001],
        input_shape=[32,1200,6],output_shape=[32,192],dtype='float32',AMP=False,finite=True,
        fixture_sha256='d'*64,reference_latent_sha256='e'*64)
    write(root/'gate/reference_report.json',reference)
    write(root/'gate/fresh_report.json',dict(reference,fresh_reload_bitexact=True,max_abs_difference=0.))
    protocol=protocol_payload(plan);protocol_sha=object_sha(protocol);write(cache/'protocol.json',protocol)
    manifest=dict(schema_version=1,actual_exit_code=0,protocol=protocol,protocol_sha256=protocol_sha,
        source_arrays_receipt=dict(path=str(arrays_path),sha256=sha(arrays_path)),splits={},
        source_encoder_state_sha256=frozen_tensor,checkpoint_sha256=plan['checkpoint_sha256'],
        full_raw_RGB_or_PCD_persisted=False,training_updates=0,SR_episodes=0,world_model_dynamics_calls=0)
    for split,count in counts.items():
        manifest['splits'][split]=[]
        for episode in range(count):
            start=episode*1001;directory=cache/split/f'episode_{episode:06d}';directory.mkdir(parents=True)
            latent=directory/'latent.npy';np.save(latent,np.full((1001,192),.25+episode,np.float32),allow_pickle=False)
            seed=episode_seed(split,episode);rng=np.random.default_rng(seed)
            before=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
            selected=np.concatenate([rng.choice(np.arange(400),300,replace=False),rng.choice(np.arange(1600),900,replace=False)])
            rng.permutation(selected)
            after=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
            bounds={}
            for index in (0,999,1000):
                bounds[str(index)]=dict(phase='current',capture_index=index,sampling_seed=seed,
                    rng_before_sha256=before if index==0 else ('a' if index==999 else 'b')*64,
                    rng_after_sha256=after if index==0 else ('b' if index==999 else 'c')*64,
                    shape=[1200,6],cloud_sha256=hashlib.sha256(np.zeros((1200,6),np.float32).tobytes()).hexdigest(),RGB_AA4_sha256='b'*64,
                    source_UV_sha256='c'*64,source_geom_sha256='d'*64,
                    core_source_sha256=pins['source/lewm_pcd/envs/pcd_wrapper.py'],
                    provider_source_sha256=pins['external/pcd_geometry.py'],geometry_offsamples=0,no_feature_cache=True,
                    sampling=dict(raw_foreground_pool=2200,foreground_cropped_pool=2100,voxel_unique_pool=2000,
                        cube_unique_pool=400,other_unique_pool=1600,sampled_cube_points=300,
                        actual_selected_unique_pool_indices=1200,repeated_samples=0,fallback_used=False,fallback_reason=None),
                    source_pose_sha256={key:'e'*64 for key in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']},source_time=0.)
            actual_camera=dict(camera,source_id=0,render_id=1,source_position=camera['position'],
                source_quaternion=camera['base_quaternion'],arm_material_alpha={'ur5e':1.})
            receipt=dict(status='episode_committed',actual_exit_code=0,split=split,episode=episode,
                raw_row_start=start,raw_row_count=1001,main_rows=1000,terminal_raw_id=start+1000,
                canonical_raw_ids_sha256=hashlib.sha256(np.arange(start,start+1001,dtype='<i8').tobytes()).hexdigest(),
                latent=dict(path=str(latent),bytes=latent.stat().st_size,sha256=sha(latent),shape=[1001,192],dtype='float32'),
                protocol_sha256=protocol_sha,checkpoint_sha256=plan['checkpoint_sha256'],
                source_arrays_receipt=manifest['source_arrays_receipt'],source_npz=raw[split],sampling_seed=seed,
                capture_boundaries=bounds,capture_receipt_stream_sha256='a'*64,RGB_sha256='b'*64,
                PCD_xyz_sha256='c'*64,PCD_rgb_sha256='d'*64,camera=actual_camera,
                render_geometry=dict(qpos_max_abs_error=0.,qvel_max_abs_error=0.,nq=35,nv=32,nu=7),
                encoder_rows_valid=1001,padding_rows=23,encoder_rows_executed=1024)
            receipt_path=directory/'receipt.json';write(receipt_path,receipt)
            manifest['splits'][split].append(dict(episode=episode,receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path)))
    write(cache/'cache_manifest.json',manifest)
    write(cache/'cache_seal.json',dict(actual_exit_code=0,ready_for_downstream=False,
        manifest_sha256=sha(cache/'cache_manifest.json'),protocol_sha256=protocol_sha,
        train_rows=2002,val_rows=2002,full_raw_ID_coverage=True))
    write(cache/'pilot_receipt.json',dict(actual_exit_code=0,source_encoder_frozen=True,
        protocol_sha256=protocol_sha,source_arrays_receipt=manifest['source_arrays_receipt'],
        completed_episodes=2,completed_rows=2002,
        splits={split:[entries[0]] for split,entries in manifest['splits'].items()}))
    for split in counts:
        np.savez(cache/f'pilot_{split}_boundary.npz',raw_ids=np.array([999,1000,1001],np.int64),
            pcd_xyz=np.zeros((3,1200,3),np.float32),pcd_rgb=np.zeros((3,1200,3),np.uint8),
            latents=np.stack([np.full(192,.25,np.float32),np.full(192,.25,np.float32),np.full(192,1.25,np.float32)]))
    return cache,plan_path,counts


class AuditContract(unittest.TestCase):
    def auditor(self):
        path=HERE/'audit_cache.py'
        self.assertTrue(path.is_file(),'3D cache auditor implementation is missing')
        spec=importlib.util.spec_from_file_location('pcd_auditor_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_positive_new_features_old_sources_stat_only(self):
        auditor=self.auditor()
        with tempfile.TemporaryDirectory() as folder:
            cache,plan,counts=fixture(Path(folder))
            with mock.patch.dict(auditor.EXPECTED_EPISODES,counts,clear=True):
                report=auditor.audit(cache,plan)
            self.assertEqual(report['episodes_verified'],4)
            self.assertEqual(report['raw_rows_verified'],4004)
            self.assertEqual(report['actual_exit_code'],0)
            self.assertFalse(report['old_NPZ_contents_read'])
            self.assertFalse(report['source_arrays_contents_rehashed'])
            self.assertFalse(report['model_checkpoint_contents_read'])

    def test_negative_one_real_latent_byte_must_fail_SHA(self):
        auditor=self.auditor()
        with tempfile.TemporaryDirectory() as folder:
            cache,plan,counts=fixture(Path(folder))
            path=cache/'train/episode_000001/latent.npy'
            with path.open('r+b') as stream:
                stream.seek(-1,2);old=stream.read(1);stream.seek(-1,2);stream.write(bytes([old[0]^1]))
            with mock.patch.dict(auditor.EXPECTED_EPISODES,counts,clear=True):
                with self.assertRaisesRegex(AssertionError,'latent SHA'):
                    auditor.audit(cache,plan)


if __name__=='__main__':
    unittest.main()
