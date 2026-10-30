"""One positive fixture and one real-byte corruption case; no model/source NPZ reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fixture(root):
    counts = {'train': 2, 'val': 1}
    cache, shared = root / 'cache', root / 'shared'
    pins = {}
    for name in ('official_state_2d_cache_v1/produce.py',
                 'official_state_2d_cache_v1/render_adapter.py',
                 'source/lewm_pcd/utils.py', 'checkpoint_encoder.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen fixture: ' + name, encoding='utf-8')
        pins[name] = digest(path)
    camera = dict(source_name='front', render_name='front_pixels',
        position=[1.287, 0., .509], base_quaternion=[.5792028135240382, .4056157058175116,
        .4056157058175116, .5792028135240384], actual_quaternion=[.5609588321928806,
        .4304941214288525, .4304941214288525, .5609588321928807],
        FOV=45., width=224, height=224, offsamples=4, ob_type='states',
        visualize_info=False, pixel_transparent_arm=False, pullback=0, crop=None)
    sources = {}
    for split, count in counts.items():
        raw = root / ('official_' + split + '.npz')
        # Deliberately not a ZIP/NPZ: a passing fixture proves only stat is read.
        raw.write_bytes(b'OLD NPZ CONTENT MUST NOT BE READ OR LOADED')
        sources[split] = dict(path=str(raw), accepted_stat=stat(raw))
    accepted = root / 'accepted.json'
    write(accepted, dict(stage='complete', actual_exit_code=0))
    frozen = root / 'source_stat_freeze.json'
    write(frozen, dict(actual_exit_code=0, acceptance_sha256=digest(accepted),
                      datasets={'train': sources['train'], 'validation': sources['val']}))
    checkpoint = root / 'old_checkpoint.pt'
    checkpoint.write_bytes(b'OLD MODEL MUST NOT BE OPENED BY AUDITOR')
    gate = dict(pixel_atol=1e-6, pixel_rtol=0., latent_atol=1e-5, latent_rtol=1e-5,
                minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20)
    plan = dict(workspace=str(root), cache_dir=str(cache), shared_state_root=str(shared),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        data_acceptance=str(accepted), data_acceptance_sha256=digest(accepted),
        source_stat_freeze=str(frozen), source_stat_freeze_sha256=digest(frozen),
        source_pins=pins, camera=camera, renderer_workers=2, encoder_batch=32,
        torch_cpu_threads=1, split_episodes=counts, raw_rows_per_episode=1001,
        pixel_preprocess_device='auto_validated', preprocess_gate=gate)
    plan_path = root / 'plan.json'
    write(plan_path, plan)
    keys = ('checkpoint_sha256', 'data_acceptance_sha256', 'source_stat_freeze_sha256',
            'source_pins', 'camera', 'renderer_workers', 'encoder_batch',
            'torch_cpu_threads', 'split_episodes', 'raw_rows_per_episode')
    base_protocol = {key: plan[key] for key in keys}
    base_protocol.update(producer_source_sha256=pins['official_state_2d_cache_v1/produce.py'],
        render_adapter_source_sha256=pins['official_state_2d_cache_v1/render_adapter.py'],
        latent_dim=192, dtype='float32', pixel_preprocess_device=plan['pixel_preprocess_device'],
        preprocess_gate=gate)
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    selection_path = cache / 'preprocess_selection.json'
    write(selection_path, dict(base_contract_sha256=canonical(base_protocol), actual_exit_code=0,
        thresholds=gate, batch=32, CPU_path_finite=True, CUDA_path_finite=True,
        numeric_passed=True, cuda_faster_by_required_gain=True, cuda_gain=.25,
        selected_device='cuda', scope='synthetic receipt only; no NN/GPU test'))
    selection_sha = digest(selection_path)
    protocol = dict(base_protocol, pixel_preprocess_device='cuda', preprocessing_selection_sha256=selection_sha)
    protocol_sha = canonical(protocol)
    write(cache / 'protocol.json', protocol)
    arrays_contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
        source_stat_freeze_sha256=plan['source_stat_freeze_sha256'], split_episodes=counts,
        raw_rows_per_episode=1001, qpos_width=35, qvel_width=32,
        extractor_source_sha256=pins['official_state_2d_cache_v1/produce.py'])
    arrays_protocol = canonical(arrays_contract)
    arrays = dict(protocol_sha256=arrays_protocol, contract=arrays_contract, splits={})
    for split, count in counts.items():
        item = dict(source_npz=sources[split]['path'], accepted_stat=sources[split]['accepted_stat'],
                    rows=count * 1001)
        for key, width in (('qpos', 35), ('qvel', 32)):
            path = shared / (split + '_' + key + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'UNPACKED SOURCE ARRAY IS STAT-ONLY HERE')
            source_stat = stat(path)
            item[key] = dict(path=str(path), shape=[count * 1001, width], dtype='float32',
                stat={key: source_stat[key] for key in ('size', 'mtime_ns', 'inode', 'device')}, sha256=digest(path))
        arrays['splits'][split] = item
    source_arrays_path = shared / 'source_arrays.json'
    write(source_arrays_path, arrays)
    manifest = dict(protocol=protocol, protocol_sha256=protocol_sha,
        base_protocol=base_protocol, base_contract_sha256=canonical(base_protocol), actual_exit_code=0,
        preprocessing_selection=dict(path=str(selection_path), sha256=selection_sha),
        source_arrays_receipt=dict(path=str(source_arrays_path), sha256=digest(source_arrays_path)),
        source_arrays_protocol_sha256=arrays_protocol, splits={})
    for split, count in counts.items():
        manifest['splits'][split] = []
        for episode in range(count):
            start = episode * 1001
            directory = cache / split / f'episode_{episode:06d}'
            directory.mkdir(parents=True)
            latent = directory / 'latent.npy'
            np.save(latent, np.full((1001, 192), episode + .25, np.float32), allow_pickle=False)
            actual_camera = dict(camera)
            actual_camera.pop('base_quaternion')
            actual_camera.update(source_id=0, render_id=1, source_position=camera['position'],
                                 source_quaternion=camera['base_quaternion'],
                                 arm_material_alpha={'robot_material': 1.0})
            receipt = dict(status='episode_committed', actual_exit_code=0, split=split,
                episode=episode, raw_row_start=start, raw_row_count=1001, main_rows=1000,
                terminal_raw_id=start + 1000, canonical_raw_ids_sha256=hashlib.sha256(
                    np.arange(start, start + 1001, dtype='<i8').tobytes()).hexdigest(),
                latent=dict(path=str(latent), bytes=latent.stat().st_size, sha256=digest(latent),
                            shape=[1001, 192], dtype='float32'),
                producer_source_sha256=protocol['producer_source_sha256'],
                render_adapter_source_sha256=protocol['render_adapter_source_sha256'],
                protocol_sha256=protocol_sha, checkpoint_sha256=plan['checkpoint_sha256'],
                pixel_preprocess_device='cuda', preprocessing_selection_sha256=selection_sha,
                preprocess_source_sha256=pins['source/lewm_pcd/utils.py'],
                checkpoint_encoder_sha256=pins['checkpoint_encoder.py'],
                source_arrays_receipt=dict(path=str(source_arrays_path), sha256=digest(source_arrays_path)),
                source_npz=sources[split], camera_contract_sha256=canonical(camera), camera=actual_camera,
                render_geometry=dict(qpos_max_abs_error=0., qvel_max_abs_error=0., nq=35, nv=32, nu=7),
                RGB_sha256='a' * 64, encoder_rows_valid=1001, padding_rows=23,
                encoder_rows_executed=1024)
            receipt_path = directory / 'receipt.json'
            write(receipt_path, receipt)
            manifest['splits'][split].append(dict(episode=episode, receipt_path=str(receipt_path),
                                                   receipt_sha256=digest(receipt_path)))
    write(cache / 'cache_manifest.json', manifest)
    write(cache / 'cache_seal.json', dict(status='producer_pending_outer_audit',
        ready_for_downstream=False, actual_exit_code=0, protocol_sha256=protocol_sha,
        manifest_sha256=digest(cache / 'cache_manifest.json')))
    return cache, plan_path, counts


class AuditContract(unittest.TestCase):
    def auditor(self):
        path = HERE / 'audit_cache.py'
        self.assertTrue(path.is_file(), 'CPU cache auditor implementation is missing')
        spec = importlib.util.spec_from_file_location('cache_auditor_under_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_positive_all_rows_and_provenance_without_reading_old_sources(self):
        auditor = self.auditor()
        with tempfile.TemporaryDirectory() as folder:
            cache, plan, counts = fixture(Path(folder))
            with mock.patch.dict(auditor.EXPECTED_EPISODES, counts, clear=True):
                report = auditor.audit(cache, plan)
            self.assertEqual(report['actual_exit_code'], 0)
            self.assertEqual(report['episodes_verified'], 3)
            self.assertEqual(report['raw_rows_verified'], 3003)
            self.assertFalse(report['old_NPZ_contents_read'])
            self.assertFalse(report['source_arrays_contents_rehashed'])
            self.assertFalse(report['ready_for_downstream'])

    def test_negative_real_latent_byte_change_is_rejected(self):
        auditor = self.auditor()
        with tempfile.TemporaryDirectory() as folder:
            cache, plan, counts = fixture(Path(folder))
            path = cache / 'train/episode_000001/latent.npy'
            with path.open('r+b') as stream:
                stream.seek(-1, 2)
                old = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([old[0] ^ 1]))
            with mock.patch.dict(auditor.EXPECTED_EPISODES, counts, clear=True):
                with self.assertRaisesRegex(AssertionError, 'latent SHA'):
                    auditor.audit(cache, plan)


if __name__ == '__main__':
    unittest.main()
