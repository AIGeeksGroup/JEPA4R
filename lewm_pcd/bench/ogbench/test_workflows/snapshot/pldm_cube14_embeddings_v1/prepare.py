"""Bind two existing official datasets/assets to the unchanged cache pipeline."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'pldm_cube14_embeddings_v1'
BASE = ROOT / 'foundation_visual_gciql_v1'
STORE = Path('/autodl-fs/data/lewm/pldm_cube14_embeddings_20261025_v1')
PY = ROOT / 'env_runtime/bin/python'
ASSETS = {
    1: ('cube1_pldm_impala512_epoch10_20261023_v1', '238c3af486097fe8875ad1fc0f4081d3e322b7e9527d5bf0268f99046fd21ae8', 'a14eaa740ff31d5de07af3f524f0200313a5a95de483de9440cfc8591e5e59f6', None),
    4: ('cube4_pldm_impala512_epoch10_20261023_v1', '1f803133971de183be74b4f3e0734b6a9bb1060a4f342f2537129474bcef4b8f', 'c0c7b8cb715309c564f96fd98e335b3a3e69cebccf210d98dc2608875671457e', 'c79118167f780fe626e5e9adb8e806e8920f8033e815f6f2a277119efc52ee87'),
}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare():
    assert not (CASE / 'groups').exists() and not STORE.exists(), 'Fresh namespaces required'
    payload, headers, metadata, reserve = 13530316800, 6600 * 128, 256 << 20, 2 << 30
    free = shutil.disk_usage(STORE.parent).free
    assert free >= payload + headers + metadata + reserve, (free, payload + headers + metadata + reserve)
    STORE.mkdir()
    probe = STORE / 'storage_write_probe.bin'
    with probe.open('xb') as f:
        f.write(b'PLDM14_STORAGE_PROBE\n' * 1024)
        f.flush()
        os.fsync(f.fileno())
    save(CASE / 'STORAGE_ADMISSION.json', dict(actual_exit_code=0, at=time.time(), store=str(STORE), free_before=free,
         feature_payload_bytes=payload, headers_bytes=headers, metadata_budget_bytes=metadata, reserve_bytes=reserve,
         small_write_readback_SHA256=sha(probe), filesystem_free_not_user_quota=True))
    (STORE / 'control').mkdir()
    (CASE / 'control').symlink_to(STORE / 'control', target_is_directory=True)
    common = ['producer_bridge.py', 'audit_bridge.py', 'render_adapter.py', 'common.py', 'deliver.py',
              'base/produce.py', 'base/audit_cache.py', 'base/render_adapter.py']
    pins = {str((BASE / 'cache' / f).relative_to(ROOT)): sha(BASE / 'cache' / f) for f in common}
    for rel in ['foundation_visual_gciql_v1/control/serial.py', 'foundation_visual_gciql_v1/control/barrier.py',
                'cube2_official_embeddings_v1/serial_relay.py', 'cube4_visual_gciql_v1/control_v1/gpu_lease.py',
                'prepare_runtime.py']:
        pins[rel] = sha(ROOT / rel)
    for f in ['cache_bridge.py', 'prepare.py', 'run.py', 'run.sh']:
        pins[str((CASE / f).relative_to(ROOT))] = sha(CASE / f)
    original_adapter = ROOT / 'cube2_pldm_gciql_v1/cache_v1/encoder_adapter.py'
    pins[str(original_adapter.relative_to(ROOT))] = sha(original_adapter)
    plans = []
    for n, (asset_id, weight_sha, manifest_sha, addon_sha) in ASSETS.items():
        old = BASE / 'cache/w8_groups' / f'visreg_vitb768_cube{n}'
        plan = copy.deepcopy(read(old / 'plan.json'))
        arrays = Path(plan['shared_state_root']) / 'source_arrays.json'
        assert sha(arrays) == plan['source_arrays_receipt_sha256']
        record = read(arrays)
        for item in record['splits'].values():
            for p, expected in [(item['source_npz'], item['accepted_stat'])] + [(item[k]['path'], item[k]['stat']) for k in ('qpos', 'qvel')]:
                s = Path(p).stat()
                actual = dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, device=s.st_dev, inode=s.st_ino)
                assert actual == expected, p
        for key, value in plan['source_pins'].items():
            if key.startswith('source/ogbench/'):
                pins[key] = value
        asset = ROOT / 'assets/baselines' / f'cube{n}' / asset_id
        assert sha(asset / 'MANIFEST.json') == manifest_sha
        manifest = read(asset / 'MANIFEST.json')
        assert manifest['asset_id'] == asset_id and manifest['embedding_dim'] == 512 and manifest['epochs'] == 10
        checkpoint = asset / 'weights/weights_epoch_10.pt'
        assert sha(checkpoint) == weight_sha
        for name, row in manifest['members'].items():
            if Path(name).suffix in ('.py', '.json', '.yaml', '.yml', '.toml'):
                pins[str((asset / name).relative_to(ROOT))] = row['sha256']
        pins[str((asset / 'MANIFEST.json').relative_to(ROOT))] = manifest_sha
        if addon_sha:
            assert sha(asset / 'RUNTIME_ADDENDUM.json') == addon_sha
            pins[str((asset / 'RUNTIME_ADDENDUM.json').relative_to(ROOT))] = addon_sha
            for name, value in read(asset / 'RUNTIME_ADDENDUM.json')['files'].items():
                if name.endswith('.py'):
                    pins[str((asset / name).relative_to(ROOT))] = value
        group = CASE / 'groups' / f'pldm_cube{n}'
        spec = dict(schema_version=1, family='pldm', cube_count=n, asset_id=asset_id, asset_root=str(asset),
                    manifest_sha256=manifest_sha, checkpoint=str(checkpoint), checkpoint_sha256=weight_sha,
                    runtime_addendum_sha256=addon_sha, output_dim=512, inference_precision='fp32', preprocess_device='cpu')
        save(group / 'encoder_spec.json', spec)
        plan.update(encoder_family='pldm', latent_dim=512, checkpoint=str(checkpoint), checkpoint_sha256=weight_sha,
                    encoder_spec=str(group / 'encoder_spec.json'), encoder_spec_sha256=sha(group / 'encoder_spec.json'),
                    encoder_adapter=str(CASE / 'cache_bridge.py'), encoder_adapter_sha256=sha(CASE / 'cache_bridge.py'),
                    renderer_workers=2, encoder_batch=32, torch_cpu_threads=1, prefetch=2,
                    pixel_preprocess_device='shared_cpu', encoder_preprocessing_device='cpu', host_tensor_dtype='uint8',
                    pixel_preprocess_policy='original PLDM CPU preprocessing; CUDA FP32 encoder',
                    inference_precision='fp32', storage_dtype='float32',
                    encoder_preprocess=dict(input_dtype='uint8', input_shape=[3, 224, 224], output_dtype='float32',
                       output_shape=[3, 65, 65], divisor=255., antialias=True, ImageNet_normalization=False,
                       history_duplication=False, Native_B1_alignment_required=True),
                    cache_dir=str(STORE / 'groups' / group.name / 'cache'), source_pins={}, training_updates=0, SR_episodes=0)
        plans.append((group, old, plan))
    for key, value in pins.items():
        assert sha(ROOT / key) == value, key
    for group, old, plan in plans:
        plan['source_pins'] = dict(pins)
        plan['source_pins'][str((group / 'encoder_spec.json').relative_to(ROOT))] = plan['encoder_spec_sha256']
        save(group / 'plan.json', plan)
        save(group / 'data_binding.json', read(old / 'data_binding.json'))
        save(group / 'camera.json', plan['camera'])
    original = read(BASE / 'control/fp16_bulk_w8_v1/launch_plan.json')
    env = original['phases'][0]['environment']
    for queue in ['pilot', 'full']:
        directory = CASE / 'control' / queue
        phases = []
        for group, _, plan in plans:
            base = [str(PY), '-B', str(CASE / 'cache_bridge.py'), '--mode', 'producer', '--plan', str(group / 'plan.json'), '--execute']
            argv = [str(PY), '-B', str(ROOT / 'cube4_visual_gciql_v1/control_v1/gpu_lease.py'), '--'] + base
            phases.append(dict(name=group.name + ('_00_pilot' if queue == 'pilot' else '_01_encode_full'), argv=argv + (['--pilot'] if queue == 'pilot' else []), cwd=str(ROOT), environment=env))
            if queue == 'full':
                audit = group / 'full_audit_v1'
                phases.append(dict(name=group.name + '_02_cache_audit', argv=[str(PY), '-B', str(CASE / 'cache_bridge.py'), '--mode', 'audit', '--plan', str(group / 'plan.json'), '--cache-dir', plan['cache_dir'], '--output-dir', str(audit)], cwd=str(ROOT), environment=env))
                phases.append(dict(name=group.name + '_03_deliver', argv=[str(PY), '-B', str(BASE / 'cache/deliver.py'), '--plan', str(group / 'plan.json'), '--audit-report', str(audit / 'report.json'), '--producer-exit-file', str(directory / 'runtime' / (group.name + '_01_encode_full_exit.json'))], cwd=str(ROOT), environment=env))
        allpins = {str(ROOT / key): value for key, value in pins.items()}
        for group, _, _ in plans:
            for name in ['plan.json', 'encoder_spec.json', 'data_binding.json', 'camera.json']:
                allpins[str(group / name)] = sha(group / name)
        save(directory / 'launch_plan.json', dict(schema_version=1, queue='encoding_only', mode=queue, families=['pldm'],
             cube_counts=[1, 4], expected_features=4004 if queue == 'pilot' else 6606600, training_updates=0, SR_episodes=0,
             process_guard=str(ROOT / 'prepare_runtime.py'), own_RSS_limit_bytes=60 << 30, phases=phases, source_pins=allpins))
    save(CASE / 'PREPARED.json', dict(actual_exit_code=0, at=time.time(), groups=[p[0].name for p in plans],
         original_sources_modified=False, source_arrays_reextracted=False, NN_started=False, full_encoding_started=False,
         original_PLDM4_training_camera_exact_parameters='not established; use locked official downstream camera'))
    print(json.dumps(dict(actual_exit_code=0, groups=2, source_pins=len(pins), store=str(STORE))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    if parser.parse_args().execute:
        prepare()
    else:
        print(json.dumps(dict(prepared_only=True, NN_started=False)))
