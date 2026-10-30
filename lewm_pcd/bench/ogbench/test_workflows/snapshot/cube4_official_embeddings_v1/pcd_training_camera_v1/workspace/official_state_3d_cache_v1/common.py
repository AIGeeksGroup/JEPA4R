"""Reuse frozen 2D cache IO helpers; original files and source arrays stay immutable."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy(workspace):
    directory = Path(workspace) / 'official_state_2d_cache_v1'
    sys.path.insert(0, str(directory))
    try:
        return load_source('_frozen_2d_cache_io', directory / 'produce.py')
    finally:
        sys.path.pop(0)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def episode_seed(split, episode):
    """New derived PCD seed: original official NPZ has no point-sampling seed."""
    assert split in ('train', 'val') and isinstance(episode, int) and episode >= 0
    payload = json.dumps(['quadruple_official_pcd_epoch10_v1', split, episode],
                         separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def load_arrays(plan, old, sha, stat):
    """Adopt accepted old extraction with its actual producer identity; no re-extract/hash."""
    path = Path(plan['shared_state_root']) / 'source_arrays.json'
    assert sha(path) == plan['source_arrays_receipt_sha256']
    record = json.loads(path.read_text())
    old_plan = json.loads((Path(plan['workspace']) / 'official_state_2d_cache_v1/producer_plan.json').read_text())
    contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
        source_stat_freeze_sha256=plan['source_stat_freeze_sha256'],
        split_episodes=plan['split_episodes'], raw_rows_per_episode=1001,
        qpos_width=42, qvel_width=38,
        extractor_source_sha256=old_plan['source_pins']['official_state_2d_cache_v1/produce.py'])
    assert record['contract'] == contract and record['protocol_sha256'] == object_sha(contract)
    frozen = json.loads(Path(plan['source_stat_freeze']).read_text())
    for split, item in record['splits'].items():
        prior = frozen['datasets']['validation' if split == 'val' else 'train']
        assert item['source_npz'] == prior['path']
        assert old.raw_stat(item['source_npz']) == item['accepted_stat'] == prior['accepted_stat']
        for key in ('qpos', 'qvel'):
            assert stat(item[key]['path']) == item[key]['stat']
    return path, record


def protocol_payload(plan):
    return {key: plan[key] for key in (
        'checkpoint_sha256', 'data_acceptance_sha256', 'source_stat_freeze_sha256',
        'source_arrays_receipt_sha256', 'source_pins', 'camera', 'pcd', 'split_episodes',
        'raw_rows_per_episode', 'encoder_batch', 'renderer_workers', 'preprocess', 'precision')}


def validate_plan(plan, sha):
    assert plan['split_episodes'] == {'train': 5000, 'val': 500}
    assert plan['raw_rows_per_episode'] == 1001 and plan['encoder_batch'] == 32
    assert plan['renderer_workers'] == 2
    for key in ('checkpoint', 'data_acceptance', 'source_stat_freeze'):
        assert sha(plan[key]) == plan[key + '_sha256'], key
    root = Path(plan['workspace'])
    for relative, expected in plan['source_pins'].items():
        assert sha(root / relative) == expected, relative
    assert plan['pcd'] == dict(points=1200, channels=6, crop_box=[[-.25, -.5, -.02], [.75, .5, .9]],
        voxel_size=.005, cube_fraction=.25, foreground_only=True, geometry_offsamples=0,
        sampling_namespace='quadruple_official_pcd_epoch10_v1',
        sampling_policy='SHA256([namespace,split,episode]) first4bytes little uint32; continuous1001 current captures')
    assert plan['preprocess'] == dict(device='cpu', center=[.425, 0., .2], scale=4.,
                                      use_rgb=True, original_factory=True)
    assert plan['precision'] == dict(dtype='float32', AMP=False, change_original_settings=False)
