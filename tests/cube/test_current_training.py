"""Current paper training contracts."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

LEWM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEWM))


# the packages the split placed them in
PACKAGES = {'fresh_data': 'jepa4r.data.cube', 'fresh_train': 'jepa4r.models.cube',
            'training_runtime': 'jepa4r.models.cube', 'pldm.shared_data': 'jepa4r.baselines.cube'}


def api(name):
    prefix = PACKAGES.get(name, 'jepa4r.policy.gciql')
    module = prefix + '.' + name
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, 'clean current training projection is missing'
    return importlib.import_module(module)


def bind(tmp_path, cubes):
    data = api('fresh_data')
    horizon = {1: 200, 2: 500, 3: 1000, 4: 600}[cubes]
    producer = tmp_path / 'producer.py'
    producer.write_text('# current fixture producer\n')
    value = dict(schema='cube_fixed_collection_v1', cube_count=cubes, mode='smoke', raw_horizon=horizon,
                 train_episodes=2, validation_episodes=1, split_counts={'train': 2, 'validation': 1},
                 actual_exit_code=0, paths={}, protocol='paper_cube_fixed_mix_v1', collection_seed=0,
                 collection_protocol={'name': 'paper_cube_fixed_mix_v1'},
                 producer_binding={'files': {str(producer): data.sha256(producer)}})
    for split, episodes in [('train', 2), ('validation', 1)]:
        rows = episodes * (horizon + 1)
        action = np.zeros((rows, 5), np.float32)
        action[np.arange(episodes) * (horizon + 1) + horizon] = np.nan
        for mode in ('rgb', 'pcd'):
            path = tmp_path / f'{split}_{mode}.h5'
            with h5py.File(path, 'x', libver='latest') as f:
                f['action'] = action
                f['ep_len'] = np.full(episodes, horizon + 1, np.int32)
                f['ep_offset'] = np.arange(episodes, dtype=np.int64) * (horizon + 1)
                for key, width in [('observation', 19 + 9 * cubes), ('qpos', 14 + 7 * cubes), ('qvel', 14 + 6 * cubes)]:
                    f.create_dataset(key, shape=(rows, width), dtype='f8')
                if mode == 'rgb':
                    f.create_dataset('pixels', shape=(rows, 224, 224, 3), dtype='u1')
                else:
                    f.create_dataset('pcd_xyz', shape=(rows, 1200, 3), dtype='f4')
                    f.create_dataset('pcd_rgb', shape=(rows, 1200, 3), dtype='u1')
            value['paths'][split + '_' + mode] = str(path)
    collected = tmp_path / 'COLLECTION.json'
    collected.write_text(json.dumps(value))
    return data.create_binding(collected, tmp_path / 'data')


@pytest.mark.parametrize('cubes,horizon', [(1, 200), (2, 500), (3, 1000), (4, 600)])
def test_all_cubes_use_all_action_rows_and_the_same_heldout_seed0_recipe(tmp_path, cubes, horizon):
    manifest = bind(tmp_path, cubes)
    data, train = api('fresh_data'), api('fresh_train')
    bound = data.load_binding(manifest)
    assert bound['training_rows_per_episode'] == horizon
    assert bound['training_seed'] == 0
    for row in bound['views'].values():
        with h5py.File(row['path'], 'r') as f:
            assert np.all(f['ep_len'][:] == horizon)
            assert np.isfinite(f['action'][:]).all()
    for method in ('lewm_rgb', 'lewm_pcd'):
        plan = train.build_plan(cubes, method, manifest, tmp_path / method, mode='smoke')
        cfg = plan['config']
        assert cfg['seed'] == cfg['fresh_seed'] == 0
        assert 'train_split' not in cfg
        assert cfg['loader']['batch_size'] == 128
        assert cfg['trainer']['precision'] == 'bf16-mixed'
        assert cfg['optimizer'] == {'type': 'AdamW', 'lr': 5e-5, 'weight_decay': .001}
        assert cfg['loss']['sigreg']['weight'] == .09
        assert cfg['data']['dataset']['num_steps'] == 4
        assert cfg['data']['dataset']['frameskip'] == 5
        assert cfg['data']['dataset']['name'] != cfg['data']['validation_dataset']['name']
        assert plan['protocol'] == 'episode_heldout_v1'
        assert not (tmp_path / method).exists()


def test_train_modes_and_cube_counts_are_explicit(tmp_path):
    manifest = bind(tmp_path, 1)
    train = api('fresh_train')
    with pytest.raises(ValueError, match='mode'):
        train.build_plan(1, 'pldm', manifest, tmp_path / 'new', mode='formal')
    with pytest.raises(ValueError, match='Cube'):
        train.build_plan(2, 'pldm', manifest, tmp_path / 'new', mode='smoke')
    tiny = train.build_plan(1, 'lejepa', manifest, tmp_path / 'tiny', mode='smoke')
    assert tiny['recipe']['epochs'] == 10 and tiny['mode'] == 'smoke'
    pldm = api('pldm.shared_data')
    assert pldm.contract('cube1', data_binding=manifest)['train_samples'] == 2 * 185
    with pytest.raises(ValueError):
        pldm.contract('cube1')


@pytest.mark.parametrize(
    'field,value', [('protocol', 'other'), ('collection_seed', 3072), ('collection_protocol', {'name': 'other'})]
)
def test_binding_rejects_noncurrent_collection_contract(tmp_path, field, value):
    bind(tmp_path, 1)
    data = api('fresh_data')
    source = tmp_path / 'COLLECTION.json'
    collected = json.loads(source.read_text())
    collected[field] = value
    source.write_text(json.dumps(collected))
    with pytest.raises(ValueError, match='protocol|seed'):
        data.create_binding(source, tmp_path / 'rejected')
