"""Small HDF5 fixtures and static interface checks; no model training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def load_builder():
    path = CODE / 'scripts/triple/build_training_views.py'
    assert path.is_file(), 'Triple contract-driven training view builder is missing'
    spec = importlib.util.spec_from_file_location('triple_view_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, *, env_type='triple', cubes=3, horizon=1000):
    contract = dict(env_type=env_type, physical_cube_count=cubes, horizon_actions=horizon,
                    raw_rows_per_episode=horizon + 1, observation_dim=46, qpos_dim=35,
                    qvel_dim=32, action_dim=5, train_episodes=10000,
                    validation_episodes=1000)
    parts, accepted = [], {}
    rows = horizon + 1
    for split in ('train', 'validation'):
        files = {}
        for suffix in ('pix', 'pcd'):
            path = tmp_path / f'{split}_{suffix}.h5'
            with h5py.File(path, 'x') as file:
                file['ep_len'] = np.array([rows], np.int32)
                file['ep_offset'] = np.array([0], np.int64)
                actions = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5) / 10000
                actions[-1] = np.nan
                file['action'] = actions
                for key, dim in (('observation', 46), ('qpos', 35), ('qvel', 32)):
                    file.create_dataset(key, shape=(rows, dim), dtype='float32')
                file.create_dataset('collection_goal_xyzs', shape=(rows, cubes, 3), dtype='float64')
                if suffix == 'pix':
                    file.create_dataset('pixels', shape=(rows, 224, 224, 3), dtype='uint8')
                else:
                    file.create_dataset('pcd_xyz', shape=(rows, 1200, 3), dtype='float32')
                    file.create_dataset('pcd_rgb', shape=(rows, 1200, 3), dtype='uint8')
            info = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha(path))
            files[str(path)] = dict(info)
            accepted[str(path)] = dict(info)
        parts.append(dict(spec=dict(name=split, split=split, episodes=1), files=files,
                          attempt='attempt_000', split_episode_offset=0, split_row_offset=0,
                          producer_source_freeze_sha256='a' * 64))
    index = dict(contract=contract, source_freeze_sha256='a' * 64,
                 no_physical_merge=True, parts=parts)
    index_path, seal_path = tmp_path / 'index.json', tmp_path / 'seal.json'
    index_path.write_text(json.dumps(index), encoding='utf-8')
    seal = dict(actual_exit_code=0, contract=contract, scope='fixture',
                canonical_index=dict(path=str(index_path), sha256=sha(index_path)),
                source_freeze_sha256='a' * 64, parts=2, episodes=2, raw_rows=2 * rows,
                split_counts=dict(train=1, validation=1), accepted_files=accepted)
    seal_path.write_text(json.dumps(seal), encoding='utf-8')
    return index_path, seal_path


def test_1001_raw_rows_become_1000_action_rows_without_terminal(tmp_path):
    builder = load_builder()
    index, seal = fixture(tmp_path)
    receipt = builder.build(index, seal, tmp_path / 'views', fixture=True)
    assert receipt['window']['starts'] == [0, 980]
    assert receipt['window']['clips_per_episode'] == 981
    assert receipt['scope'] == 'fixture'
    with h5py.File(tmp_path / 'views/2d_train_action_rows1000.h5', 'r') as file:
        assert file['pixels'].is_virtual
        assert file['action'].shape == (1000, 5)
        assert np.isfinite(file['action'][999]).all()
        assert file['action'][999, 0] == pytest.approx(4995 / 10000)
        assert file['ep_len'][0] == 1000
        assert file.attrs['env_type'] == 'triple'
        assert file.attrs['physical_cube_count'] == 3


def test_old_double_identity_is_rejected_before_output(tmp_path):
    builder = load_builder()
    index, seal = fixture(tmp_path, env_type='double', cubes=2, horizon=500)
    with pytest.raises((ValueError, RuntimeError), match='Triple|triple'):
        builder.build(index, seal, tmp_path / 'views', fixture=True)
    assert not (tmp_path / 'views').exists()


def test_fixture_cannot_be_accepted_as_production_quota(tmp_path):
    builder = load_builder()
    index, seal = fixture(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match='production|fixture|quota'):
        builder.build(index, seal, tmp_path / 'views')
    assert not (tmp_path / 'views').exists()


def test_fixture_reader_requires_explicit_smoke_opt_in(tmp_path):
    from triple_training_contract import validate_view_attributes
    builder = load_builder()
    index, seal = fixture(tmp_path)
    builder.build(index, seal, tmp_path / 'views', fixture=True)
    with h5py.File(tmp_path / 'views/2d_train_action_rows1000.h5', 'r') as file:
        with pytest.raises(ValueError, match='fixture'):
            validate_view_attributes(file)
        assert validate_view_attributes(file, allow_fixture=True)['physical_cube_count'] == 3


def test_terminal_and_nonfinite_executed_action_rejected(tmp_path):
    builder = load_builder()
    index, seal = fixture(tmp_path)
    index_data, seal_data = json.loads(index.read_text()), json.loads(seal.read_text())
    path = Path(next(p for p in index_data['parts'][0]['files'] if p.endswith('_pix.h5')))
    with h5py.File(path, 'r+') as file:
        file['action'][999] = np.nan
    info = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha(path))
    index_data['parts'][0]['files'][str(path)] = info
    index.write_text(json.dumps(index_data), encoding='utf-8')
    seal_data['accepted_files'][str(path)] = info
    seal_data['canonical_index']['sha256'] = sha(index)
    seal.write_text(json.dumps(seal_data), encoding='utf-8')
    with pytest.raises((ValueError, RuntimeError), match='action'):
        builder.build(index, seal, tmp_path / 'views', fixture=True)


def test_forward_bytes_match_accepted_source():
    portable = CODE / 'train.py'
    assert portable.is_file(), 'portable training entry is missing'
    manifest = json.loads((CODE / 'training_source_provenance.json').read_text())
    tree = ast.parse(portable.read_text(encoding='utf-8'))
    forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lejepa_forward')
    text = ''.join(portable.read_text(encoding='utf-8').splitlines(keepends=True)[forward.lineno - 1:forward.end_lineno])
    assert hashlib.sha256(text.encode()).hexdigest() == manifest['lejepa_forward_sha256']


def test_resume_requires_same_triple_model_data_binding():
    import triple_training_contract as contract
    assert hasattr(contract, 'validate_resume_binding'), 'bound standard Triple resume interface is missing'
    expected = dict(contract=contract.EXPECTED, model={'encoder': 'original'}, views={'train': 'abc'}, optimizer={'lr': 5e-5})
    contract.validate_resume_binding({'triple_training_binding': expected, 'state_dict': {}, 'optimizer_states': [{}]}, expected)
    with pytest.raises(ValueError, match='Triple|binding'):
        contract.validate_resume_binding({'state_dict': {}, 'optimizer_states': [{}]}, expected)
    wrong = json.loads(json.dumps(expected))
    wrong['contract']['env_type'] = 'double'
    with pytest.raises(ValueError, match='Triple|binding'):
        contract.validate_resume_binding({'triple_training_binding': wrong, 'state_dict': {}, 'optimizer_states': [{}]}, expected)


def test_profiles_keep_parent_math_and_small_worker_defaults():
    import yaml
    provenance = json.loads((CODE / 'training_source_provenance.json').read_text())
    assert 'model_profile_sha256' in provenance, 'parent model/math profile hashes are missing'
    for modality in ('2d', '3d'):
        cfg = yaml.safe_load((CODE / f'config/train/triple_{modality}.yaml').read_text())
        selected = {key: cfg[key] for key in ('model', 'optimizer', 'loss', 'history_size', 'num_preds', 'embed_dim', 'pcd_in_dim', 'img_size')}
        assert hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest() == provenance['model_profile_sha256'][modality]
        assert cfg['num_workers'] == 2 and cfg['validation_num_workers'] == 0
        assert cfg['resume_checkpoint'] is None
        assert cfg['data']['dataset']['name'].startswith('${oc.env:TRIPLE_ROOT}/data/views/')
        assert cfg['data']['validation_dataset']['num_steps'] == 4
