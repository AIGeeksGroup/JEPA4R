"""Small CPU fixtures for the public official Relocate asset preparer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'jepa4r/data/demos/prepare_official_assets.py'
# SCRIPT moved to jepa4r/data/demos/
CONTENT_CONTRACT = SCRIPT.parents[3] / 'jepa4r/data/relocate/official_contract.py'


def load_preparer():
    assert SCRIPT.is_file(), 'official asset preparation entry is missing'
    spec = importlib.util.spec_from_file_location('prepare_official_assets', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def raw_source(tmp_path):
    demos = []
    for index in range(2):
        actions = np.full((200, 30), 0.25, dtype=np.float32)
        actions[0, :2] = [1.6, -1.7]
        demos.append(dict(
            observations=np.full((200, 1536), index, dtype=np.float32),
            state_obs=np.full((200, 39), index / 10, dtype=np.float64),
            actions=actions,
            rewards=np.arange(200, dtype=np.float64),
            images=np.full((200, 2, 2, 3), index + 17, dtype=np.uint8),
            env_infos=dict(proprio=np.full((200, 30), index, dtype=np.float64),
                           goal_achieved=np.array([np.bool_(False)] * 200, dtype=object)),
            init_state_dict=dict(
                obj_pos=np.array([index / 100, -0.1, 0.03], dtype=object),
                target_pos=np.array([0.1, index / 100, 0.2], dtype=np.float64),
                hand_qpos=np.zeros(30, dtype=object),
                palm_pos=np.array([0.0, 0.0, 0.1], dtype=object),
                qpos=np.zeros(36, dtype=np.float64),
                qvel=np.zeros(36, dtype=np.float32)),
            auxiliary=dict(label='official', frame_ids=np.arange(200, dtype=np.int64))))
    source = tmp_path / 'official.pickle'
    source.write_bytes(pickle.dumps(demos, protocol=4))
    return source, demos


def test_preserves_nonvisual_values_and_binds_lineage(raw_source, tmp_path):
    source, originals = raw_source
    original_bytes = source.read_bytes()
    output = tmp_path / 'prepared'
    result = load_preparer().prepare_assets(source, output, expected_count=2)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert source.read_bytes() == original_bytes
    assert result['source_sha256'] == digest(source) == manifest['source_sha256']
    assert manifest['producer_sha256'] == digest(SCRIPT)
    assert manifest['formal_ready'] is False
    assert [row['demo_index'] for row in manifest['demo_files']] == [0, 1]
    for row, original in zip(manifest['demo_files'], originals):
        path = output / row['relative_path']
        assert path.parent == output / 'demos'
        assert digest(path) == row['sha256'] and path.stat().st_size == row['bytes']
        copied = pickle.loads(path.read_bytes())
        assert set(copied) == set(original) - {'images'}
        for key in copied:
            assert pickle.dumps(copied[key], protocol=4) == pickle.dumps(original[key], protocol=4)
        assert copied['observations'].shape == (200, 1536)
        assert copied['observations'].dtype == np.float32
        assert copied['state_obs'].shape == (200, 39)
        assert copied['state_obs'].dtype == np.float64
        assert copied['actions'][0, 0] > 1 and copied['actions'][0, 1] < -1
    initial = json.loads((output / 'rh56_official_task_init.json').read_text())
    assert initial['schema'] == 'official-task-init-v1'
    assert initial['source_manifest'] == str((output / 'manifest.json').resolve())
    assert initial['source_sha'] == digest(output / 'manifest.json')
    semantic_records = [{key: row[key] for key in
        ('demo_index', 'original_init_sha256', 'obj_pos', 'target_pos')}
        for row in initial['records']]
    semantic_sha = hashlib.sha256(json.dumps(semantic_records, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    assert manifest['initials_semantic_sha256'] == initial['initials_semantic_sha256'] == semantic_sha
    assert manifest['source_raw_sha256'] == digest(source)
    content = load_content_contract()
    assert manifest['policy_critical_content_sha256'] == content.content_digest([
        content.demo_content_record(index, demo) for index, demo in enumerate(originals)])
    assert [r['source_id'] for r in initial['records']] == ['official-relocate-000', 'official-relocate-001']
    for row, original in zip(initial['records'], originals):
        init = original['init_state_dict']
        assert row['original_init_sha256'] == hashlib.sha256(pickle.dumps(init, protocol=4)).hexdigest()
        np.testing.assert_array_equal(row['obj_pos'], init['obj_pos'])
        np.testing.assert_array_equal(row['target_pos'], init['target_pos'])
    assert json.loads((output / 'STATUS.json').read_text())['actual_exit_code'] == 0
    assert (output / 'actual_exit_code').read_text().strip() == '0'


def test_existing_output_is_never_overwritten(raw_source, tmp_path):
    source, _ = raw_source
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('previous evidence')
    with pytest.raises(FileExistsError):
        load_preparer().prepare_assets(source, output, expected_count=2)
    assert sentinel.read_text() == 'previous evidence'
    assert list(output.iterdir()) == [sentinel]


def test_paper_count_guard_keeps_failed_evidence(raw_source, tmp_path):
    source, _ = raw_source
    output = tmp_path / 'count_mismatch'
    with pytest.raises(ValueError, match='100'):
        load_preparer().prepare_assets(source, output)
    assert not (output / 'manifest.json').exists()
    assert (output / 'actual_exit_code').read_text().strip() == '1'


@pytest.mark.parametrize('field', ['actions', 'state_obs', 'observations', 'init_state_dict'])
def test_invalid_contract_is_rejected(raw_source, tmp_path, field):
    source, demos = raw_source
    if field == 'init_state_dict':
        demos[0][field]['obj_pos'] = np.zeros(2)
    else:
        demos[0][field] = demos[0][field][:, :-1]
    source.write_bytes(pickle.dumps(demos, protocol=4))
    output = tmp_path / ('invalid_' + field)
    with pytest.raises(ValueError, match=field if field != 'init_state_dict' else 'obj_pos'):
        load_preparer().prepare_assets(source, output, expected_count=2)
    assert not (output / 'manifest.json').exists()


def test_public_cli_requires_the_paper_count(raw_source, tmp_path):
    source, _ = raw_source
    output = tmp_path / 'not_created'
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--source', str(source),
        '--output', str(output), '--expected-count', '2'],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert '100' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('field', ['rewards', 'env_infos', 'env_infos.proprio',
    'env_infos.goal_achieved', 'init_state_dict.qpos', 'init_state_dict.qvel',
    'init_state_dict.hand_qpos', 'init_state_dict.palm_pos'])
def test_missing_replay_fields_are_rejected(raw_source, tmp_path, field):
    source, demos = raw_source
    pieces = field.split('.')
    owner = demos[0]
    for piece in pieces[:-1]:
        owner = owner[piece]
    del owner[pieces[-1]]
    source.write_bytes(pickle.dumps(demos, protocol=4))
    with pytest.raises(ValueError, match=pieces[-1]):
        load_preparer().prepare_assets(source, tmp_path / 'missing', expected_count=2)


def test_complex_actions_are_rejected(raw_source, tmp_path):
    source, demos = raw_source
    demos[0]['actions'] = demos[0]['actions'].astype(np.complex64) + 0.25j
    source.write_bytes(pickle.dumps(demos, protocol=4))
    with pytest.raises(ValueError, match='actions'):
        load_preparer().prepare_assets(source, tmp_path / 'complex', expected_count=2)


@pytest.mark.parametrize('bad_value', [float('inf'), '0.1'])
def test_invalid_object_state_is_rejected_without_rewriting_it(raw_source, tmp_path, bad_value):
    source, demos = raw_source
    demos[0]['init_state_dict']['qpos'] = np.zeros(36, dtype=object)
    demos[0]['init_state_dict']['qpos'][0] = bad_value
    source.write_bytes(pickle.dumps(demos, protocol=4))
    before = source.read_bytes()
    with pytest.raises(ValueError, match='qpos'):
        load_preparer().prepare_assets(source, tmp_path / 'invalid_state', expected_count=2)
    assert source.read_bytes() == before


def test_initial_semantic_digest_uses_only_four_fields_and_original_order():
    preparer = load_preparer()
    records = [dict(demo_index=index, original_init_sha256=f'{index:064x}',
                    obj_pos=[index / 100, 0.0, 0.03], target_pos=[0.1, 0.1, 0.2],
                    source_id=f'ignored-{index}', ignored_path='/unused/path')
               for index in range(2)]
    expected_records = [{key: row[key] for key in
        ('demo_index', 'original_init_sha256', 'obj_pos', 'target_pos')} for row in records]
    expected = hashlib.sha256(json.dumps(expected_records, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    assert preparer.initials_semantic_sha256(records) == expected
    records[0]['source_id'] = 'metadata-does-not-change-semantics'
    assert preparer.initials_semantic_sha256(records) == expected
    assert preparer.initials_semantic_sha256(records[::-1]) != expected
    records[0]['target_pos'][0] += 0.01
    assert preparer.initials_semantic_sha256(records) != expected


def load_content_contract():
    assert CONTENT_CONTRACT.is_file(), 'shared official content contract is missing'
    spec = importlib.util.spec_from_file_location('official_contract_test_target', CONTENT_CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_content_records(contract, demos):
    records = []
    for index in range(100):
        demo = dict(demos[index % 2])
        demo['init_state_dict'] = dict(demo['init_state_dict'],
            obj_pos=np.array([index / 100, -0.1, 0.03], dtype=object))
        records.append(contract.demo_content_record(index, demo))
    return records


def test_content_record_matches_the_policy_critical_definition(raw_source):
    _, demos = raw_source
    contract = load_content_contract()
    demo = demos[0]
    before = pickle.dumps(demo, protocol=4)
    row = contract.demo_content_record(0, demo)
    assert row['demo_index'] == 0
    assert row['original_init_sha256'] == hashlib.sha256(pickle.dumps(demo['init_state_dict'], protocol=4)).hexdigest()
    values = dict(actions=demo['actions'], state_obs=demo['state_obs'],
                  proprio=demo['env_infos']['proprio'], rewards=demo['rewards'])
    for name, value in values.items():
        array = np.asarray(value)
        assert row['arrays'][name] == dict(shape=list(array.shape), dtype=array.dtype.str,
            sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
    assert pickle.dumps(demo, protocol=4) == before
    assert demo['actions'][0, 0] > 1


def test_content_validation_reports_the_digest(raw_source):
    _, demos = raw_source
    contract = load_content_contract()
    records = toy_content_records(contract, demos)
    report = contract.validate_content_records(records)
    assert report['passed'] is True and report['count'] == 100
    assert report['policy_critical_content_sha256'] == contract.content_digest(records)
    assert report['checker_sha256'] == digest(CONTENT_CONTRACT)
    with pytest.raises(TypeError):
        contract.validate_content_records(records, expected_digest=contract.content_digest(records))


def test_repeated_demo_100_cannot_be_official_content(raw_source):
    _, demos = raw_source
    contract = load_content_contract()
    repeated = [contract.demo_content_record(index, demos[0]) for index in range(100)]
    with pytest.raises(ValueError, match='duplicate|content digest'):
        contract.validate_content_records(repeated)


@pytest.mark.parametrize('field', ['actions', 'state_obs', 'proprio', 'rewards'])
def test_content_builder_rejects_nonfinite_arrays(raw_source, field):
    _, demos = raw_source
    contract = load_content_contract()
    if field == 'proprio':
        demos[0]['env_infos'][field][0, 0] = np.nan
    else:
        demos[0][field].flat[0] = np.inf
    with pytest.raises(ValueError, match=field):
        contract.demo_content_record(0, demos[0])


def test_content_builder_rejects_complex_actions(raw_source):
    _, demos = raw_source
    contract = load_content_contract()
    demos[0]['actions'] = demos[0]['actions'].astype(np.complex64)
    with pytest.raises(ValueError, match='actions'):
        contract.demo_content_record(0, demos[0])


def test_content_schema_is_checked(raw_source):
    _, demos = raw_source
    contract = load_content_contract()
    records = toy_content_records(contract, demos)
    records[0]['arrays']['actions']['dtype'] = '|O'
    with pytest.raises(ValueError, match='dtype'):
        contract.validate_content_records(records)


def test_formal_producer_calls_the_shared_content_validator(raw_source, tmp_path, monkeypatch):
    source, demos = raw_source
    preparer = load_preparer()
    monkeypatch.setattr(preparer, 'OFFICIAL_COUNT', 2)
    called = []
    def reject_fixture(records):
        called.append(records)
        raise ValueError('shared content boundary rejected the fixture')
    monkeypatch.setattr(preparer, 'validate_content_records', reject_fixture)
    output = tmp_path / 'content_gate'
    with pytest.raises(ValueError, match='shared content boundary'):
        preparer.prepare_assets(source, output, expected_count=2)
    assert len(called) == 1 and len(called[0]) == 2
    assert not (output / 'manifest.json').exists()


def test_standalone_preparer_finds_shared_contract_without_pythonpath(tmp_path):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
