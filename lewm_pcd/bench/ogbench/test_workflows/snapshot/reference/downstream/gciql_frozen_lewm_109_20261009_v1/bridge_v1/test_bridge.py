"""CPU gates only: original sampler RNG, boundaries and real raw-row reads."""
import json
from pathlib import Path

import numpy as np
import pytest

from bridge import (OBS_FIELDS, RawRows, agent_config, load_alias, load_compact,
                    make_sampler, preprocess_rows, sample_ids, setup)


@pytest.fixture(scope='module')
def context():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    setup(config)
    compact = load_compact(config)
    return config, compact, agent_config(config['batch_size'])


def rng_equal(a, b):
    assert a[0] == b[0] and a[2:] == b[2:]
    np.testing.assert_array_equal(a[1], b[1])


def compare(context, idxs=None, override=None):
    _, compact, config = context
    config = dict(config, **(override or {}))
    state = make_sampler(compact, config)
    ids = make_sampler(compact, config, ids=True)
    np.random.seed(971)
    rng = np.random.get_state()
    size = len(idxs) if idxs is not None else 4
    state_batch = state.sample(size, idxs=idxs)
    final = np.random.get_state()
    np.random.set_state(rng)
    id_batch = sample_ids(ids, size, idxs=idxs)
    rng_equal(final, np.random.get_state())
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(state_batch[key], compact['observations'][id_batch[key][:, 0]])
        assert id_batch[key].dtype == np.int64 and id_batch[key].shape == (size, 1)
    for key in ('actions', 'rewards', 'masks', 'terminals', 'valids'):
        np.testing.assert_array_equal(state_batch[key], id_batch[key])
    return id_batch


def test_original_sampler_and_rng_parity(context):
    compare(context)  # Initial main.py sample(1) is also checked below.
    _, compact, cfg = context
    ids, state = make_sampler(compact, cfg, True), make_sampler(compact, cfg)
    np.random.seed(0)
    rng = np.random.get_state()
    initial = ids.sample(1)
    final = np.random.get_state()
    np.random.set_state(rng)
    expected = state.sample(1)
    np.testing.assert_array_equal(expected['observations'], compact['observations'][initial['observations'][:, 0]])
    rng_equal(final, np.random.get_state())


def test_declared_config_and_original_augmentation_rng(context):
    from agents.gciql import get_config
    config, compact, cfg = context
    expected = get_config().to_dict()
    expected.update(config['agent_overrides'])
    assert cfg == expected and cfg['p_aug'] == 0.0 and cfg['frame_stack'] is None
    sampler = make_sampler(compact, cfg, True)
    np.random.seed(172)
    saved = np.random.get_state()
    sampler.sample(2)
    final = np.random.get_state()
    np.random.set_state(saved)
    # The original p_aug=0 still consumes one rand outside evaluation mode.
    sampler.sample(2, evaluation=True)
    np.random.rand()
    rng_equal(final, np.random.get_state())


def test_boundary_current_next_and_goals(context):
    _, compact, cfg = context
    last = int(np.flatnonzero(compact['valids'] > 0)[-1])
    current = np.array([0, 998, 999, 1001, last])
    batch = compare(context, current)
    np.testing.assert_array_equal(batch['next_observations'][:, 0], current + 1)
    assert batch['actor_goals'][2, 0] == 999
    assert compact['valids'][batch['value_goals'][:, 0]].all()
    assert compact['valids'][batch['actor_goals'][:, 0]].all()
    with pytest.raises(ValueError, match='terminal'):
        sample_ids(make_sampler(compact, cfg, True), 1, idxs=np.array([1000]))
    forced = compare(context, current, dict(value_p_curgoal=1.0, value_p_trajgoal=0.0, value_p_randomgoal=0.0))
    np.testing.assert_array_equal(forced['value_goals'][:, 0], current)
    np.testing.assert_array_equal(forced['rewards'], np.zeros(len(current)))
    np.testing.assert_array_equal(forced['masks'], np.zeros(len(current)))
    random_goals = compare(context, np.zeros(20, dtype=int),
                           dict(value_p_curgoal=0.0, value_p_trajgoal=0.0, value_p_randomgoal=1.0))
    assert np.any(random_goals['value_goals'][:, 0] >= 1001)


def test_original_random_sampling_with_replacement(context):
    _, compact, cfg = context
    dataset = make_sampler(compact, cfg, True).dataset
    np.random.seed(22)
    expected = dataset.valid_idxs[np.random.randint(len(dataset.valid_idxs), size=10000)]
    np.random.seed(22)
    np.testing.assert_array_equal(dataset.get_random_idxs(10000), expected)
    assert len(np.unique(expected)) < len(expected)


@pytest.mark.parametrize('modality', ['rgb', 'pcd'])
def test_real_reader_unsorted_duplicate_source_and_terminal(context, modality):
    config, compact, _ = context
    reader = RawRows(config, modality, compact)
    ids = np.array([[1001], [999], [0], [1001]], dtype=np.int64)
    result = reader.read(ids, 'observations')
    for value in result.values():
        np.testing.assert_array_equal(value[0], value[3])
    reader.read(np.array([[1000], [2001]], dtype=np.int64), 'next_observations')
    for role in ('observations', 'actor_goals', 'value_goals'):
        with pytest.raises(ValueError, match='terminal'):
            reader.read(np.array([[1000]], dtype=np.int64), role)
    with pytest.raises(ValueError, match='int64'):
        reader.read(np.array([[0]], dtype=np.float32), 'observations')
    # Deliberately wrong source binding must fail at the precise requested row.
    reader.entries[0] = reader.entries[1]
    with pytest.raises(ValueError, match='source_episode'):
        reader.read(np.array([[999]], dtype=np.int64), 'observations')


def test_preprocessing_namespace_stays_official(context):
    import torch
    import sys
    config, _, _ = context
    official = sys.modules['utils']
    path = Path(config['workspace']) / config['repo'] / 'lewm_pcd/utils.py'
    module = load_alias('gciql_test_lewm_preprocessing', path)
    assert sys.modules['utils'] is official
    assert Path(module.__file__).resolve() == path.resolve()
    assert 'official-source/impls/utils' in official.__file__
    rows = RawRows(config, 'rgb', context[1]).read(np.array([[0]], dtype=np.int64), 'observations')
    actual = preprocess_rows(module.get_img_preprocessor('pixels', 'pixels', 224), rows, 'pixels')
    expected = torch.from_numpy(rows['pixels']).permute(0, 3, 1, 2).float() / 255.0
    expected = (expected - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    assert actual.shape == (1, 3, 224, 224) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    rows = RawRows(config, 'pcd', context[1]).read(np.array([[0]], dtype=np.int64), 'observations')
    actual = preprocess_rows(module.get_pcd_preprocessor(), rows, 'pcd')
    expected = torch.cat(((torch.from_numpy(rows['pcd_xyz']) - torch.tensor([.425, 0, .2])) * 4,
                          torch.from_numpy(rows['pcd_rgb']).float() / 255), dim=-1)
    assert actual.shape == (1, 1200, 6) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
