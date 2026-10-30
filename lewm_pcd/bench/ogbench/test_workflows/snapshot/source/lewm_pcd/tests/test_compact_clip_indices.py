"""Narrow CPU controls for the opt-in compact reader; no model/GPU calls.

Metadata controls load the exact class AST without importing training runtime.
Real source-boundary controls require explicit TRIPLE_REAL_VIEWS after the old
trainers release memory. They build only two-episode, temporary VDS fixtures.
"""
import ast
import itertools
import operator
import os
from pathlib import Path
import pickle
import random
import sys
import types

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
READER = CODE / 'training_hdf5.py'
NODE = next(node for node in ast.parse(READER.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == 'ReadOnlyClipIndex')
MODULE = types.ModuleType('_compact_index_from_exact_source')
MODULE.__dict__.update(np=np, operator=operator)
sys.modules[MODULE.__name__] = MODULE
exec(compile(ast.Module(body=[NODE], type_ignores=[]), str(READER), 'exec'), MODULE.__dict__)
ReadOnlyClipIndex = MODULE.ReadOnlyClipIndex


def test_tuple_compatible_index_and_readonly_storage():
    pairs = [(0, 0), (0, 1), (1, 0), (1, 980)]
    index = ReadOnlyClipIndex(pairs)
    assert len(index) == len(pairs) and list(index) == pairs
    for key in (0, -1, np.int64(2), True, slice(None), slice(None, None, -1), slice(1, 4, 2)):
        assert index[key] == pairs[key]
    assert all(type(value) is int for value in index[0])
    assert index._pairs.dtype == np.int64 and index._pairs.shape == (4, 2)
    assert index._pairs.nbytes == 4 * 16 and not index._pairs.flags.writeable
    with pytest.raises(ValueError): index._pairs[0, 0] = 3
    with pytest.raises(TypeError): index[0] = (1, 2)
    for key in (4, -5):
        with pytest.raises(IndexError): index[key]
    for key in (1.5, [0], np.array([0])):
        with pytest.raises(TypeError): index[key]


def test_small_heterogeneous_lengths_order_and_empty():
    lengths, span = [0, 19, 20, 21, 1000], 20
    old = [(ep, start) for ep, length in enumerate(lengths) if length >= span
           for start in range(length - span + 1)]
    compact = ReadOnlyClipIndex(old)
    assert list(compact) == old and len(compact) == 984
    assert compact[:4] == [(2, 0), (3, 0), (3, 1), (4, 0)]
    assert compact[-8:] == [(4, n) for n in range(973, 981)]
    empty = ReadOnlyClipIndex([])
    assert len(empty) == 0 and empty[:] == [] and empty._pairs.shape == (0, 2)
    with pytest.raises(IndexError): empty[0]
    restored = pickle.loads(pickle.dumps(compact))
    assert list(restored) == old and not restored._pairs.flags.writeable


def test_construction_access_does_not_consume_python_numpy_rng():
    random.seed(123); np.random.seed(456)
    before_python, before_numpy = random.getstate(), np.random.get_state()
    index = ReadOnlyClipIndex([(ep, start) for ep in range(2) for start in range(981)])
    assert index[980] == (0, 980) and index[981] == (1, 0)
    assert index[-8:] == [(1, n) for n in range(973, 981)]
    assert random.getstate() == before_python
    after = np.random.get_state()
    assert before_numpy[0] == after[0] and np.array_equal(before_numpy[1], after[1])
    assert before_numpy[2:] == after[2:]


def test_sampler_full_epoch_next_head_and_capacity_prefix_rng():
    torch = pytest.importorskip('torch')
    old = [(ep, start) for ep in range(2) for start in range(981)]
    compact = ReadOnlyClipIndex(old)
    global_state = torch.random.get_rng_state().clone()
    for count in (len(old), 8 * 128):
        generators = [torch.Generator().manual_seed(0) for _ in range(2)]
        samplers = [torch.utils.data.RandomSampler(source, generator=generator)
                    for source, generator in zip((old, compact), generators)]
        if count == len(old):
            ids = [list(sampler) for sampler in samplers]  # consume StopIteration too
        else:
            ids = [list(itertools.islice(iter(sampler), count)) for sampler in samplers]
        assert ids[0] == ids[1]
        assert [old[n] for n in ids[0]] == [compact[n] for n in ids[1]]
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
        next_heads = [list(itertools.islice(iter(sampler), 128)) for sampler in samplers]
        assert next_heads[0] == next_heads[1]
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize('modality', ['2d', '3d'])
@pytest.mark.parametrize('role', ['train', 'validation'])
def test_real_production_boundary_samples_same_new_reader_flags(tmp_path, modality, role):
    root = os.environ.get('TRIPLE_REAL_VIEWS')
    if not root: pytest.skip('Requires explicit production views after old memory release')
    import hdf5plugin  # noqa: F401 registers source codecs
    import h5py
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(CODE))
    import training_hdf5 as reader
    from stable_worldmodel.data import load_dataset
    assert Path(reader.__file__).resolve() == READER.resolve(), 'Wrong consumer source imported'
    with initialize_config_dir(version_base=None, config_dir=str(CODE / 'config/train')):
        profile = compose(config_name=f'triple_{modality}', overrides=[
            '+data.dataset.compact_clip_indices=true',
            '+data.validation_dataset.compact_clip_indices=true'])
    assert profile.data.dataset.compact_clip_indices is True
    assert profile.data.validation_dataset.compact_clip_indices is True
    config_role = 'dataset' if role == 'train' else 'validation_dataset'
    options = OmegaConf.to_container(profile.data[config_role], resolve=False)
    options.pop('name')
    assert options.pop('compact_clip_indices') is True
    source_path = Path(root) / f'{modality}_{role}_action_rows1000.h5'
    fixture = tmp_path / f'{modality}_{role}.h5'
    with h5py.File(source_path, 'r') as source, h5py.File(fixture, 'x', libver='latest') as out:
        assert source.attrs['training_scope'] == 'production'
        for key in options['keys_to_load']:
            dataset = source[key]
            layout = h5py.VirtualLayout((2000, *dataset.shape[1:]), dtype=dataset.dtype)
            virtual = h5py.VirtualSource(str(source_path), key, shape=dataset.shape)
            for dst, ep in enumerate((0, len(source['ep_len']) - 1)):
                start = int(source['ep_offset'][ep])
                assert int(source['ep_len'][ep]) == 1000
                layout[dst * 1000:(dst + 1) * 1000] = virtual[start:start + 1000]
            out.create_virtual_dataset(key, layout)
        out['ep_len'] = np.full(2, 1000, dtype=np.int32)
        out['ep_offset'] = np.array([0, 1000], dtype=np.int64)
        for key, value in source.attrs.items(): out.attrs[key] = value
        out.attrs['training_scope'] = 'fixture'
    datasets = [load_dataset(str(fixture), transform=None, allow_fixture=True, **options),
                load_dataset(str(fixture), transform=None, allow_fixture=True,
                             compact_clip_indices=True, **options)]
    try:
        plain, compact = datasets
        assert type(plain.clip_indices) is list
        assert isinstance(compact.clip_indices, reader.ReadOnlyClipIndex)
        assert len(plain) == len(compact) == 1962
        assert list(compact.clip_indices) == plain.clip_indices
        for index in [0, 980, 981, *range(1954, 1962)]:
            old, new = plain[index], compact[index]
            assert old.keys() == new.keys()
            assert old['action'].shape == new['action'].shape == (4, 25)
            for key in old: assert torch.equal(old[key], new[key]), (modality, role, index, key)
    finally:
        for dataset in datasets: dataset.close()
