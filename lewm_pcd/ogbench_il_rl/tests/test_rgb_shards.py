"""Synthetic CPU episode/atomic boundaries, never a rendered-data claim."""
import hashlib
import zipfile
import h5py
import numpy as np
import pytest

from ogbench_il_rl.data import rgb_shards as core


def test_full_interface_exists():
    assert core is not None, 'Full RGB source/writer interface missing'


def raw_fixture():
    terms = np.zeros(10, dtype=bool)
    terms[[4, 9]] = True
    return dict(observation=np.arange(280, dtype=np.float32).reshape(10, 28),
                action=np.arange(50, dtype=np.float32).reshape(10, 5),
                qpos=np.arange(210, dtype=np.float32).reshape(10, 21),
                qvel=np.arange(200, dtype=np.float32).reshape(10, 20), terminals=terms)


def test_source_episode_global_ids_local_offsets():
    fields = core.episode_fields(raw_fixture(), 1, expected_rows=5)
    assert fields['source_row'].tolist() == list(range(5, 10))
    assert fields['source_episode'].tolist() == [1] * 5
    assert fields['source_step'].tolist() == list(range(5))
    assert fields['source_terminal'].tolist() == [False] * 4 + [True]
    assert not fields['valid_transition'][-1]
    np.testing.assert_array_equal(fields['action'], raw_fixture()['action'][5:10])


def test_atomic_prefix_resume_and_complete_file_protection(tmp_path):
    fields = core.episode_fields(raw_fixture(), 1, expected_rows=5)
    pixels = np.arange(5, dtype=np.uint8)[:, None, None, None] * np.ones((1, 8, 8, 3), np.uint8)
    final = tmp_path / 'ep1.h5'
    writer = core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3), chunk_rows=2)
    writer.append(pixels[:2]); writer.close()
    assert not final.exists()
    writer = core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3), chunk_rows=2)
    assert writer.completed == 2
    writer.append(pixels[2:4]); writer.append(pixels[4:]); receipt = writer.commit()
    assert receipt['rows'] == 5 and receipt['clips'] == 2
    with h5py.File(final) as f:
        assert f['ep_offset'][:].tolist() == [0]
        assert f['ep_len'][:].tolist() == [5]
        np.testing.assert_array_equal(f['pixels'][:], pixels)
        np.testing.assert_array_equal(f['source_row'][:], np.arange(5, 10))
    with pytest.raises(FileExistsError):
        core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3))


@pytest.mark.parametrize('kind', ['protocol', 'source', 'pixel'])
def test_bad_resume_prefix_rejected(tmp_path, kind):
    fields = core.episode_fields(raw_fixture(), 0, expected_rows=5)
    final = tmp_path / 'ep0.h5'
    writer = core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3), chunk_rows=2)
    writer.append(np.zeros((2, 8, 8, 3), np.uint8)); writer.close()
    with h5py.File(final.with_suffix('.partial.h5'), 'r+') as f:
        if kind == 'protocol': f.attrs['protocol_sha256'] = 'b' * 64
        elif kind == 'source': f['action'][0, 0] += 1
        else: f['pixels'][0, 0, 0, 0] = 1
    with pytest.raises(ValueError):
        core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3), chunk_rows=2)


def test_incomplete_episode_cannot_commit(tmp_path):
    fields = core.episode_fields(raw_fixture(), 0, expected_rows=5)
    final = tmp_path / 'ep0.h5'
    writer = core.RGBEpisodeWriter(final, fields, 'a' * 64, rgb_shape=(8, 8, 3), chunk_rows=2)
    writer.append(np.zeros((2, 8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match='Incomplete'):
        writer.commit()
    writer.close()
    assert not final.exists()


def test_original_numeric_cache_member_hash_and_corruption(tmp_path):
    raw = raw_fixture()
    source = tmp_path / 'cube-single-play-v0.npz'
    np.savez_compressed(source, observations=raw['observation'], actions=raw['action'], qpos=raw['qpos'], qvel=raw['qvel'], terminals=raw['terminals'])
    with zipfile.ZipFile(source) as archive:
        fields = {key: {'uncompressed_npy_sha256': hashlib.sha256(archive.read(key)).hexdigest()} for key in archive.namelist()}
    receipt = dict(path=str(source), archive_sha256=core.sha(source), fields=fields)
    cached = core.extract_numeric_source(receipt, tmp_path / 'numeric')
    for key in raw: np.testing.assert_array_equal(cached[key], raw[key])
    del cached
    values = np.load(tmp_path / 'numeric/action.npy', mmap_mode='r+')
    values[0, 0] += 1; values.flush(); del values
    with pytest.raises(ValueError, match='cache SHA'):
        core.extract_numeric_source(receipt, tmp_path / 'numeric')
