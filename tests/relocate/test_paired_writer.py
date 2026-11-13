# Moved from OGBench-cube@4fbb0c6:tests/test_paired_writer.py (verbatim except: module names / paths of the moved code
# adapted; lines wrapped at 120 columns)
import importlib.util
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location('paired_dependency_under_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_paired_writer_keeps_terminal_action_and_nonvisual_alignment(tmp_path):
    api = module('jepa4r/data/relocate/paired_writer.py')
    episode = {'pixels': np.zeros((3, 3, 4, 4), dtype=np.uint8),
               'pcd_xyz': np.ones((3, 7, 3), dtype=np.float32),
               'pcd_rgb': np.zeros((3, 7, 3), dtype=np.uint8),
               'action': np.array([[.5] * 12, [-.5] * 12, [np.nan] * 12], dtype=np.float32),
               'observation': np.arange(3 * 56, dtype=np.float32).reshape(3, 56)}
    rgb, pcd = tmp_path / 'rgb.h5', tmp_path / 'pcd.h5'
    with api.AdroitPairedWriter(rgb, pcd) as writer:
        writer.write_episode(episode)
        writer.write_episode(episode)
    with h5py.File(rgb) as a, h5py.File(pcd) as b:
        assert 'pcd_xyz' not in a and 'pixels' not in b
        np.testing.assert_array_equal(a['ep_len'][:], [3, 3])
        np.testing.assert_array_equal(a['ep_offset'][:], [0, 3])
        for key in ('action', 'observation', 'ep_len', 'ep_offset'):
            np.testing.assert_array_equal(a[key][:], b[key][:])
        assert np.isnan(a['action'][2]).all()


def test_paired_writer_refuses_to_overwrite_existing_files(tmp_path):
    api = module('jepa4r/data/relocate/paired_writer.py')
    rgb = tmp_path / 'rgb.h5'
    rgb.write_bytes(b'preserved')
    try:
        with api.AdroitPairedWriter(rgb, tmp_path / 'pcd.h5'):
            pass
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing input was overwritten')
    assert rgb.read_bytes() == b'preserved'


def test_metric_depth_projection_keeps_world_axes_and_units():
    api = module('jepa4r/envs/adroit/surface_geometry.py')
    depth = np.full((2, 2), 2., dtype=np.float64)
    points = api.depth_to_world(depth, np.array([1., 2., 3.]), np.eye(3), 90.)
    expected = np.array([[[0., 3., 1.], [2., 3., 1.]],
                         [[0., 1., 1.], [2., 1., 1.]]])
    np.testing.assert_allclose(points, expected, rtol=0, atol=1e-14)
