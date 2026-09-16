from pathlib import Path
import sys

import numpy as np
import pytest


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_object_aware_render import (  # noqa: E402
    camera_observation_to_world,
    default_camera_specs,
)


def test_default_camera_protocol_has_base_plus_five_added_views():
    specs = default_camera_specs(extra_resolution=256)
    assert [spec.uid for spec in specs] == [
        "base_camera",
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
        "top",
    ]
    assert specs[0].width == specs[0].height == 128
    assert all(spec.width == spec.height == 256 for spec in specs[1:])
    assert all(spec.near == 0.01 and spec.far == 100.0 for spec in specs)


def test_camera_observation_conversion_returns_world_points_and_semantic_ids():
    position_mm = np.array(
        [[[[1000, 2000, 3000], [4000, 5000, 6000]]]], dtype=np.int16
    )
    rgb = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.uint8)
    segmentation = np.array([[[[7], [0]]]], dtype=np.int16)
    cam2world = np.eye(4, dtype=np.float32)[None]

    xyz, colors, semantic = camera_observation_to_world(
        position_mm, rgb, segmentation, cam2world
    )

    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(colors[0], [1, 2, 3])
    np.testing.assert_array_equal(semantic, [7])


def test_camera_observation_conversion_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        camera_observation_to_world(
            np.zeros((1, 1, 1, 3), dtype=np.int16),
            np.zeros((1, 1, 2, 3), dtype=np.uint8),
            np.zeros((1, 1, 1, 1), dtype=np.int16),
            np.eye(4, dtype=np.float32)[None],
        )
