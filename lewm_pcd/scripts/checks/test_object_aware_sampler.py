from pathlib import Path
import sys

import numpy as np
import pytest


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from object_aware_sampler import (  # noqa: E402
    ObjectAwareSamplingError,
    sample_object_aware_point_cloud,
)


def _scene(cube_count=400, robot_count=780, table_count=120):
    total = cube_count + robot_count + table_count
    points = np.zeros((total, 3), dtype=np.float32)
    colors = np.zeros((total, 3), dtype=np.uint8)
    groups = np.zeros(total, dtype=np.int32)

    # Each point is in a distinct 1 cm cell, so the 2.5 mm voxel contract is clear.
    for i in range(total):
        points[i] = [0.01 * (i % 20), 0.01 * ((i // 20) % 20), 0.01 * (i // 400)]

    groups[:cube_count] = 7
    colors[:cube_count] = [220, 40, 40]
    groups[cube_count : cube_count + robot_count] = 10
    colors[cube_count : cube_count + robot_count] = [40, 220, 40]
    groups[cube_count + robot_count :] = 20
    colors[cube_count + robot_count :] = [40, 40, 220]
    return points, colors, groups


def test_object_aware_sampling_is_deterministic_and_meets_lower_bound():
    points, colors, groups = _scene()
    first = sample_object_aware_point_cloud(
        points,
        colors,
        groups,
        tcp_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cube_id=7,
        robot_ids={10},
        table_id=20,
        seed=17,
    )
    second = sample_object_aware_point_cloud(
        points,
        colors,
        groups,
        tcp_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cube_id=7,
        robot_ids={10},
        table_id=20,
        seed=17,
    )

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)

    xyz, rgb, selected_groups = first
    assert xyz.shape == (1200, 3)
    assert rgb.shape == (1200, 3)
    assert selected_groups.shape == (1200,)
    assert np.count_nonzero(selected_groups == 1) >= 300
    assert np.count_nonzero(selected_groups == 2) == 300
    assert np.count_nonzero(selected_groups == 4) <= 120

    voxel_keys = np.floor(xyz / 0.0025).astype(np.int32)
    assert np.unique(voxel_keys, axis=0).shape[0] == 1200


def test_near_tcp_group_is_closer_than_remaining_robot_group():
    points, colors, groups = _scene(cube_count=400, robot_count=780, table_count=120)
    tcp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xyz, _, selected_groups = sample_object_aware_point_cloud(
        points, colors, groups, tcp, cube_id=7, robot_ids={10}, table_id=20, seed=3
    )

    near = np.linalg.norm(xyz[selected_groups == 2] - tcp, axis=1)
    remaining = np.linalg.norm(xyz[selected_groups == 3] - tcp, axis=1)
    assert near.shape == (300,)
    assert remaining.shape == (480,)
    assert near.max() <= remaining.min() + 1e-6


def test_short_robot_pool_uses_extra_unique_cube_points_instead_of_replacement():
    points, colors, groups = _scene(cube_count=900, robot_count=300, table_count=120)
    xyz, _, selected_groups = sample_object_aware_point_cloud(
        points,
        colors,
        groups,
        tcp_xyz=np.zeros(3, dtype=np.float32),
        cube_id=7,
        robot_ids={10},
        table_id=20,
        seed=9,
    )

    assert xyz.shape == (1200, 3)
    assert np.count_nonzero(selected_groups == 1) > 300
    assert np.count_nonzero(selected_groups == 2) == 300
    assert np.count_nonzero(selected_groups == 3) == 0
    assert np.count_nonzero(selected_groups == 4) == 120
    voxel_keys = np.floor(xyz / 0.0025).astype(np.int32)
    assert np.unique(voxel_keys, axis=0).shape[0] == 1200


@pytest.mark.parametrize(
    "cube_count,robot_count,match",
    [(299, 780, "Cube"), (400, 299, "near-TCP")],
)
def test_object_aware_sampling_rejects_missing_mandatory_pool(
    cube_count, robot_count, match
):
    points, colors, groups = _scene(
        cube_count=cube_count, robot_count=robot_count, table_count=120
    )
    with pytest.raises(ObjectAwareSamplingError, match=match):
        sample_object_aware_point_cloud(
            points,
            colors,
            groups,
            tcp_xyz=np.zeros(3, dtype=np.float32),
            cube_id=7,
            robot_ids={10},
            table_id=20,
            seed=1,
        )


def test_object_aware_sampling_rejects_insufficient_total_unique_points():
    points, colors, groups = _scene(cube_count=300, robot_count=300, table_count=120)
    with pytest.raises(ObjectAwareSamplingError, match="1200"):
        sample_object_aware_point_cloud(
            points,
            colors,
            groups,
            tcp_xyz=np.zeros(3, dtype=np.float32),
            cube_id=7,
            robot_ids={10},
            table_id=20,
            seed=1,
        )


def test_online_fallback_can_fill_cube_quota_with_replacement():
    points, colors, groups = _scene(cube_count=120, robot_count=900, table_count=120)

    xyz, _, selected_groups = sample_object_aware_point_cloud(
        points,
        colors,
        groups,
        tcp_xyz=np.zeros(3, dtype=np.float32),
        cube_id=7,
        robot_ids={10},
        table_id=20,
        seed=4,
        allow_cube_replacement=True,
    )

    assert xyz.shape == (1200, 3)
    assert np.count_nonzero(selected_groups == 1) >= 300


def test_voxel_deduplication_prefers_cube_over_other_semantics():
    points, colors, groups = _scene(cube_count=300, robot_count=780, table_count=120)
    # Insert a table observation before the first Cube observation in the same voxel.
    points = np.concatenate([points[-1:], points], axis=0)
    colors = np.concatenate([colors[-1:], colors], axis=0)
    groups = np.concatenate([groups[-1:], groups], axis=0)
    points[0] = points[1]

    _, _, selected_groups = sample_object_aware_point_cloud(
        points,
        colors,
        groups,
        tcp_xyz=np.zeros(3, dtype=np.float32),
        cube_id=7,
        robot_ids={10},
        table_id=20,
        seed=11,
    )

    assert np.count_nonzero(selected_groups == 1) >= 300
