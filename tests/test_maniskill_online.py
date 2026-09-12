from __future__ import annotations

import numpy as np
import pytest

from data.maniskill_online import preprocess_online_cloud


def _cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71)
    scene_xyz = rng.uniform(
        low=[-0.70, -0.55, 0.00], high=[0.70, 0.55, 0.70], size=(1400, 3)
    ).astype(np.float32)
    robot_xyz = rng.uniform(
        low=[-0.30, -0.30, 0.10], high=[0.30, 0.30, 0.50], size=(80, 3)
    ).astype(np.float32)
    rejected_xyz = np.array(
        [[2.0, 0.0, 0.1], [0.0, 0.0, np.nan], [0.0, 0.0, 0.1]], dtype=np.float32
    )
    xyzw = np.concatenate(
        [
            np.column_stack([scene_xyz, np.ones(len(scene_xyz), dtype=np.float32)]),
            np.column_stack([robot_xyz, np.ones(len(robot_xyz), dtype=np.float32)]),
            np.column_stack([rejected_xyz, np.array([1.0, 1.0, 0.0], dtype=np.float32)]),
        ]
    )
    robot_seg = np.zeros((len(xyzw), 1), dtype=bool)
    robot_seg[len(scene_xyz) : len(scene_xyz) + len(robot_xyz)] = True
    return xyzw, robot_seg, robot_xyz


def test_online_cloud_filters_robot_and_invalid_points_deterministically() -> None:
    xyzw, robot_seg, robot_xyz = _cloud()

    first = preprocess_online_cloud(
        xyzw, robot_seg, sample_key=4100, point_count=512
    )
    second = preprocess_online_cloud(
        xyzw, robot_seg, sample_key=4100, point_count=512
    )

    assert first.shape == (512, 3)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert not np.any(np.all(first[:, None, :] == robot_xyz[None, :, :], axis=-1))
    assert np.logical_and(
        first >= np.array([-0.75, -0.60, -0.05], dtype=np.float32),
        first <= np.array([0.75, 0.60, 0.80], dtype=np.float32),
    ).all()


def test_online_cloud_returns_rgb_aligned_with_sampled_xyz() -> None:
    xyzw, robot_seg, _ = _cloud()
    rgb = np.arange(len(xyzw) * 3, dtype=np.uint8).reshape(len(xyzw), 3)

    first = preprocess_online_cloud(
        xyzw, robot_seg, sample_key=4100, point_count=512, rgb=rgb
    )
    second = preprocess_online_cloud(
        xyzw, robot_seg, sample_key=4100, point_count=512, rgb=rgb
    )

    assert first.shape == (512, 6)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first[:, 3:], rgb[np.isin(xyzw[:, 0], first[:, 0])][:, :3] / 255.0,
        atol=1 / 255.0,
    )


def test_online_cloud_rejects_misaligned_rgb() -> None:
    xyzw, robot_seg, _ = _cloud()
    with pytest.raises(ValueError, match="rgb"):
        preprocess_online_cloud(
            xyzw,
            robot_seg,
            sample_key=4100,
            point_count=512,
            rgb=np.zeros((len(xyzw) - 1, 3), dtype=np.uint8),
        )


def test_online_cloud_sample_key_changes_the_sample() -> None:
    xyzw, robot_seg, _ = _cloud()
    first = preprocess_online_cloud(xyzw, robot_seg, sample_key=10, point_count=512)
    second = preprocess_online_cloud(xyzw, robot_seg, sample_key=11, point_count=512)
    assert not np.array_equal(first, second)


def test_online_cloud_rejects_too_few_valid_scene_points() -> None:
    xyzw, robot_seg, _ = _cloud()
    robot_seg[:500] = True
    with pytest.raises(ValueError, match="1024"):
        preprocess_online_cloud(xyzw, robot_seg, sample_key=0, point_count=512)


@pytest.mark.parametrize(
    ("xyzw", "robot_seg", "match"),
    [
        (np.zeros((8, 3), dtype=np.float32), np.zeros(8, dtype=bool), "xyzw"),
        (np.zeros((8, 4), dtype=np.float32), np.zeros((8, 2), dtype=bool), "robot_seg"),
    ],
)
def test_online_cloud_rejects_invalid_shapes(xyzw, robot_seg, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        preprocess_online_cloud(xyzw, robot_seg, sample_key=0, point_count=4)
