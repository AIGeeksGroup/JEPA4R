import numpy as np

from bench.cortexbench.adroit.pcd import AdroitPCDConfig, depth_to_world, sample_adroit_pcd


def test_depth_to_world_uses_mujoco_camera_axes():
    depth = np.ones((2, 2), dtype=np.float32)
    xyz = depth_to_world(depth, np.zeros(3), np.eye(3), 90.0)
    np.testing.assert_allclose(xyz[0, 0], [-0.5, 0.5, -1.0], atol=1e-6)
    np.testing.assert_allclose(xyz[1, 1], [0.5, -0.5, -1.0], atol=1e-6)


def test_sample_adroit_pcd_returns_fixed_shape_and_object_share():
    xyz = np.array([[i / 4000.0, 0.0, 0.1] for i in range(2000)], dtype=np.float32)
    rgb = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (2000, 1))
    geom_ids = np.zeros(2000, dtype=np.int64)
    geom_ids[500:] = 1
    foreground = np.array([True, True])
    objects = np.array([False, True])
    out_xyz, out_rgb, stats = sample_adroit_pcd(
        xyz, rgb, geom_ids, foreground, objects, np.random.default_rng(7), AdroitPCDConfig(num_points=1200, voxel_size=0.0)
    )
    assert out_xyz.shape == (1200, 3)
    assert out_rgb.shape == (1200, 3)
    assert out_xyz.dtype == np.float32
    assert out_rgb.dtype == np.uint8
    assert 0.20 <= stats["object_fraction"] <= 0.30
    assert np.isfinite(out_xyz).all()


def test_sample_adroit_pcd_rejects_empty_foreground():
    try:
        sample_adroit_pcd(
            np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8), np.zeros(2, dtype=np.int64),
            np.array([False]), np.array([False]), np.random.default_rng(0)
        )
    except ValueError as exc:
        assert "no finite foreground" in str(exc)
    else:
        raise AssertionError("empty foreground must fail")


def test_sample_adroit_pcd_allows_zero_voxel_size_without_division():
    xyz = np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1]], dtype=np.float32)
    rgb = np.zeros((2, 3), dtype=np.uint8)
    out_xyz, _, _ = sample_adroit_pcd(
        xyz,
        rgb,
        np.zeros(2, dtype=np.int64),
        np.array([True]),
        np.array([False]),
        np.random.default_rng(0),
        AdroitPCDConfig(num_points=4, voxel_size=0.0),
    )
    assert out_xyz.shape == (4, 3)
