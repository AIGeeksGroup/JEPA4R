import numpy as np
import pytest

from visual_sr import (
    CAMERA,
    camera_for_group,
    goal_xyz,
    make_policy_inputs,
    pcd_components,
    seed_for,
    validate_latent,
)


def test_visual_policy_contract_has_expected_widths_and_no_object_leakage():
    state = np.arange(37, dtype=np.float32)
    latent = np.ones(192, dtype=np.float32)
    current, goal = make_policy_inputs(latent, latent * 2, state, state)
    assert current.shape == (211,)
    assert goal.shape == (198,)
    np.testing.assert_array_equal(current[192:], state[:19])
    np.testing.assert_array_equal(goal[192:], np.array([19, 20, 21, 28, 29, 30], np.float32))
    assert not np.array_equal(current[19:22], state[19:22])


def test_seed_formula_is_zero_based_with_task_offset():
    assert seed_for(1, 0) == 970100000
    assert seed_for(5, 49) == 970500049


def test_invalid_latent_is_rejected():
    with pytest.raises(ValueError):
        validate_latent(np.zeros((191,), dtype=np.float32), "current")
    with pytest.raises(ValueError):
        validate_latent(np.full((192,), np.nan, dtype=np.float32), "goal")


def test_camera_contract_is_frozen():
    assert CAMERA["render_name"] == "front_pixels"
    assert CAMERA["width"] == CAMERA["height"] == 224
    assert CAMERA["fov"] == 45.0
    assert CAMERA["aa_samples"] == 4


def test_pcd_preprocessor_receives_xyz_and_uint8_rgb_columns():
    merged = np.zeros((1200, 6), dtype=np.float32)
    merged[:, :3] = 0.125
    merged[:, 3:] = 128.0
    xyz, rgb = pcd_components(merged)
    assert xyz.shape == (1200, 3) and xyz.dtype == np.float32
    assert rgb.shape == (1200, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, np.full((1200, 3), 128, dtype=np.uint8))


def test_camera_sampling_matches_rgb_and_pcd_protocols():
    rgb = camera_for_group("G05")
    pcd = camera_for_group("G06")
    assert rgb["aa_samples"] == 4
    assert pcd["aa_samples"] == 4  # canonical RGB owner
    assert pcd["geometry_offsamples"] == 0  # PCD depth/segmentation geometry
