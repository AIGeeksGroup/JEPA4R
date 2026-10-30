import numpy as np
import pytest

from visual_sr import (
    CAMERA,
    goal_xyz,
    make_policy_inputs,
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
