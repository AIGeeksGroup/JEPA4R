import pytest

from lewm_pcd.scripts.data.launch_adroit_rgb_bc import validate_resolved_config


def test_rgb_launcher_accepts_official_adroit_contract():
    resolved = {
        "env": "pen-v0",
        "camera": "vil_camera",
        "pixel_based": True,
        "num_demos": 100,
        "epochs": 100,
        "eval_frequency": 5,
        "eval_num_traj": 25,
        "env_kwargs": {"add_proprio": True, "proprio_key": "proprio"},
    }
    assert validate_resolved_config(resolved, 100, 100, 5, 25) is None


def test_rgb_launcher_rejects_non_official_demo_count():
    resolved = {
        "env": "pen-v0",
        "camera": "vil_camera",
        "pixel_based": True,
        "num_demos": 25,
        "epochs": 100,
        "eval_frequency": 5,
        "eval_num_traj": 25,
        "env_kwargs": {"add_proprio": True, "proprio_key": "proprio"},
    }
    with pytest.raises(ValueError, match="num_demos"):
        validate_resolved_config(resolved, 100, 100, 5, 25)
