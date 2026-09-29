import pytest

from lewm_pcd.scripts.data.launch_adroit_rgb_bc import (
    validate_encoder_output,
    validate_resolved_config,
)


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
        "bc_kwargs": {
            "hidden_sizes": "(256, 256, 256)",
            "nonlinearity": "relu",
            "loss_type": "MSE",
            "batch_size": 256,
            "lr": 0.001,
            "dropout": 0,
        },
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
        "bc_kwargs": {
            "hidden_sizes": "(256, 256, 256)",
            "nonlinearity": "relu",
            "loss_type": "MSE",
            "batch_size": 256,
            "lr": 0.001,
            "dropout": 0,
        },
    }
    with pytest.raises(ValueError, match="num_demos"):
        validate_resolved_config(resolved, 100, 100, 5, 25)


def test_encoder_output_contract_rejects_wrong_dimension_or_nonfinite_values():
    assert validate_encoder_output((1, 3, 224, 224), (1, 768), "torch.float32", True, True, 768) is None
    with pytest.raises(ValueError, match="output shape"):
        validate_encoder_output((1, 3, 224, 224), (1, 1024), "torch.float32", True, True, 768)
    with pytest.raises(ValueError, match="finite"):
        validate_encoder_output((1, 3, 224, 224), (1, 768), "torch.float32", True, False, 768)
