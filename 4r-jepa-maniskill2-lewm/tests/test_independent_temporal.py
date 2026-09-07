"""CPU checks for our independent temporal encoder, without LeWM imports."""
import inspect
import json

import pytest
import torch
import yaml
from torch import nn

from ms2_lewm.temporal import (
    DEFAULT_TEMPORAL_CONFIG,
    TemporalEncoder,
    resolve_temporal_config,
)


def test_defaults_come_from_our_own_yaml_and_stay_plain_values():
    with DEFAULT_TEMPORAL_CONFIG.open() as source:
        defaults = yaml.safe_load(source)
    config = resolve_temporal_config()
    assert config == defaults
    assert json.loads(json.dumps(config)) == config
    assert config == {
        "architecture": "temporal_encoder_v1",
        "history_size": 3,
        "num_layers": 2,
        "num_heads": 8,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
    }


def test_relative_config_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_temporal_config("configs/temporal_encoder.yaml") == resolve_temporal_config()


def test_custom_config_is_resolved_without_mutating_caller(tmp_path):
    supplied = {"history_size": 4, "num_layers": 1}
    model = TemporalEncoder(32, config=supplied)
    assert supplied == {"history_size": 4, "num_layers": 1}
    assert model.num_frames == 4
    assert len(model.layers) == 1
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text("history_size: 4\nnum_layers: 1\n")
    assert resolve_temporal_config(custom_path) == model.config


def test_full_dimensional_attention_and_independently_initialized_layers():
    torch.manual_seed(513)
    model = TemporalEncoder(512)
    assert model.latent_dim == 512
    assert model.K == 1
    assert model.pos_embedding.shape == (1, 3, 512)
    assert model.norm.normalized_shape == (512,)
    assert len(model.layers) == 2
    for layer in model.layers:
        assert type(layer) is nn.TransformerEncoderLayer
        assert layer.self_attn.embed_dim == 512
        assert layer.self_attn.num_heads == 8
        assert layer.linear1.in_features == layer.linear2.out_features == 512
        assert layer.linear1.out_features == layer.linear2.in_features == 2048
        assert layer.norm_first
        assert layer.dropout.p == 0
    assert all(192 not in param.shape for param in model.parameters())
    assert not torch.equal(
        model.layers[0].self_attn.in_proj_weight,
        model.layers[1].self_attn.in_proj_weight,
    )
    assert all(module.__class__.__module__.startswith(("torch.", "ms2_lewm.temporal"))
               for module in model.modules())
    assert "action" not in inspect.signature(model.forward).parameters


def test_all_parameters_and_every_context_frame_receive_gradients():
    torch.manual_seed(514)
    model = TemporalEncoder(512)
    latent1 = torch.randn(2, 3, 512, requires_grad=True)
    latent2 = model(latent1)
    assert latent2.shape == (2, 512)
    target = torch.randn_like(latent2)
    (latent2 - target).square().mean().backward()
    assert all(latent1.grad[:, t].abs().sum() > 0 for t in range(3))
    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_causal_attention_preserves_prefix_when_future_changes():
    torch.manual_seed(515)
    model = TemporalEncoder(32).eval()
    latent1 = torch.randn(2, 3, 32)
    changed_future = latent1.clone()
    changed_future[:, -1] = torch.randn(2, 32) * 30
    outputs = []
    handle = model.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: outputs.append(output.detach().clone())
    )
    with torch.no_grad():
        original = model(latent1)
        changed = model(changed_future)
        prefix = model(latent1[:, :2])
    handle.remove()
    torch.testing.assert_close(outputs[0][:, :2], outputs[1][:, :2], rtol=0, atol=0)
    torch.testing.assert_close(prefix, model.norm(outputs[0][:, 1]))
    assert not torch.allclose(original, changed)


def test_k1_view_and_short_history():
    model = TemporalEncoder(32).eval()
    latent1 = torch.randn(2, 3, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(latent1), model(latent1.unsqueeze(2)))
        for history in (1, 2, 3):
            assert model(latent1[:, :history]).shape == (2, 32)


@pytest.mark.parametrize("bad_k", [0, 2, True])
def test_invalid_k_is_rejected(bad_k):
    with pytest.raises(ValueError, match="K=1"):
        TemporalEncoder(32, K=bad_k)


@pytest.mark.parametrize("shape,match", [
    ((2, 0, 32), "context length"),
    ((2, 4, 32), "context length"),
    ((2, 3, 16), "Expected"),
    ((2, 32), "Expected"),
    ((2, 3, 2, 32), "K=1"),
])
def test_invalid_input_shape_is_rejected(shape, match):
    with pytest.raises(ValueError, match=match):
        TemporalEncoder(32)(torch.randn(shape))


@pytest.mark.parametrize("config,match", [
    ({"architecture": "legacy"}, "architecture"),
    ({"history_size": 0}, "history_size"),
    ({"history_size": True}, "history_size"),
    ({"num_layers": 0}, "num_layers"),
    ({"num_heads": 0}, "num_heads"),
    ({"mlp_ratio": 0}, "mlp_ratio"),
    ({"dropout": 1}, "dropout"),
    ({"dropout": float("nan")}, "dropout"),
    ({"activation": "relu"}, "GELU"),
    ({"norm_first": "true"}, "boolean"),
    ({"hidden_dim": 192}, "Unknown"),
])
def test_invalid_config_is_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        resolve_temporal_config(config)


def test_width_must_allow_whole_attention_heads():
    with pytest.raises(ValueError, match="divisible"):
        TemporalEncoder(31)
