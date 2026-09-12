import torch

from models.action import ActionConfig, ActionEncoder, EnvSpec
from models.predictor import Predictor, PredictorConfig


def test_sequence_predictor_preserves_time_and_action_alignment():
    torch.manual_seed(0)
    model = Predictor(
        PredictorConfig(num_layers=1, num_heads=2, fusion="modulate"),
        D=16,
        envs=["synthetic"],
    ).eval()
    latent = torch.randn(1, 3, 8, 16)
    action = torch.randn(1, 3, 16)
    with torch.no_grad():
        out = model(latent, action, "synthetic")
        changed_action = action.clone()
        changed_action[:, 1] += 5.0
        out_changed = model(latent, changed_action, "synthetic")
        future_latent = latent.clone()
        future_latent[:, 2] += 7.0
        out_future_changed = model(future_latent, action, "synthetic")

    assert out.shape == latent.shape
    assert not torch.allclose(out[:, 1], out_changed[:, 1])
    assert torch.allclose(out[:, 0], out_changed[:, 0], atol=1e-6)
    assert torch.allclose(out[:, 1], out_future_changed[:, 1], atol=1e-6)


def test_sequence_predictor_uses_all_action_positions():
    torch.manual_seed(1)
    model = Predictor(
        PredictorConfig(num_layers=1, num_heads=2, fusion="add"),
        D=16,
        envs=["synthetic"],
    ).eval()
    latent = torch.randn(1, 3, 8, 16)
    action = torch.randn(1, 3, 16)
    with torch.no_grad():
        baseline = model(latent, action, "synthetic")
        changed = action.clone()
        changed[:, 0] += 5.0
        changed_out = model(latent, changed, "synthetic")

    assert not torch.allclose(baseline[:, 0], changed_out[:, 0])


def test_temporal_sequence_readout_is_causal_when_requested():
    from models.temporal import TemporalConfig, TemporalEncoder

    torch.manual_seed(2)
    temporal = TemporalEncoder(
        TemporalConfig(num_layers=1, num_heads=2, causal=False),
        K=8,
        T=3,
        D=16,
    ).eval()
    latent = torch.randn(1, 3, 8, 16)
    with torch.no_grad():
        out = temporal(latent, return_sequence=True, causal=True)
        changed = latent.clone()
        changed[:, 2] += 9.0
        changed_out = temporal(changed, return_sequence=True, causal=True)

    assert out.shape == latent.shape


def test_action_chunk_encoding_uses_ordered_all_actions():
    encoder = ActionEncoder(
        ActionConfig(envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])}), D=8, chunk_size=2
    ).eval()
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    swapped = actions.flip(1)
    with torch.no_grad():
        out = encoder.encode_chunks(actions, "synthetic")
        out_swapped = encoder.encode_chunks(swapped, "synthetic")
    assert out.shape == (1, 1, 8)
    assert not torch.allclose(out, out_swapped)


def test_action_mlp_keeps_visible_input_variation():
    torch.manual_seed(3)
    encoder = ActionEncoder(
        ActionConfig(envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])}), D=32
    ).eval()
    actions = torch.randn(128, 1, 2)
    with torch.no_grad():
        embedding = encoder(actions, "synthetic")
    assert embedding.std(dim=0).mean() > 0.02


def test_action_token_encoding_preserves_chunk_positions():
    torch.manual_seed(4)
    encoder = ActionEncoder(
        ActionConfig(envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])}),
        D=16,
        chunk_size=2,
        preserve_tokens=True,
    ).eval()
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.5]]])
    with torch.no_grad():
        tokens = encoder.encode_tokens(actions, "synthetic")
        swapped_then_realigned = encoder.encode_tokens(actions.view(1, 2, 2, 2).flip(2).reshape(1, 4, 2), "synthetic").flip(2)
    assert tokens.shape == (1, 2, 2, 16)
    assert not torch.allclose(tokens, swapped_then_realigned)


def test_cross_attention_predictor_uses_ordered_action_tokens():
    torch.manual_seed(5)
    encoder = ActionEncoder(
        ActionConfig(envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])}),
        D=16,
        chunk_size=2,
        preserve_tokens=True,
    ).eval()
    predictor = Predictor(
        PredictorConfig(num_layers=2, num_heads=2, fusion="cross_attention"),
        D=16,
        envs=["synthetic"],
    ).eval()
    latent = torch.randn(1, 2, 8, 16)
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.8, -0.2], [-0.3, 0.7]]])
    swapped = actions.view(1, 2, 2, 2).flip(2).reshape(1, 4, 2)
    with torch.no_grad():
        baseline = predictor(latent, encoder.encode_tokens(actions, "synthetic"), "synthetic")
        changed = predictor(latent, encoder.encode_tokens(swapped, "synthetic"), "synthetic")
    assert baseline.shape == latent.shape
    assert not torch.allclose(baseline, changed)


def test_contextual_action_tokens_remain_sequence_sensitive_as_a_set():
    torch.manual_seed(6)
    encoder = ActionEncoder(
        ActionConfig(
            envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])},
            token_encoder_layers=1,
            token_encoder_heads=2,
            token_mlp_ratio=2.0,
            token_position_encoding="sinusoidal",
            token_position_scale=1.0,
        ),
        D=16,
        chunk_size=4,
        preserve_tokens=True,
    ).eval()
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-0.5, 0.2], [0.3, -0.7]]])
    reversed_actions = actions.flip(1)

    with torch.no_grad():
        forward_tokens = encoder.encode_tokens(actions, "synthetic")
        reverse_tokens = encoder.encode_tokens(reversed_actions, "synthetic")

    assert forward_tokens.shape == reverse_tokens.shape == (1, 1, 4, 16)
    assert encoder.chunk_position.square().mean().sqrt() > 0.5
    assert (forward_tokens.mean(dim=2) - reverse_tokens.mean(dim=2)).square().mean().sqrt() > 1e-4


def test_contextual_action_tokens_make_cross_attention_sensitive_to_reversal():
    torch.manual_seed(7)
    encoder = ActionEncoder(
        ActionConfig(
            envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])},
            token_encoder_layers=1,
            token_encoder_heads=2,
            token_position_encoding="sinusoidal",
        ),
        D=16,
        chunk_size=4,
        preserve_tokens=True,
    ).eval()
    predictor = Predictor(
        PredictorConfig(num_layers=2, num_heads=2, fusion="cross_attention"),
        D=16,
        envs=["synthetic"],
    ).eval()
    latent = torch.randn(1, 1, 8, 16)
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-0.5, 0.2], [0.3, -0.7]]])

    with torch.no_grad():
        forward = predictor(latent, encoder.encode_tokens(actions, "synthetic"), "synthetic")
        reverse = predictor(latent, encoder.encode_tokens(actions.flip(1), "synthetic"), "synthetic")

    relative_change = (forward - reverse).square().mean().sqrt() / forward.square().mean().sqrt()
    assert relative_change > 1e-3


def test_default_action_token_encoder_keeps_legacy_state_dict_shape():
    encoder = ActionEncoder(
        ActionConfig(envs={"synthetic": EnvSpec(dim=2, mean=[0, 0], std=[1, 1])}),
        D=16,
        chunk_size=2,
        preserve_tokens=True,
    )

    keys = set(encoder.state_dict())
    assert "chunk_position" in keys
    assert not any(key.startswith("token_encoder.") for key in keys)


def test_cross_attention_predictor_uses_robot_token_in_every_layer():
    torch.manual_seed(8)
    predictor = Predictor(
        PredictorConfig(num_layers=2, num_heads=2, fusion="cross_attention"),
        D=16,
        envs=["synthetic"],
    ).eval()
    latent = torch.randn(1, 2, 8, 16)
    action_tokens = torch.randn(1, 2, 3, 16)
    robot_tokens = torch.randn(1, 2, 16)

    with torch.no_grad():
        baseline, features = predictor(
            latent,
            action_tokens,
            "synthetic",
            robot_tokens=robot_tokens,
            return_features=True,
        )
        changed_robot = robot_tokens.clone()
        changed_robot[:, 1, 0] += 5.0
        changed = predictor(
            latent,
            action_tokens,
            "synthetic",
            robot_tokens=changed_robot,
        )

    assert baseline.shape == features.shape == latent.shape
    assert torch.allclose(baseline[:, 0], changed[:, 0], atol=1e-6)
    assert not torch.allclose(baseline[:, 1], changed[:, 1])


def test_cross_attention_action_and_robot_tokens_receive_gradients():
    torch.manual_seed(9)
    predictor = Predictor(
        PredictorConfig(num_layers=2, num_heads=2, fusion="cross_attention"),
        D=16,
        envs=["synthetic"],
    )
    latent = torch.randn(1, 2, 8, 16)
    action_tokens = torch.randn(1, 2, 3, 16, requires_grad=True)
    robot_tokens = torch.randn(1, 2, 16, requires_grad=True)

    predictor(
        latent,
        action_tokens,
        "synthetic",
        robot_tokens=robot_tokens,
    ).square().mean().backward()

    assert action_tokens.grad is not None and action_tokens.grad.abs().sum() > 0
    assert robot_tokens.grad is not None and robot_tokens.grad.abs().sum() > 0
