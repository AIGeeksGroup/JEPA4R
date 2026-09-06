"""Check actual upstream reuse and the semantically important adapter behavior."""

from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from maniskill2_learn.networks import build_model
from maniskill2_learn.networks.backbones.mlp import LinearMLP
from maniskill2_learn.networks.backbones.pointnet import PointNet
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)

from ms2_lewm.rl_components import (
    DEFAULT_RL_CONFIG,
    OfficialCriticPredictor,
    OfficialPointNetFrameEncoder,
    load_ppo_config,
)
from ms2_lewm.pointnet_identity import PointNetLastIdentity


def test_frame_encoder_uses_shared_identity_class_and_resolved_ppo_configuration():
    obs_shape = {"xyz": (24, 3), "rgb": (24, 3), "state": 25}
    cfg = load_ppo_config()
    raw_visual_cfg = deepcopy(cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg)
    expected_cfg = replace_placeholder_with_args(
        deepcopy(raw_visual_cfg), **get_kwargs_from_shape(obs_shape, 7)
    )
    encoder = OfficialPointNetFrameEncoder(obs_shape, action_dim=7)

    assert type(encoder.pointnet) is PointNetLastIdentity
    assert isinstance(encoder.pointnet, PointNet)
    assert PointNetLastIdentity.forward is PointNet.forward
    assert Path(inspect.getfile(PointNet)).resolve() == (
        Path(__file__).resolve().parents[1]
        / "vendor/ManiSkill2-Learn/maniskill2_learn/networks/backbones/pointnet.py"
    ).resolve()
    assert DEFAULT_RL_CONFIG.name == "ppo_pointnet_identity.py"
    assert encoder.source_visual_nn_cfg == raw_visual_cfg
    assert encoder.visual_nn_cfg == expected_cfg
    assert encoder.visual_nn_cfg.feat_dim == 6
    assert encoder.output_dim == encoder.latent_dim == 512
    assert encoder.visual_nn_cfg.mlp_spec == [64, 128, 512]
    assert encoder.visual_nn_cfg.feature_transform == []
    # Reading/building never mutates the pristine configuration.
    assert load_ppo_config().agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg == raw_visual_cfg


def test_time_batching_matches_independent_frames_and_dict_input():
    torch.manual_seed(31)
    encoder = OfficialPointNetFrameEncoder().eval()
    xyz = torch.randn(2, 3, 24, 3)
    rgb = torch.rand(2, 3, 24, 3)
    batched = encoder({"xyz": xyz, "rgb": rgb})
    sequential = torch.stack(
        [encoder({"xyz": xyz[:, t], "rgb": rgb[:, t]}) for t in range(3)],
        dim=1,
    )
    concatenated = encoder(torch.cat((xyz, rgb), dim=-1))
    assert batched.shape == (2, 3, 512)
    torch.testing.assert_close(batched, sequential)
    torch.testing.assert_close(batched, concatenated)


def test_every_frame_including_target_backpropagates_to_shared_encoder():
    torch.manual_seed(32)
    encoder = OfficialPointNetFrameEncoder()
    latent1 = encoder(torch.randn(2, 4, 24, 6))
    parameters = tuple(encoder.pointnet.parameters())
    for frame_index in range(4):
        gradients = torch.autograd.grad(
            latent1[:, frame_index].square().mean(),
            parameters,
            retain_graph=frame_index < 3,
        )
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


@pytest.mark.parametrize("action_dim", [7, 35])
def test_predictor_preserves_official_critic_hidden_architecture(action_dim):
    cfg = load_ppo_config()
    source_cfg = deepcopy(cfg.agent_cfg.critic_cfg.nn_cfg.mlp_cfg)
    predictor = OfficialCriticPredictor(action_dim=action_dim)
    expected_cfg = deepcopy(source_cfg)
    expected_cfg.mlp_spec[0] = 512 + action_dim
    expected_cfg.mlp_spec[-1] = 512

    assert type(predictor.head) is LinearMLP
    assert predictor.source_mlp_cfg == source_cfg
    assert predictor.mlp_cfg == expected_cfg
    assert predictor.mlp_cfg.mlp_spec == [512 + action_dim, 256, 256, 512]
    modules = tuple(predictor.head.mlp.children())
    assert [type(module) for module in modules] == [
        nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear
    ]
    assert predictor.head.last_linear.out_features == 512
    assert predictor(torch.randn(2, 512), torch.randn(2, action_dim)).shape == (2, 512)
    assert load_ppo_config().agent_cfg.critic_cfg.nn_cfg.mlp_cfg == source_cfg


def test_predictor_retains_upstream_output_initialization_exactly():
    torch.manual_seed(33)
    predictor = OfficialCriticPredictor(action_dim=7)
    unscaled_cfg = deepcopy(predictor.mlp_cfg)
    unscaled_cfg.zero_init_output = False
    torch.manual_seed(33)
    unscaled = build_model(unscaled_cfg)
    torch.testing.assert_close(
        predictor.head.last_linear.weight, 0.01 * unscaled.last_linear.weight,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        predictor.head.last_linear.bias,
        torch.zeros_like(predictor.head.last_linear.bias),
        rtol=0, atol=0,
    )
    for name in ("linear0", "linear1"):
        torch.testing.assert_close(
            getattr(predictor.head.mlp, name).weight,
            getattr(unscaled.mlp, name).weight,
            rtol=0, atol=0,
        )


def test_mismatched_point_channels_and_action_dimensions_fail_clearly():
    encoder = OfficialPointNetFrameEncoder()
    with pytest.raises(ValueError, match="expects 6 channels"):
        encoder(torch.randn(2, 3, 24, 3))
    with pytest.raises(ValueError, match="incompatible shape"):
        encoder({"xyz": torch.randn(2, 3, 24, 3), "rgb": torch.randn(2, 4, 24, 3)})
    predictor = OfficialCriticPredictor(action_dim=7)
    with pytest.raises(ValueError, match="action dimension 7"):
        predictor(torch.randn(2, 512), torch.randn(2, 6))
