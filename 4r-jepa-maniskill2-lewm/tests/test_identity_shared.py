"""Check the shared PPO/WM activation boundary without creating an environment."""

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.networks.backbones.pointnet import PointNet
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)

from ms2_lewm.pointnet_identity import PointNetLastIdentity
from ms2_lewm.rl_components import (
    OfficialPointNetFrameEncoder,
    load_ppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG = ROOT / "vendor/ManiSkill2-Learn/configs/mfrl/ppo/maniskill2_pn.py"


def build_cpu_ppo(point_count=24):
    """Instantiate the production agent from spaces, without a simulator."""
    cfg = load_ppo_config()
    obs_shape = {"xyz": (point_count, 3), "rgb": (point_count, 3), "state": 30}
    cfg.agent_cfg.env_params = dict(
        obs_shape=obs_shape,
        action_shape=7,
        action_space=gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
        is_discrete=False,
    )
    resolved = replace_placeholder_with_args(
        cfg, **get_kwargs_from_shape(obs_shape, 7)
    )
    return build_agent(resolved.agent_cfg).float().cpu(), obs_shape


def test_only_the_last_pointnet_activation_changes_and_weights_remain_compatible():
    official_cfg = load_ppo_config(OFFICIAL_CONFIG)
    shared_cfg = load_ppo_config()
    expected = deepcopy(official_cfg.agent_cfg)
    expected.actor_cfg.nn_cfg.visual_nn_cfg.type = "PointNetLastIdentity"
    assert shared_cfg.agent_cfg == expected
    for key in ("train_cfg", "env_cfg", "replay_cfg", "rollout_cfg", "eval_cfg"):
        assert shared_cfg[key] == official_cfg[key]

    original = OfficialPointNetFrameEncoder(config_path=OFFICIAL_CONFIG).pointnet
    modified = OfficialPointNetFrameEncoder().pointnet
    assert type(original) is PointNet
    assert type(modified) is PointNetLastIdentity
    assert PointNetLastIdentity.forward is PointNet.forward
    original_modules = dict(original.named_modules())
    modified_modules = dict(modified.named_modules())
    assert original_modules.keys() == modified_modules.keys()
    changed = [
        name for name in original_modules
        if name and type(original_modules[name]) is not type(modified_modules[name])
    ]
    assert changed == ["conv.mlp.layer2.act"]
    assert isinstance(modified.conv.mlp.layer0.act, nn.ReLU)
    assert isinstance(modified.conv.mlp.layer1.act, nn.ReLU)
    assert isinstance(modified.conv.mlp.layer2.act, nn.Identity)
    assert original.state_dict().keys() == modified.state_dict().keys()
    assert {
        name: tuple(parameter.shape) for name, parameter in original.named_parameters()
    } == {
        name: tuple(parameter.shape) for name, parameter in modified.named_parameters()
    }
    modified.load_state_dict(original.state_dict(), strict=True)
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, modified.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("bias", [-2.0, 2.0])
def test_identity_emits_unbounded_signed_features_after_official_max_pooling(bias):
    encoder = OfficialPointNetFrameEncoder().eval()
    # Exercise the actual official max-pooling path with known final LN output.
    # A retained ReLU would fail the negative case; tanh would fail both cases.
    last_norm = encoder.pointnet.conv.mlp.layer2._modules["norm"]
    with torch.no_grad():
        last_norm.weight.zero_()
        last_norm.bias.fill_(bias)
        latent = encoder(torch.randn(2, 3, 24, 6))
    torch.testing.assert_close(latent, torch.full_like(latent, bias), rtol=0, atol=0)
    assert bool((latent.abs() > 1).all())


def test_ppo_and_worldmodel_use_the_same_encoder_configuration_and_features():
    torch.manual_seed(93)
    agent, obs_shape = build_cpu_ppo()
    agent.eval()
    visual = agent.actor.backbone.visual_nn
    critic_visual = agent.critic.values[0].backbone.visual_nn
    assert visual is critic_visual
    assert type(visual) is PointNetLastIdentity
    frame = OfficialPointNetFrameEncoder(obs_shape=obs_shape, action_dim=7).eval()
    frame.pointnet.load_state_dict(visual.state_dict(), strict=True)
    obs = {"xyz": torch.randn(2, 24, 3), "rgb": torch.rand(2, 24, 3)}
    with torch.no_grad():
        torch.testing.assert_close(visual(obs), frame(obs), rtol=0, atol=0)
    assert frame.source_visual_nn_cfg == load_ppo_config().agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg
    for head in (agent.actor.backbone.final_mlp, agent.critic.values[0].backbone.final_mlp):
        assert [type(module) for module in head.mlp.children()] == [
            nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear
        ]
