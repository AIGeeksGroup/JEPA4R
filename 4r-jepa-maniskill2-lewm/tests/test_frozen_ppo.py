"""Stage-three integration checks using CPU SYNTHETIC compatibility fixtures.

No simulator, renderer, real expert dataset, or policy-quality assertion. The
source WM receives one optimizer update on explicitly synthetic consecutive data.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from maniskill2_learn.env import build_replay
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.networks.regression_heads.gaussian import GaussianHead
from maniskill2_learn.networks.utils import get_kwargs_from_shape, replace_placeholder_with_args
from maniskill2_learn.utils.torch import load_checkpoint, save_checkpoint

from ms2_lewm.frozen_ppo import FrozenWorldModelEncoder, export_encoder_bundle
from ms2_lewm.model import WorldModel
from ms2_lewm.rl_components import load_ppo_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppo_frozen_wm.py"
FIXTURE = Path(os.environ.get("MS2_LEWM_TEST_WM_FIXTURE", str(
    ROOT / "artifacts/tests" / f"frozen_ppo_stride1_source_{uuid4().hex}" / "weights_epoch_1.pt"
)))
NOTICE = "CPU SYNTHETIC compatibility fixture; not expert WM or policy-quality evidence"
pytestmark = pytest.mark.usefixtures("source_state")


def make_agent(checkpoint=FIXTURE, point_count=24):
    cfg = load_ppo_config(CONFIG)
    cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg.checkpoint = str(checkpoint)
    shape = {"xyz": (3, point_count, 3), "rgb": (3, point_count, 3), "state": 30}
    cfg.agent_cfg.env_params = dict(
        obs_shape=shape,
        action_shape=7,
        action_space=gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
        is_discrete=False,
    )
    cfg.agent_cfg.batch_size = 8
    cfg.agent_cfg.num_epoch = 1
    cfg.agent_cfg.critic_warmup_epoch = 0
    cfg = replace_placeholder_with_args(cfg, **get_kwargs_from_shape(shape, 7))
    return build_agent(cfg.agent_cfg).float().cpu()


def synthetic_samples(count=16, point_count=24):
    rng = np.random.default_rng(70619)
    assert count % 8 == 0
    histories = {"obs": [], "next_obs": []}
    for _ in range(count // 8):
        # Nine consecutive synthetic control observations form eight transitions.
        # Reset pads only from this episode, never from the preceding episode.
        xyz = rng.normal(size=(9, point_count, 3)).astype(np.float32)
        rgb = rng.uniform(size=(9, point_count, 3)).astype(np.float32)
        states = rng.normal(size=(9, 30)).astype(np.float32)
        for step in range(8):
            for key, index in (("obs", step), ("next_obs", step + 1)):
                indices = np.maximum(np.arange(index - 2, index + 1), 0)
                histories[key].append(dict(xyz=xyz[indices], rgb=rgb[indices], state=states[index]))
    observations = {key: {name: np.stack([obs[name] for obs in values]) for name in ("xyz", "rgb", "state")}
                    for key, values in histories.items()}
    done = np.zeros((count, 1), dtype=np.bool_)
    done[7::8] = True
    return dict(
        **observations,
        actions=rng.uniform(-0.5, 0.5, size=(count, 7)).astype(np.float32),
        rewards=np.linspace(-0.2, 0.7, count, dtype=np.float32)[:, None],
        dones=np.zeros((count, 1), dtype=np.bool_),
        episode_dones=done,
    )


def replay(samples):
    memory = build_replay(dict(type="ReplayMemory", capacity=len(samples["actions"])))
    memory.push_batch(deepcopy(samples))
    return memory


def tensor_observations(samples, count=4):
    return {key: torch.from_numpy(value[:count].copy()) for key, value in samples["obs"].items()}


def exact_tree(expected, actual):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            exact_tree(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            exact_tree(left, right)
    else:
        assert expected == actual


def bytes_snapshot(module):
    return {
        key: (str(value.dtype), tuple(value.shape), value.detach().cpu().contiguous().numpy().tobytes())
        for key, value in module.state_dict().items()
    }


def build_source_state():
    """Perform one actual WM optimizer update on consecutive SYNTHETIC inputs."""
    torch.manual_seed(7299)
    model = WorldModel(action_dim=7, history_stride=1).float().cpu().train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=5e-5, weight_decay=1e-3)
    before = {name: bytes_snapshot(module) for name, module in (
        ("frame", model.frame_encoder), ("temporal", model.temporal_encoder), ("predictor", model.predictor))}
    info = dict(pointcloud=torch.randn(4, 4, 24, 6), action=torch.randn(4, 4, 7))
    losses = model(info)
    assert bool(torch.isfinite(losses["loss"]))
    optimizer.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    for name, module in (("frame", model.frame_encoder), ("temporal", model.temporal_encoder),
                         ("predictor", model.predictor)):
        assert before[name] != bytes_snapshot(module), f"Synthetic WM {name} must actually be optimized."
    FIXTURE.parent.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), FIXTURE)
    config = dict(_target_="ms2_lewm.model.WorldModel", action_dim=7, history_stride=1,
                  rl_config_path="configs/ppo_pointnet_identity.py")
    (FIXTURE.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (FIXTURE.parent / "SYNTHETIC_FIXTURE.json").write_text(json.dumps(dict(
        notice=NOTICE, frameskip=1, history_size=3, optimizer_updates=1, production_recipe=False,
        genuine_expert_training=False, source=str(FIXTURE), loss=float(losses["loss"].detach()),
    ), indent=2) + "\n")
    return torch.load(FIXTURE, map_location="cpu", weights_only=False)


def test_encoder_exactly_loads_trained_wm_up_to_latent2(source_state):
    torch.manual_seed(7301)
    agent = make_agent()
    frozen = agent.actor.backbone.visual_nn
    assert type(frozen) is FrozenWorldModelEncoder
    assert frozen.output_dim == frozen.latent_dim == 512
    assert frozen.history_size == 3 and frozen.history_stride == 1
    assert frozen.pretraining_frameskip == 1
    assert frozen is agent.critic.values[0].backbone.visual_nn
    expected = {key.removeprefix("encoder."): value for key, value in source_state.items()
                if key.startswith("encoder.")}
    exact_tree(expected, frozen.encoder.state_dict())
    assert not any("predictor" in name or "loss_fn" in name for name, _ in agent.named_modules())
    model = WorldModel(action_dim=7, history_stride=1).cpu().eval()
    model.load_state_dict(source_state, strict=True)
    obs = tensor_observations(synthetic_samples())
    with torch.no_grad():
        expected_latent = model.encode(obs)["latent2"]
        actual_latent = frozen(obs)
    assert actual_latent.shape == (4, 512)
    torch.testing.assert_close(expected_latent, actual_latent, rtol=0, atol=0)
    assert not actual_latent.requires_grad
    assert not torch.cuda.is_initialized()


def test_source_checkpoint_fingerprint_is_enforced():
    checksum = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    frozen = FrozenWorldModelEncoder(checkpoint=str(FIXTURE), expected_sha256=checksum)
    assert frozen.output_dim == 512
    with pytest.raises((RuntimeError, ValueError), match="(?i)(sha|hash|fingerprint|mismatch)"):
        FrozenWorldModelEncoder(checkpoint=str(FIXTURE), expected_sha256="0" * 64)
    with pytest.raises((ValueError, FileNotFoundError)):
        FrozenWorldModelEncoder(checkpoint=None)


def test_old_frameskip5_checkpoint_cannot_be_silently_used_for_consecutive_ppo(tmp_path, source_state):
    # Generate the negative case in this test; published checkouts do not ship
    # old private/local training artifacts or depend on an earlier test run.
    legacy = tmp_path / "SYNTHETIC_legacy_stride5.pt"
    torch.save(source_state, legacy)
    config = json.loads((FIXTURE.parent / "config.json").read_text())
    config["history_stride"] = 5
    config["action_dim"] = 35
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="(?i)(stride|frameskip|consecutive|action)"):
        FrozenWorldModelEncoder(checkpoint=str(legacy))


def test_encoder_bundle_is_reusable_without_original_wm_file(tmp_path):
    bundle = export_encoder_bundle(FIXTURE, tmp_path / "frozen_encoder.pt")
    original = FrozenWorldModelEncoder(checkpoint=str(FIXTURE))
    source_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    # Only rename this test's newly generated file, restore it even on failure.
    hidden = FIXTURE.with_suffix(".temporarily_hidden")
    FIXTURE.rename(hidden)
    try:
        loaded = FrozenWorldModelEncoder(checkpoint=str(bundle))
        assert loaded.provenance["sha256"] == source_hash
        exact_tree(original.encoder.state_dict(), loaded.encoder.state_dict())
        obs = tensor_observations(synthetic_samples())
        torch.testing.assert_close(original(obs), loaded(obs), rtol=0, atol=0)
        assert not any("predictor" in name for name, _ in loaded.named_modules())
    finally:
        hidden.rename(FIXTURE)


def test_encoder_bundle_rejects_changed_config_or_implementation(tmp_path):
    bundle = export_encoder_bundle(FIXTURE, tmp_path / "frozen_encoder.pt")
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    snapshot = bundle.parent / payload["encoder_config"]["rl_config_path"]
    assert snapshot.parent == tmp_path
    original_config = snapshot.read_bytes()
    snapshot.write_bytes(original_config + b"\n# Deliberate test-only change to saved model configuration.\n")
    try:
        with pytest.raises(ValueError, match="configuration SHA256 mismatch"):
            FrozenWorldModelEncoder(checkpoint=str(bundle))
    finally:
        snapshot.write_bytes(original_config)
    assert payload["encoder_definition_sha256"]
    definition = next(iter(payload["encoder_definition_sha256"]))
    payload["encoder_definition_sha256"][definition] = "0" * 64
    changed_manifest = tmp_path / "deliberately_wrong_definition_digest.pt"
    torch.save(payload, changed_manifest)
    with pytest.raises(ValueError, match="implementation SHA256 mismatch"):
        FrozenWorldModelEncoder(checkpoint=str(changed_manifest))


def test_encoder_bundle_moves_between_checkout_and_checkpoint_directories(tmp_path, monkeypatch):
    import ms2_lewm.frozen_ppo as frozen_module

    bundle = export_encoder_bundle(FIXTURE, tmp_path / "original_run/frozen_encoder.pt")
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    assert payload["format"] == "ms2_lewm_frozen_encoder_v3"
    assert not Path(payload["encoder_config"]["rl_config_path"]).is_absolute()
    relocated_root = tmp_path / "new_checkout"
    for relative in payload["encoder_definition_sha256"]:
        assert not Path(relative).is_absolute()
        destination = relocated_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    relocated_run = tmp_path / "new_run"
    shutil.move(str(bundle.parent), relocated_run)
    monkeypatch.setattr(frozen_module, "ROOT", relocated_root)
    monkeypatch.chdir(tmp_path)
    loaded = FrozenWorldModelEncoder(checkpoint=str(relocated_run / bundle.name))
    expected = {key.removeprefix("encoder."): value for key, value in
                torch.load(FIXTURE, map_location="cpu", weights_only=False).items()
                if key.startswith("encoder.")}
    exact_tree(expected, loaded.encoder.state_dict())


def test_project_relative_encoder_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_ppo_config("configs/ppo_pointnet_identity.py")
    assert config.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg.type == "PointNetLastIdentity"
    frozen = FrozenWorldModelEncoder(checkpoint=str(FIXTURE))
    assert frozen.history_stride == 1 and frozen.output_dim == 512


def test_only_latent2_feeds_original_actor_critic_and_gaussian():
    agent = make_agent()
    baseline = load_ppo_config()
    cfg = load_ppo_config(CONFIG)
    expected_agent = deepcopy(baseline.agent_cfg)
    actual_agent = deepcopy(cfg.agent_cfg)
    for branch in ("actor_cfg", "critic_cfg"):
        expected_agent[branch].pop("nn_cfg")
        actual_agent[branch].pop("nn_cfg")
    assert expected_agent == actual_agent, "Stage 3 must retain PPO losses, optimizers and algorithm settings."
    for key in ("train_cfg", "replay_cfg", "rollout_cfg", "eval_cfg"):
        assert cfg[key] == baseline[key]
    assert cfg.agent_cfg.actor_cfg.head_cfg == baseline.agent_cfg.actor_cfg.head_cfg
    assert type(agent.actor.head) is GaussianHead
    assert agent.actor.head.log_std.shape == (1, 7)
    for head, output_dim in ((agent.actor.backbone.final_mlp, 7),
                             (agent.critic.values[0].backbone.final_mlp, 1)):
        modules = list(head.mlp.children())
        assert [type(part) for part in modules] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
        assert [(part.in_features, part.out_features) for part in modules if isinstance(part, nn.Linear)] == [
            (512, 256), (256, 256), (256, output_dim)]
    obs = tensor_observations(synthetic_samples())
    agent.eval()
    with torch.no_grad():
        original = {method: getattr(agent, method)(obs, mode="eval") for method in ("actor", "critic")}
        obs["state"].fill_(1e6)
        for method in ("actor", "critic"):
            torch.testing.assert_close(original[method], getattr(agent, method)(obs, mode="eval"), rtol=0, atol=0)


def test_freeze_survives_train_mode_and_shuffled_minibatches():
    agent = make_agent()
    frozen = agent.actor.backbone.visual_nn
    agent.train()
    assert agent.actor.backbone.final_mlp.training
    assert agent.critic.values[0].backbone.final_mlp.training
    assert all(not module.training for module in frozen.modules())
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    encoder_ids = {id(parameter) for parameter in frozen.parameters()}
    for optimizer in (agent.actor_optim, agent.critic_optim):
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert not encoder_ids & optimizer_ids
    obs = tensor_observations(synthetic_samples())
    before = bytes_snapshot(frozen)
    first = frozen(obs)
    second = frozen(obs)
    assert not first.requires_grad
    assert torch.equal(first, second), "Frozen temporal dropout must remain disabled during PPO updates."
    order = torch.tensor([2, 0, 3, 1])
    shuffled = frozen({key: value[order] for key, value in obs.items()})
    torch.testing.assert_close(first[order], shuffled, rtol=1e-6, atol=1e-6)
    assert before == bytes_snapshot(frozen), "The encoder must not store/update implicit episode history."


def test_actual_upstream_ppo_updates_only_heads_and_exactly_restores(tmp_path):
    torch.manual_seed(7302)
    np.random.seed(7302)
    agent = make_agent()
    agent.train()
    frozen = agent.actor.backbone.visual_nn
    frozen_before = bytes_snapshot(frozen)
    actor_before = bytes_snapshot(agent.actor.backbone.final_mlp)
    critic_before = bytes_snapshot(agent.critic.values[0].backbone.final_mlp)
    log_std_before = agent.actor.head.log_std.detach().clone()
    samples = synthetic_samples()
    metrics = agent.update_parameters(replay(samples), updates=0)
    assert frozen_before == bytes_snapshot(frozen), "Frozen parameters AND buffers must stay byte-identical."
    assert actor_before != bytes_snapshot(agent.actor.backbone.final_mlp)
    assert critic_before != bytes_snapshot(agent.critic.values[0].backbone.final_mlp)
    assert not torch.equal(log_std_before, agent.actor.head.log_std.detach())
    assert all(parameter.grad is None for parameter in frozen.parameters())
    gradients = [parameter.grad for parameter in agent.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.parameters())
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert float(metrics["grad/visual_grad"]) == 0
    assert float(metrics["grad/actor_mlp_grad"]) > 0
    assert float(metrics["grad/critic_mlp_grad"]) > 0

    checkpoint = tmp_path / "SYNTHETIC_frozen_ppo_compatibility.ckpt"
    provenance = dict(validation_kind=NOTICE, wm_checkpoint=str(FIXTURE),
                      wm_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
    save_checkpoint(agent, str(checkpoint),
                    optimizer=dict(actor=agent.actor_optim, critic=agent.critic_optim), meta=provenance)
    restored = make_agent()
    stored = load_checkpoint(restored, str(checkpoint), map_location="cpu", strict=True)
    assert stored["meta"] == provenance
    restored.actor_optim.load_state_dict(stored["optimizer"]["actor"])
    restored.critic_optim.load_state_dict(stored["optimizer"]["critic"])
    exact_tree(agent.state_dict(), restored.state_dict())
    exact_tree(agent.actor_optim.state_dict(), restored.actor_optim.state_dict())
    exact_tree(agent.critic_optim.state_dict(), restored.critic_optim.state_dict())
    obs = tensor_observations(samples)
    agent.eval()
    restored.eval()
    with torch.no_grad():
        for method in ("actor", "critic"):
            torch.testing.assert_close(getattr(agent, method)(obs, mode="eval"),
                                       getattr(restored, method)(obs, mode="eval"), rtol=0, atol=0)
    # Same subsequent minibatches/RNG must reproduce the optimizer continuation.
    agent.train()
    restored.train()
    for current in (agent, restored):
        torch.manual_seed(7303)
        np.random.seed(7303)
        current.update_parameters(replay(samples), updates=1)
    exact_tree(agent.state_dict(), restored.state_dict())
    exact_tree(agent.actor_optim.state_dict(), restored.actor_optim.state_dict())
    exact_tree(agent.critic_optim.state_dict(), restored.critic_optim.state_dict())
    assert frozen_before == bytes_snapshot(frozen)
    assert not torch.cuda.is_initialized()
