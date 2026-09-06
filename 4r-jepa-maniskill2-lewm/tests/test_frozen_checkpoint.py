"""Check actual CLI-style checkpoint calls and stage-three preparation on CPU."""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import maniskill2_learn.utils.torch as upstream_torch
from maniskill2_learn.utils.meta import Config
from ms2_lewm.rl_components import load_ppo_config

from test_frozen_ppo import make_agent, synthetic_samples, replay, exact_tree, FIXTURE


ROOT = Path(__file__).resolve().parents[1]
SOURCE = FIXTURE
pytestmark = pytest.mark.usefixtures("source_state")


def updated_agent():
    assert SOURCE.is_file(), "The session's newly generated SYNTHETIC WM fixture is required."
    torch.manual_seed(8901)
    np.random.seed(8901)
    agent = make_agent(checkpoint=SOURCE)
    agent.train()
    agent.update_parameters(replay(synthetic_samples()), updates=0)
    return agent


def assert_restored(agent, restored):
    exact_tree(agent.state_dict(), restored.state_dict())
    assert agent.actor_optim.state and agent.critic_optim.state
    exact_tree(agent.actor_optim.state_dict(), restored.actor_optim.state_dict())
    exact_tree(agent.critic_optim.state_dict(), restored.critic_optim.state_dict())
    assert all(not parameter.requires_grad for parameter in restored.actor.backbone.visual_nn.parameters())


def load_launcher():
    spec = importlib.util.spec_from_file_location("_frozen_ppo_launcher_test", ROOT / "scripts/run_frozen_ppo.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_unmodified_upstream_default_checkpoint_calls_restore_embedded_optimizers(tmp_path):
    # The official train_rl uses this two-argument save call. Upstream includes
    # Optimizer attributes in its custom state_dict, even without optimizer=.
    agent = updated_agent()
    checkpoint = tmp_path / "upstream_default_call.ckpt"
    upstream_torch.save_checkpoint(agent, str(checkpoint))
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "actor_optim" in stored["state_dict"] and "critic_optim" in stored["state_dict"]
    restored = make_agent(checkpoint=SOURCE)
    upstream_torch.load_checkpoint(restored, str(checkpoint), map_location="cpu", strict=True)
    assert_restored(agent, restored)
    for current in (agent, restored):
        torch.manual_seed(8902)
        np.random.seed(8902)
        current.update_parameters(replay(synthetic_samples()), updates=1)
    assert_restored(agent, restored)
    assert not torch.cuda.is_initialized()


def test_prepared_launcher_keeps_production_ppo_recipe_and_consecutive_history(tmp_path):
    launcher = load_launcher()
    # All prepared artifacts remain within this task; preparation creates no env.
    work_dir = ROOT / "artifacts/tests" / ("frozen_launcher_" + tmp_path.name)
    args = SimpleNamespace(wm_checkpoint=str(SOURCE), work_dir=str(work_dir), prepare_only=True,
                           smoke=False, evaluation=False, resume_from=None, num_envs=1,
                           device="cpu", seed=8903)
    cfg, provenance, command = launcher.prepare(args)
    expected = load_ppo_config(ROOT / "configs/ppo_frozen_wm.py")
    actual_agent = deepcopy(cfg.agent_cfg)
    expected_agent = deepcopy(expected.agent_cfg)
    for current in (actual_agent, expected_agent):
        current.actor_cfg.nn_cfg.pop("visual_nn_cfg")
    assert actual_agent == expected_agent
    for key in ("train_cfg", "replay_cfg"):
        assert cfg[key] == expected[key]
    assert cfg.env_cfg.extra_wrappers == [dict(type="PointCloudHistory", history_size=3, history_stride=1)]
    assert cfg.rollout_cfg.num_procs == 1
    assert provenance["history_stride"] == 1 and provenance["history_size"] == 3
    assert provenance["action_dim"] == 7 and provenance["representation"] == "latent2"
    assert provenance["test_source"] is True
    assert "--num-gpus" in command and command[command.index("--num-gpus") + 1] == "0"
    saved = Config.fromfile(provenance["config"])
    assert saved.agent_cfg == cfg.agent_cfg
    assert not work_dir.exists(), "Preparing must not execute a training work directory."
    assert not torch.cuda.is_initialized()


def test_resume_preparation_rejects_different_worldmodel_source(tmp_path):
    agent = make_agent(checkpoint=SOURCE)
    checkpoint = tmp_path / "ppo_from_known_wm.ckpt"
    upstream_torch.save_checkpoint(agent, str(checkpoint))
    launcher = load_launcher()
    args = SimpleNamespace(wm_checkpoint=str(SOURCE),
                           work_dir=str(ROOT / "artifacts/tests" / ("frozen_resume_" + tmp_path.name)),
                           prepare_only=True, smoke=False, evaluation=False,
                           resume_from=str(checkpoint), num_envs=1, device="cpu", seed=8904)
    _, provenance, command = launcher.prepare(args)
    assert "--resume-from" in command
    assert command[command.index("--resume-from") + 1] == str(checkpoint)
    assert provenance["source"]["checkpoint"] == str(SOURCE)

    # Create a new, deliberately different fixture; original export stays intact.
    replacement = tmp_path / "different_SYNTHETIC_worldmodel" / "weights_epoch_2.pt"
    replacement.parent.mkdir()
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    parameter_name = next(key for key, value in state.items()
                          if key.startswith("encoder.") and torch.is_floating_point(value))
    state[parameter_name] = state[parameter_name] + 0.001
    torch.save(state, replacement)
    shutil.copyfile(SOURCE.parent / "config.json", replacement.parent / "config.json")
    args.wm_checkpoint = str(replacement)
    with pytest.raises(ValueError, match="(?i)(source|world.?model|sha|fingerprint|identity)"):
        launcher.prepare(args)
    assert not torch.cuda.is_initialized()
