#!/usr/bin/env python
"""Bounded CPU PPO compatibility check; never constructs a simulator/renderer.

Choose --synthetic for a self-contained checkout test, or --fixture PATH to
provide a saved point-cloud/action NPZ. Robot states, rewards and episode flags
are always synthetic; an external fixture is not verified as real data. This exercises the
unmodified upstream PPO update; it is not fresh on-policy training, a complete
real-transition replay, or evidence of policy quality/convergence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[variable] = "1"
if hasattr(os, "nice") and os.nice(0) < 19:
    os.nice(19 - os.nice(0))

import gymnasium as gym
import numpy as np
import torch

from maniskill2_learn.env import build_replay
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.networks.utils import get_kwargs_from_shape, replace_placeholder_with_args
from maniskill2_learn.utils.torch import load_checkpoint, save_checkpoint

from ms2_lewm.pointnet_identity import PointNetLastIdentity
from ms2_lewm.rl_components import DEFAULT_RL_CONFIG, load_ppo_config


ROOT = Path(__file__).resolve().parents[1]


def make_agent(point_count):
    cfg = load_ppo_config()
    obs_shape = {"xyz": (point_count, 3), "rgb": (point_count, 3), "state": 30}
    cfg.agent_cfg.env_params = dict(
        obs_shape=obs_shape, action_shape=7,
        action_space=gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
        is_discrete=False,
    )
    # Bound this check independently of the production training configuration.
    cfg.agent_cfg.batch_size = 16
    cfg.agent_cfg.num_epoch = 1
    cfg.agent_cfg.critic_warmup_epoch = 0
    cfg = replace_placeholder_with_args(cfg, **get_kwargs_from_shape(obs_shape, 7))
    return build_agent(cfg.agent_cfg).float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path, help="Explicit NPZ with xyz/rgb/next_xyz/next_rgb/actions; never bundled.")
    source.add_argument("--synthetic", action="store_true", help="Generate clearly labeled random CPU compatibility inputs.")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260906)
    np.random.seed(20260906)
    if args.synthetic:
        rng = np.random.default_rng(20260906)
        data = {name: rng.normal(size=(32, 64, 3)).astype(np.float32)
                for name in ("xyz", "next_xyz")}
        data.update({name: rng.uniform(size=(32, 64, 3)).astype(np.float32)
                     for name in ("rgb", "next_rgb")})
        data["actions"] = rng.uniform(-0.5, 0.5, size=(32, 7)).astype(np.float32)
    else:
        args.fixture = args.fixture.expanduser().resolve()
        with np.load(args.fixture, allow_pickle=False) as stored:
            data = {name: stored[name].copy() for name in stored.files}
    data_notice = ("all point clouds/actions/states/rewards/flags are SYNTHETIC" if args.synthetic else
                   "user-supplied point clouds/actions of unverified provenance; synthetic state/reward/flags")
    count, point_count, _ = data["xyz"].shape
    assert count == 32 and data["actions"].shape == (32, 7)
    zero_states = np.zeros((count, 30), dtype=np.float32)
    episode_flags = np.zeros((count, 1), dtype=np.bool_)
    episode_flags[15::16] = True
    samples = dict(
        obs=dict(xyz=data["xyz"], rgb=data["rgb"], state=zero_states.copy()),
        next_obs=dict(xyz=data["next_xyz"], rgb=data["next_rgb"], state=zero_states.copy()),
        actions=data["actions"],
        rewards=np.linspace(-0.25, 0.5, count, dtype=np.float32)[:, None],
        dones=np.zeros((count, 1), dtype=np.bool_),
        episode_dones=episode_flags,
    )
    memory = build_replay(dict(type="ReplayMemory", capacity=count))
    memory.push_batch(samples)
    agent = make_agent(point_count)
    encoder = agent.actor.backbone.visual_nn
    assert type(encoder) is PointNetLastIdentity
    assert encoder is agent.critic.values[0].backbone.visual_nn
    parts = dict(encoder=encoder, actor_head=agent.actor.backbone.final_mlp,
                 critic_head=agent.critic.values[0].backbone.final_mlp)
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in parts.items()}
    metrics = agent.update_parameters(memory, updates=0)
    changes = {
        name: sum(float((p.detach() - old).abs().sum())
                  for p, old in zip(module.parameters(), before[name]))
        for name, module in parts.items()
    }
    assert all(value > 0 for value in changes.values()), changes
    assert all(np.isfinite(float(value)) for value in metrics.values()), metrics
    assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())
    grads = [p.grad for p in agent.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert float(metrics["grad/visual_grad"]) > 0

    output = ROOT / "artifacts/tests" / datetime.now(timezone.utc).strftime("identity_ppo_cpu_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "compatibility_only.ckpt"
    save_checkpoint(agent, str(checkpoint_path),
                    optimizer=dict(actor=agent.actor_optim, critic=agent.critic_optim),
                    meta=dict(rl_config=str(DEFAULT_RL_CONFIG),
                              validation_kind="CPU compatibility only: " + data_notice))
    restored = make_agent(point_count)
    checkpoint = load_checkpoint(restored, str(checkpoint_path), map_location="cpu", strict=True)
    restored.actor_optim.load_state_dict(checkpoint["optimizer"]["actor"])
    restored.critic_optim.load_state_dict(checkpoint["optimizer"]["critic"])
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for original_optim, restored_optim in ((agent.actor_optim, restored.actor_optim),
                                         (agent.critic_optim, restored.critic_optim)):
        expected = original_optim.state_dict()
        actual = restored_optim.state_dict()
        assert expected["param_groups"] == actual["param_groups"]
        assert expected["state"].keys() == actual["state"].keys()
        for key, entries in expected["state"].items():
            for name, value in entries.items():
                torch.testing.assert_close(value, actual["state"][key][name], rtol=0, atol=0)
    obs = {key: torch.as_tensor(value[:2]) for key, value in samples["obs"].items()}
    agent.eval()
    restored.eval()
    with torch.no_grad():
        for method in ("actor", "critic"):
            torch.testing.assert_close(getattr(agent, method)(obs, mode="eval"),
                                       getattr(restored, method)(obs, mode="eval"), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    report = dict(
        validation_kind="upstream PPO CPU optimizer/checkpoint compatibility only",
        limitations=[data_notice,
                     "stored actions are not collected by the current identity policy",
                     "no new environment/renderer, no policy quality or convergence validation"],
        fixture=str(args.fixture) if args.fixture else None,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest() if args.fixture else None,
        synthetic_pointclouds_and_actions=args.synthetic,
        config=str(DEFAULT_RL_CONFIG), pointcloud_shape=list(data["xyz"].shape),
        synthetic_fields=(["obs.xyz", "obs.rgb", "next_obs.xyz", "next_obs.rgb", "actions"] if args.synthetic else [])
                         + ["obs.state", "next_obs.state", "rewards", "dones", "episode_dones"],
        smoke_overrides=dict(batch_size=16, num_epoch=1, critic_warmup_epoch=0),
        elapsed_seconds=time.monotonic() - started, device=str(agent.device),
        cuda_initialized=torch.cuda.is_initialized(), cpu_threads=torch.get_num_threads(),
        shared_actor_critic_encoder=True, parameter_l1_change=changes,
        finite_gradients=True, finite_parameters=True, strict_checkpoint_reload=True,
        optimizer_states_reloaded=True, restored_outputs_identical=True,
        checkpoint=str(checkpoint_path), metrics={k: float(v) for k, v in metrics.items()},
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
