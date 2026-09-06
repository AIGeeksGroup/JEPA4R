"""下游规划评测（LeWM 官方 eval 协议）。

- 起点：从数据集回合里随机抽 (回合, 起始步)，用记录的仿真器状态 set_state 复位；
- 目标：同一回合 goal_offset_steps（25）步之后的观测，编码为目标 latent1；
- 规划：CEM（300 样本 / 30 轮 / top-30 / 初始方差 1），视界 5 个动作块（每块 frameskip=5 步），
        代价 = 世界模型自回归 rollout 的末态 latent1 与目标 latent1 的 MSE（按维求和）；整段执行后重规划；预算 50 步；
- 成功：末态方块位置与目标帧方块位置的距离 < success_thresh；另可跑 policy=random 基线。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from pprl.data.wm_dataset import ColumnNormalizer, load_episode
from pprl.models.latent_world_model import LatentWorldModel


def load_world_model(ckpt_path: str, device) -> tuple[LatentWorldModel, ColumnNormalizer, dict]:
    from train_latent_wm import build_model  # 同目录脚本

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    dims = ckpt["dims"]
    model = build_model(cfg, dims["point_dim"], dims["state_dim"], dims["action_dim"] * int(cfg.data.frameskip))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval().requires_grad_(False)
    norm = ColumnNormalizer(np.asarray(ckpt["action_norm"]["mean"]), np.asarray(ckpt["action_norm"]["std"]))
    return model, norm, ckpt["config"]


def wrapped_observation(env, raw_obs):
    """set_state 之后从底层观测重建 pprl 包装后的观测：按包装链从内到外依次应用 observation()。"""
    chain = []
    e = env
    while hasattr(e, "env"):
        chain.append(e)
        e = e.env
    obs = raw_obs
    for w in reversed(chain):
        if isinstance(w, gym.ObservationWrapper):
            obs = w.observation(obs)
    return obs


def obs_to_batch(obs_list: list[dict], device) -> dict:
    points = [o["points"].astype(np.float32) for o in obs_list]
    lengths = np.array([len(p) for p in points])
    return {
        "points": {"pos": torch.from_numpy(np.concatenate(points, 0)).to(device),
                   "ptr": torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)])).long().to(device)},
        "state": torch.from_numpy(np.stack([o["state"] for o in obs_list]).astype(np.float32)).to(device),
    }


@torch.no_grad()
def cem_plan(model, emb_hist, act_hist, goal_emb, horizon, block_dim, solver, gen, device):
    """emb_hist (1, H, D)，act_hist (1, H, A) 已执行的历史动作块（标准化空间），goal_emb (1, D) -> 最优动作块序列 (horizon, A)"""
    S = int(solver.num_samples)
    mean = torch.zeros(horizon, block_dim, device=device)
    std = torch.full((horizon, block_dim), float(solver.var_scale), device=device)
    emb_rep, act_rep = emb_hist.expand(S, -1, -1), act_hist.expand(S, -1, -1)
    for _ in range(int(solver.n_steps)):
        cand = mean + std * torch.randn(S, horizon, block_dim, device=device, generator=gen)
        rollout = model.rollout(emb_rep, torch.cat([act_rep, cand], dim=1))
        cost = (rollout[:, -1] - goal_emb).pow(2).sum(-1)  # 官方 criterion：末态与目标的 MSE 按维求和
        elite = cand[cost.topk(int(solver.topk), largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return mean


@hydra.main(version_base=None, config_path="../conf", config_name="plan_latent_wm")
def main(config: DictConfig) -> None:
    device = torch.device(config.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(config.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, out_dir / "config.yaml")
    rng = np.random.default_rng(int(config.seed))
    gen = torch.Generator(device=device).manual_seed(int(config.seed))

    with open_dict(config.env):
        config.env.pop("name", None); config.env.pop("traj_info", None)
    env = instantiate(config.env, _convert_="partial")
    u = env.unwrapped
    use_model = config.policy == "model"
    if use_model:
        model, action_norm, _ = load_world_model(config.ckpt, device)
        H = model.history_size

    files = sorted(Path(config.data.dir).glob("ep_*.npz"))[-int(config.data.n_val_episodes):]
    goal_offset, budget = int(config.eval.goal_offset_steps), int(config.eval.eval_budget)
    block, horizon = int(config.plan_config.action_block), int(config.plan_config.horizon)
    results = []
    t0 = time.time()
    for i in range(int(config.eval.num_eval)):
        ep = load_episode(files[rng.integers(len(files))])
        n_frames = len(ep["ptr"]) - 1
        start = int(rng.integers(0, n_frames - goal_offset))
        goal_idx = start + goal_offset
        goal_cube = ep["priv"][goal_idx, :3]

        env.reset(seed=int(config.seed) + i)
        u.set_state(ep["env_state"][start])
        obs = wrapped_observation(env, u.get_obs())
        if use_model:
            goal_obs = {"points": ep["points"][ep["ptr"][goal_idx]: ep["ptr"][goal_idx + 1]], "state": ep["state"][goal_idx]}
            goal_emb = model.encode(obs_to_batch([goal_obs], device))
            obs_hist = [obs] * H                                   # 历史用初始观测填满
            act_hist = torch.zeros(1, H, block * 7, device=device)  # 对应的动作块置 0（官方边界 NaN -> 0）
        steps = 0
        while steps < budget:
            if use_model:
                emb_hist = model.encode(obs_to_batch(obs_hist[-H:], device)).unsqueeze(0)
                plan = cem_plan(model, emb_hist, act_hist[:, -H:], goal_emb, horizon, block * 7, config.solver, gen, device)
                n_exec = min(int(config.plan_config.receding_horizon), horizon)
            else:
                plan, n_exec = None, 1
            for b in range(n_exec):
                if steps >= budget:
                    break
                if use_model:
                    raw = plan[b].cpu().numpy().reshape(block, 7) * action_norm.std + action_norm.mean  # 反标准化
                    raw = np.clip(raw, -1.0, 1.0)
                else:
                    raw = rng.uniform(-1.0, 1.0, size=(block, 7)).astype(np.float32)
                for a in raw:
                    obs, _, _, _, _ = env.step(a.astype(np.float32))
                    steps += 1
                    if steps >= budget:
                        break
                if use_model:
                    obs_hist.append(obs)
                    act_hist = torch.cat([act_hist, plan[b : b + 1].unsqueeze(0)], dim=1)
        dist = float(np.linalg.norm(u.obj.pose.p - goal_cube))
        results.append({"episode": i, "start": start, "final_dist": dist, "success": dist < float(config.eval.success_thresh)})
        print(f"[{i + 1}/{config.eval.num_eval}] dist {dist:.3f} success {results[-1]['success']} ({time.time() - t0:.0f}s)", flush=True)
    env.close()
    summary = {"policy": config.policy, "success_rate": float(np.mean([r["success"] for r in results])),
               "mean_final_dist": float(np.mean([r["final_dist"] for r in results])), "n": len(results)}
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
