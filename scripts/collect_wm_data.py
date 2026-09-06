"""离线数据采集（LeWM 附录 E 的 OGBench-Cube 协议）：PickCube（ManiSkill2）+ 带噪脚本化抓放启发式。

- 定长回合（默认 200 步），不因 success 提前结束；行为策略不追求最优，只需覆盖环境动力学；
- 启发式循环：到方块上方 -> 下降 -> 合爪 -> 抬起 -> 搬到目标（环境目标点或随机点）-> 松开 -> 重复；每条回合抽一档噪声 σ；
- 每步记录：pprl 包装后的观测（点云 + 状态，与 encoder 训练时相同）、动作、仿真器状态（规划评测时 set_state 复位用）、
  特权量（方块 / 夹爪 / 目标位置、是否抓住）、info["success"]。
输出 out_dir/ep_XXXXXX.npz。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.spatial.transform import Rotation

APPROACH, DESCEND, GRASP, LIFT, CARRY, RELEASE, RETREAT = range(7)


class PickHeuristic:
    """基于特权信息的抓放启发式，输出 pd_ee_delta_pose 的 7 维动作 [dx, dy, dz, rx, ry, rz, gripper]。
    ManiSkill2 该控制器的位移增量在末端坐标系下（target = ee_pose * delta），满动作 = 0.1 m；夹爪 +1 开 / -1 合。"""

    POS_BOUND = 0.1
    WORKSPACE_LO = np.array([-0.15, -0.20, 0.02])
    WORKSPACE_HI = np.array([0.15, 0.20, 0.35])

    def __init__(self, rng: np.random.Generator, noise: float, goal_frac: float):
        self.rng, self.noise, self.goal_frac = rng, noise, goal_frac
        self.phase, self.phase_steps = APPROACH, 0
        self.speed = rng.uniform(0.25, 0.7)
        self.hover = rng.uniform(0.06, 0.10)
        self.carry_target = None

    def _set_phase(self, phase):
        self.phase, self.phase_steps = phase, 0
        self.speed = self.rng.uniform(0.25, 0.7)

    def __call__(self, obj_pos, tcp_pos, tcp_quat_wxyz, goal_pos, grasped: bool) -> np.ndarray:
        self.phase_steps += 1
        gripper = 1.0
        if self.phase == APPROACH:
            target = obj_pos + np.array([0.0, 0.0, self.hover])
            if (np.linalg.norm((target - tcp_pos)[:2]) < 0.012 and abs(target[2] - tcp_pos[2]) < 0.02) or self.phase_steps > 40:
                self._set_phase(DESCEND)
        elif self.phase == DESCEND:
            target = obj_pos + np.array([0.0, 0.0, 0.005])
            if np.linalg.norm(target - tcp_pos) < 0.012 or self.phase_steps > 25:
                self._set_phase(GRASP)
        elif self.phase == GRASP:
            target, gripper = tcp_pos, -1.0
            if self.phase_steps >= 4:
                self._set_phase(LIFT)
        elif self.phase == LIFT:
            target, gripper = np.array([tcp_pos[0], tcp_pos[1], obj_pos[2] + 0.12]), -1.0
            if self.phase_steps >= 3 and not grasped:
                self._set_phase(APPROACH)
            elif tcp_pos[2] > 0.10 or self.phase_steps > 20:
                self.carry_target = goal_pos.copy() if self.rng.uniform() < self.goal_frac else self.rng.uniform(self.WORKSPACE_LO, self.WORKSPACE_HI)
                self._set_phase(CARRY)
        elif self.phase == CARRY:
            target, gripper = self.carry_target, -1.0
            if not grasped and self.phase_steps > 2:
                self._set_phase(APPROACH)
            elif np.linalg.norm(target - tcp_pos) < 0.02 or self.phase_steps > 40:
                self._set_phase(RELEASE)
        elif self.phase == RELEASE:
            target = tcp_pos
            if self.phase_steps >= 3:
                self._set_phase(RETREAT)
        else:  # RETREAT
            target = tcp_pos + np.array([0.0, 0.0, 0.06])
            if self.phase_steps >= 4:
                self._set_phase(APPROACH)

        delta_world = np.clip(target - tcp_pos, -self.POS_BOUND, self.POS_BOUND)
        max_step = self.speed * self.POS_BOUND
        norm = np.linalg.norm(delta_world)
        if norm > max_step:
            delta_world = delta_world / norm * max_step
        rot = Rotation.from_quat(np.asarray(tcp_quat_wxyz)[[1, 2, 3, 0]])  # sapien (w,x,y,z) -> scipy (x,y,z,w)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = rot.inv().apply(delta_world) / self.POS_BOUND
        action[:3] += self.rng.normal(0.0, self.noise, 3)
        action[3:6] = self.rng.normal(0.0, 0.2 * self.noise, 3)
        action[6] = gripper
        return np.clip(action, -1.0, 1.0)


def run_episode(env, ep_id: int, cfg: dict):
    rng = np.random.default_rng(cfg["seed"] + ep_id)
    noise = float(rng.choice(cfg["noise_levels"]))
    policy = PickHeuristic(rng, noise, cfg["goal_frac"])
    obs, _ = env.reset(seed=cfg["seed"] + ep_id)
    u = env.unwrapped

    def priv():
        grasped = bool(u.agent.check_grasp(u.obj, max_angle=30))
        return np.concatenate([u.obj.pose.p, u.tcp.pose.p, u.goal_pos, [grasped]]).astype(np.float32), grasped

    p, grasped = priv()
    points, states, env_states, actions, success, privs = [obs["points"]], [obs["state"]], [u.get_state()], [], [], [p]
    for _ in range(cfg["steps"]):
        action = policy(u.obj.pose.p, u.tcp.pose.p, u.tcp.pose.q, u.goal_pos, grasped)
        obs, _, _, _, info = env.step(action)
        p, grasped = priv()
        actions.append(action); points.append(obs["points"]); states.append(obs["state"])
        env_states.append(u.get_state()); success.append(bool(info["success"])); privs.append(p)
    lengths = np.array([len(x) for x in points])
    return dict(
        points=np.concatenate(points, 0).astype(np.float16),      # 全部帧点云拼接 [N_total, 3]
        ptr=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),  # 帧 k 的点 = points[ptr[k]:ptr[k+1]]
        state=np.stack(states).astype(np.float32),                # [steps+1, S]
        env_state=np.stack(env_states).astype(np.float32),        # [steps+1, S_env]，set_state 可复位
        action=np.stack(actions).astype(np.float32),              # [steps, 7]
        success=np.asarray(success, dtype=bool),                  # [steps]
        priv=np.stack(privs).astype(np.float32),                  # [steps+1, 10]: obj xyz, tcp xyz, goal xyz, grasped
        noise=np.float32(noise),
    )


def worker(worker_id: int, ep_ids: list[int], env_cfg: dict, cfg: dict):
    from hydra.utils import instantiate

    env = instantiate(OmegaConf.create(env_cfg), _convert_="partial")
    out_dir = Path(cfg["out_dir"])
    t0, n_done = time.time(), 0
    for ep_id in ep_ids:
        path = out_dir / f"ep_{ep_id:06d}.npz"
        if path.exists():
            continue
        ep = run_episode(env, ep_id, cfg)
        np.savez_compressed(path, **ep)
        n_done += 1
        if n_done == 1 or n_done % 20 == 0:
            print(f"[worker {worker_id}] {n_done}/{len(ep_ids)}  {(time.time() - t0) / n_done:.1f}s/ep  last: σ={ep['noise']:.2f} "
                  f"grasped {ep['priv'][:, -1].mean() * 100:.0f}% steps, success {ep['success'].mean() * 100:.0f}% steps, "
                  f"points/frame {np.diff(ep['ptr']).mean():.0f}", flush=True)
    env.close()


@hydra.main(version_base=None, config_path="../conf", config_name="collect_wm_data")
def main(config: DictConfig) -> None:
    with open_dict(config.env):
        config.env.pop("name", None)
        config.env.pop("traj_info", None)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    cfg = OmegaConf.to_container(config.collect, resolve=True)
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"env": env_cfg, "collect": cfg}, ensure_ascii=False, indent=2))
    todo = [i for i in range(int(cfg["n_episodes"])) if not (out_dir / f"ep_{i:06d}.npz").exists()]
    n_workers = int(cfg["n_workers"])
    print(f"共 {cfg['n_episodes']} 条，待采 {len(todo)} 条，{n_workers} 个进程，每条 {cfg['steps']} 步 -> {out_dir}")
    if not todo:
        return
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(w, todo[w::n_workers], env_cfg, cfg)) for w in range(n_workers) if todo[w::n_workers]]
    t0 = time.time()
    for p in procs: p.start()
    for p in procs: p.join()
    files = list(out_dir.glob("ep_*.npz"))
    print(f"完成：{len(files)} 条，{sum(f.stat().st_size for f in files) / 1e9:.2f} GB，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
