"""Measured acceptance checks for the OGBench Franka Panda adapter.

The helpers in this module are intentionally small and import-safe so replay
tests can exercise them without launching the physical rollout suite.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import gymnasium as gym
import mujoco
import numpy as np

import envs  # noqa: F401

ASSET_DIR = REPO / "envs" / "descriptions" / "franka_emika_panda"
ENV_ID = "swm/OGBCubeFranka-v0"


def verify_hash_manifest(asset_dir: Path) -> bool:
    """Verify every SHA256SUMS entry and reject missing/unlisted files."""
    manifest = asset_dir / "SHA256SUMS"
    entries: list[tuple[str, Path]] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative_path = raw_line.split(maxsplit=1)
        relative_path = relative_path.strip().lstrip("*")
        entries.append((digest, REPO.parent / relative_path))
    if not entries:
        return False

    listed: set[Path] = set()
    for expected, path in entries:
        if not path.is_file():
            return False
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            return False
        listed.add(path.resolve())

    actual_files = {
        path.resolve()
        for path in asset_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    return listed == actual_files


def robot_geom_mask(model: mujoco.MjModel) -> np.ndarray:
    """Return a geom mask for bodies in the explicit ``franka/`` namespace."""
    mask = np.zeros(model.ngeom, dtype=bool)
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        root_id = int(model.body_rootid[body_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_id) or ""
        mask[geom_id] = name.startswith("franka/")
    return mask


def joint_limits_ok(
    values: np.ndarray,
    ranges: np.ndarray,
    tolerance: float = 1e-3,
) -> bool:
    """Check joint values against their MuJoCo ranges with a small tolerance."""
    values = np.asarray(values)
    ranges = np.asarray(ranges)
    return bool(
        values.shape == (ranges.shape[0],)
        and np.isfinite(values).all()
        and np.all(values >= ranges[:, 0] - tolerance)
        and np.all(values <= ranges[:, 1] + tolerance)
    )


def gripper_closure(robot) -> float:
    """Return OGBench's normalized closure, where 0 is open and 1 closed."""
    return float(
        np.clip(
            (0.04 - np.mean(robot._data.qpos[robot._gripper_qpos_ids])) / 0.038,
            0.0,
            1.0,
        )
    )


def assert_rollout_equal(
    reference: Iterable[np.ndarray],
    actual: Iterable[np.ndarray],
    atol: float = 1e-10,
) -> None:
    """Assert equal per-step rollout arrays and include the exact error gate."""
    reference = list(reference)
    actual = list(actual)
    assert len(reference) == len(actual), (
        f"replay length mismatch: {len(reference)} != {len(actual)}"
    )
    max_error = 0.0
    for step, (expected, observed) in enumerate(zip(reference, actual)):
        expected = np.asarray(expected)
        observed = np.asarray(observed)
        assert expected.shape == observed.shape, (
            f"replay shape mismatch at step {step}: "
            f"{expected.shape} != {observed.shape}"
        )
        if expected.size:
            max_error = max(max_error, float(np.max(np.abs(expected - observed))))
    assert max_error <= atol, (
        f"replay max absolute error {max_error:.3e} exceeds {atol:.1e}"
    )


def _make_env():
    return gym.make(
        ENV_ID,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
    )


def _physical_rollouts(
    episodes: int,
    seed: int,
    action_noise: float = 0.1,
) -> dict[str, float | int]:
    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

    env = _make_env()
    successes = 0
    lifted = 0
    limit_violations = 0
    self_collisions = 0
    completed = 0
    try:
        model = None
        for episode in range(episodes):
            np.random.seed(seed + episode)
            observation, info = env.reset(seed=seed + episode)
            robot = env.unwrapped
            model, data = robot._model, robot._data
            oracle = CubePlanOracle(
                env=env,
                noise=action_noise,
                noise_smoothing=0.5,
            )
            oracle.reset(observation, info)
            max_z = float(info["privileged/block_0_pos"][2])
            for _ in range(200):
                action = np.clip(oracle.select_action(observation, info), -1.0, 1.0)
                observation, _, _, _, info = env.step(action)
                max_z = max(max_z, float(info["privileged/block_0_pos"][2]))
                limits = model.jnt_range[robot._arm_joint_ids]
                limit_violations += int(
                    not joint_limits_ok(info["proprio/joint_pos"], limits)
                )
                robot_mask = robot_geom_mask(model)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if robot_mask[contact.geom1] and robot_mask[contact.geom2]:
                        body_names = {
                            model.body(int(model.geom_bodyid[geom_id])).name
                            for geom_id in (contact.geom1, contact.geom2)
                        }
                        if body_names == {
                            "franka/left_finger",
                            "franka/right_finger",
                        }:
                            continue
                        self_collisions += 1
                        break
                if oracle.done:
                    successes += int(bool(robot._success))
                    lifted += int(max_z > 0.035)
                    completed += 1
                    observation, info = robot.set_new_target(p_stack=0.0)
                    oracle.reset(observation, info)
                    max_z = float(info["privileged/block_0_pos"][2])
    finally:
        env.close()
    return {
        "source_episodes": episodes,
        "subtasks": completed,
        "action_noise": action_noise,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "lifted": lifted,
        "lift_rate": lifted / max(completed, 1),
        "joint_limit_violations": limit_violations,
        "self_collisions": self_collisions,
    }


def run_verification(episodes: int, seed: int) -> int:
    results: list[str] = []

    def check(name: str, ok: bool, measured: str, expected: str) -> None:
        tag = "PASS" if ok else "FAIL"
        line = f"[{tag}] {name}: {measured} (expected: {expected})"
        print(line, flush=True)
        results.append(line)

    commit_file = ASSET_DIR / "UPSTREAM_COMMIT"
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "panda.xml"))
    check(
        "pinned Menagerie commit",
        commit_file.read_text(encoding="utf-8").strip()
        == "8161bba264d7fa7c99ca301e91e7fb44737676ad",
        commit_file.read_text(encoding="utf-8").strip(),
        "8161bba264d7fa7c99ca301e91e7fb44737676ad",
    )
    check("asset hash manifest", verify_hash_manifest(ASSET_DIR), "verified", "all files")
    check("Panda joint and actuator identity", model.njnt == 9 and model.nu == 8, f"njnt={model.njnt}, nu={model.nu}", "9 joints, 8 actuators")

    env = _make_env()
    try:
        env.reset(seed=seed)
        robot = env.unwrapped
        check("5-D OGBench action", env.action_space.shape == (5,), str(env.action_space.shape), "(5,)")
        check("Franka arm proprioception", robot.compute_ob_info()["proprio/joint_pos"].shape == (7,), str(robot.compute_ob_info()["proprio/joint_pos"].shape), "(7,)")
        robot.set_control(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        closure_after_positive = float(robot._data.ctrl[robot._gripper_actuator_ids[0]])
        robot.set_control(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))
        closure_after_negative = float(robot._data.ctrl[robot._gripper_actuator_ids[0]])
        check("gripper sign parity", closure_after_positive < closure_after_negative, f"ctrl +1={closure_after_positive:.1f}, -1={closure_after_negative:.1f}", "+1 closes in OGBench action space")
    finally:
        env.close()

    rollout = _physical_rollouts(episodes, seed)
    check("plan-oracle physical success", rollout["success_rate"] >= 0.95, f"{rollout['success_rate']:.1%} ({rollout['successes']}/{rollout['subtasks']} subtasks, action_noise={rollout['action_noise']})", ">=95%")
    check("plan-oracle lifted cube", rollout["lift_rate"] >= 0.95, f"{rollout['lift_rate']:.1%}", ">=95%")
    check("joint-limit violations", rollout["joint_limit_violations"] == 0, str(rollout["joint_limit_violations"]), "0")
    check("robot self-collisions", rollout["self_collisions"] == 0, str(rollout["self_collisions"]), "0")

    home = Path(os.environ["STABLEWM_HOME"])
    report = home / "logs" / "franka" / "verify_franka.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(results) + "\n", encoding="utf-8")
    failures = sum(line.startswith("[FAIL]") for line in results)
    print(f"{len(results) - failures} PASS, {failures} FAIL", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()
    return run_verification(args.episodes, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
