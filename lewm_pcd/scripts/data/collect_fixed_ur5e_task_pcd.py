"""Collect accepted fixed-task UR5e point-cloud episodes with resumable batches."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import fcntl
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from functools import partial

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.policy import RandomPolicy

from envs import FixedTaskCubePlanPolicy, PointCloudWrapper
from envs.ur5e_cube_tasks import get_task
from scripts.data.accepted_episode_h5 import (
    copy_accepted_episodes,
    validate_labeled_dataset,
)
from scripts.data.merge_h5_shards import append_shard


@dataclass(frozen=True)
class CollectionSpec:
    family: str
    task_id: int
    task_name: str
    policy_kind: str
    desired_outcome: str
    episodes: int
    seed: int
    num_envs: int
    batch_size: int
    max_attempts: int
    max_episode_steps: int

    @classmethod
    def create(
        cls,
        *,
        family,
        task_id,
        policy_kind,
        episodes,
        seed,
        num_envs,
        batch_size,
        max_attempts,
    ):
        task = get_task(family, task_id)
        if policy_kind not in {"expert", "random"}:
            raise ValueError("policy_kind must be expert or random")
        for name, value in {
            "episodes": episodes,
            "num_envs": num_envs,
            "batch_size": batch_size,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if max_attempts is None:
            max_attempts = int(episodes) * (10 if policy_kind == "expert" else 2)
        if int(max_attempts) < int(episodes):
            raise ValueError("max_attempts must be at least episodes")
        return cls(
            family=task.family,
            task_id=task.task_id,
            task_name=task.name,
            policy_kind=policy_kind,
            desired_outcome="success" if policy_kind == "expert" else "failure",
            episodes=int(episodes),
            seed=int(seed),
            num_envs=int(num_envs),
            batch_size=int(batch_size),
            max_attempts=int(max_attempts),
            max_episode_steps=task.max_episode_steps,
        )

    @property
    def world_kwargs(self):
        return {
            "env_type": self.family,
            "reward_task_id": self.task_id,
            "permute_blocks": False,
            "ob_type": "states",
            "multiview": False,
            "visualize_info": False,
            "terminate_at_goal": False,
            "mode": "task",
        }


@dataclass
class CollectorState:
    spec: dict
    attempts: int = 0
    accepted: int = 0
    batches: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, spec: CollectionSpec):
        return cls(spec=asdict(spec))

    @property
    def remaining(self):
        return int(self.spec["episodes"]) - self.accepted

    def assert_compatible(self, spec: CollectionSpec):
        if self.spec != asdict(spec):
            raise ValueError("resume arguments do not match collector state")

    def record_batch(self, *, attempted: int, accepted: int, shard: str | None):
        if attempted <= 0:
            raise ValueError("attempted must be positive")
        if accepted < 0 or accepted > attempted or accepted > self.remaining:
            raise ValueError("accepted batch count is impossible")
        self.batches.append(
            {
                "batch_index": len(self.batches),
                "attempted": int(attempted),
                "accepted": int(accepted),
                "shard": shard,
            }
        )
        self.attempts += int(attempted)
        self.accepted += int(accepted)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    with open(partial_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)


def load_state(path: Path) -> CollectorState:
    with open(path, encoding="utf-8") as handle:
        return CollectorState.from_dict(json.load(handle))


def _collect_candidate(spec: CollectionSpec, path: Path, episodes: int, seed: int):
    kwargs = dict(spec.world_kwargs)
    world = swm.World(
        "swm/OGBCubePCD-v0",
        num_envs=min(spec.num_envs, episodes),
        max_episode_steps=spec.max_episode_steps,
        image_shape=(224, 224),
        env_type=kwargs.pop("env_type"),
        add_pixels=False,
        pre_wrappers=[
            partial(
                PointCloudWrapper,
                num_points=1200,
                foreground_only=True,
                crop_box=((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
                voxel_size=0.005,
                cube_fraction=0.25,
                keep_rgb=False,
            )
        ],
        **kwargs,
    )
    if spec.policy_kind == "expert":
        policy = FixedTaskCubePlanPolicy(spec.family, spec.task_id)
    else:
        policy = RandomPolicy(seed=seed + 17)
    world.set_policy(policy)
    try:
        world.collect(
            writer=HDF5Writer(path, mode="error"),
            episodes=episodes,
            seed=seed,
            progress=True,
        )
    finally:
        if hasattr(world.envs, "close"):
            world.envs.close()


def _commit(root: Path):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def merge_and_validate(spec: CollectionSpec, state: CollectorState, work_dir: Path, out: Path):
    shards = [work_dir / batch["shard"] for batch in state.batches if batch["shard"]]
    if not shards:
        raise RuntimeError("no accepted shards were produced")
    partial_path = out.with_name(out.name + ".partial")
    partial_path.unlink(missing_ok=True)
    shutil.copy2(shards[0], partial_path)
    for shard in shards[1:]:
        append_shard(partial_path, shard)
    policy_counts = {
        "expert": spec.episodes if spec.policy_kind == "expert" else 0,
        "random": spec.episodes if spec.policy_kind == "random" else 0,
    }
    report = validate_labeled_dataset(
        partial_path,
        family=spec.family,
        task_id=spec.task_id,
        expected_episodes=spec.episodes,
        expected_policy_counts=policy_counts,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, out)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["double", "triple"])
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--policy", required=True, choices=["expert", "random"])
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup-work", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    spec = CollectionSpec.create(
        family=args.family,
        task_id=args.task_id,
        policy_kind=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
    )
    out = args.out.resolve()
    work_dir = out.with_name(out.name + ".work")
    state_path = work_dir / "collector_state.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"cannot resume without {state_path}")
        state = load_state(state_path)
        state.assert_compatible(spec)
    else:
        if work_dir.exists():
            raise SystemExit(f"refusing to reuse existing work directory: {work_dir}")
        work_dir.mkdir(parents=True)
        state = CollectorState.new(spec)
        write_json_atomic(state_path, state.to_dict())

    lock_path = work_dir / ".lock"
    with open(lock_path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another collector holds {lock_path}") from exc
        while state.remaining > 0:
            available_attempts = spec.max_attempts - state.attempts
            if available_attempts <= 0:
                raise RuntimeError(
                    f"attempt cap reached: accepted {state.accepted}/{spec.episodes} "
                    f"after {state.attempts} attempts"
                )
            attempted = min(spec.batch_size, available_attempts)
            batch_index = len(state.batches)
            candidate = work_dir / f"candidate_{batch_index:04d}.h5"
            accepted_shard = work_dir / f"accepted_{batch_index:04d}.h5"
            candidate.unlink(missing_ok=True)
            accepted_shard.unlink(missing_ok=True)
            batch_seed = spec.seed + state.attempts
            _collect_candidate(spec, candidate, attempted, batch_seed)
            accepted = copy_accepted_episodes(
                candidate,
                accepted_shard,
                outcome=spec.desired_outcome,
                limit=state.remaining,
                family=spec.family,
                task_id=spec.task_id,
                policy_kind=spec.policy_kind,
            )
            candidate.unlink()
            state.record_batch(
                attempted=attempted,
                accepted=accepted,
                shard=accepted_shard.name if accepted else None,
            )
            write_json_atomic(state_path, state.to_dict())
            print(
                f"accepted {state.accepted}/{spec.episodes}; "
                f"attempts {state.attempts}/{spec.max_attempts}",
                flush=True,
            )

        report = merge_and_validate(spec, state, work_dir, out)
        root = Path(__file__).resolve().parents[2]
        manifest = {
            "version": 1,
            "code_commit": _commit(root),
            "output": str(out),
            "spec": asdict(spec),
            "attempts": state.attempts,
            "rejected": state.attempts - state.accepted,
            "report": report,
        }
        write_json_atomic(out.with_suffix(out.suffix + ".manifest.json"), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    if args.cleanup_work:
        resolved_work = work_dir.resolve()
        if resolved_work.parent != out.parent or resolved_work.name != out.name + ".work":
            raise RuntimeError(f"refusing unsafe cleanup: {resolved_work}")
        shutil.rmtree(resolved_work)


if __name__ == "__main__":
    main()
