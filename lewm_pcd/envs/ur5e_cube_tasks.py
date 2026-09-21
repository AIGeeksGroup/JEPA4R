"""Canonical fixed OGBench Cube tasks used by the UR5e multitask pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _xyz(rows: list[list[float]]) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class CubeTask:
    family: str
    task_id: int
    name: str
    init_xyzs: np.ndarray
    goal_xyzs: np.ndarray

    @property
    def env_type(self) -> str:
        return self.family

    @property
    def slug(self) -> str:
        return f"cube_{self.family}_task{self.task_id}_{self.name}"

    @property
    def max_episode_steps(self) -> int:
        return 20 + 100 * len(build_subtask_plan(self))

    @property
    def num_cubes(self) -> int:
        return 2 if self.family == "double" else 3

    @property
    def qpos_dim(self) -> int:
        return 14 + 7 * self.num_cubes

    @property
    def qvel_dim(self) -> int:
        return 14 + 6 * self.num_cubes

    @property
    def control_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def observation_dim(self) -> int:
        return 19 + 9 * self.num_cubes


@dataclass(frozen=True)
class SubtaskStep:
    block_id: int
    target_xyz: np.ndarray
    kind: str = "goal"


TASKS = (
    CubeTask(
        "double",
        1,
        "single_pnp",
        _xyz([[0.425, 0.0, 0.02], [0.425, -0.1, 0.02]]),
        _xyz([[0.425, 0.0, 0.02], [0.425, 0.1, 0.02]]),
    ),
    CubeTask(
        "double",
        2,
        "double_pnp1",
        _xyz([[0.35, -0.1, 0.02], [0.50, -0.1, 0.02]]),
        _xyz([[0.35, 0.1, 0.02], [0.50, 0.1, 0.02]]),
    ),
    CubeTask(
        "double",
        3,
        "double_pnp2",
        _xyz([[0.35, 0.0, 0.02], [0.50, 0.0, 0.02]]),
        _xyz([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]]),
    ),
    CubeTask(
        "double",
        4,
        "swap",
        _xyz([[0.425, -0.1, 0.02], [0.425, 0.1, 0.02]]),
        _xyz([[0.425, 0.1, 0.02], [0.425, -0.1, 0.02]]),
    ),
    CubeTask(
        "double",
        5,
        "stack",
        _xyz([[0.425, -0.2, 0.02], [0.425, 0.2, 0.02]]),
        _xyz([[0.425, 0.0, 0.02], [0.425, 0.0, 0.06]]),
    ),
    CubeTask(
        "triple",
        1,
        "single_pnp",
        _xyz([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.50, -0.1, 0.02]]),
        _xyz([[0.35, -0.1, 0.02], [0.35, 0.1, 0.02], [0.50, 0.1, 0.02]]),
    ),
    CubeTask(
        "triple",
        2,
        "triple_pnp",
        _xyz([[0.35, -0.2, 0.02], [0.35, 0.0, 0.02], [0.35, 0.2, 0.02]]),
        _xyz([[0.50, 0.0, 0.02], [0.50, 0.2, 0.02], [0.50, -0.2, 0.02]]),
    ),
    CubeTask(
        "triple",
        3,
        "pnp_from_stack",
        _xyz([[0.425, 0.2, 0.02], [0.425, 0.2, 0.06], [0.425, 0.2, 0.10]]),
        _xyz([[0.35, -0.1, 0.02], [0.50, -0.2, 0.02], [0.50, 0.0, 0.02]]),
    ),
    CubeTask(
        "triple",
        4,
        "cycle",
        _xyz([[0.35, 0.0, 0.02], [0.50, -0.1, 0.02], [0.50, 0.1, 0.02]]),
        _xyz([[0.50, -0.1, 0.02], [0.50, 0.1, 0.02], [0.35, 0.0, 0.02]]),
    ),
)

_BY_KEY = {(task.family, task.task_id): task for task in TASKS}
_BUFFER_XYZ = np.array([0.325, 0.27, 0.02], dtype=np.float64)


def _normalize_family(family: str) -> str:
    return family.lower().removeprefix("cube-").removeprefix("cube_")


def get_task(family: str, task_id: int) -> CubeTask:
    key = (_normalize_family(family), int(task_id))
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unsupported UR5e cube task: {key[0]} task{key[1]}") from exc


def _step(task: CubeTask, block_id: int, *, target=None, kind="goal") -> SubtaskStep:
    xyz = task.goal_xyzs[block_id] if target is None else target
    return SubtaskStep(block_id=block_id, target_xyz=np.array(xyz, copy=True), kind=kind)


def build_subtask_plan(task: CubeTask) -> list[SubtaskStep]:
    """Return a collision-safe order of single-cube oracle moves."""

    key = (task.family, task.task_id)
    if key == ("double", 1):
        return [_step(task, 1)]
    if key in {("double", 2), ("double", 3), ("double", 5)}:
        return [_step(task, 0), _step(task, 1)]
    if key == ("double", 4):
        return [
            _step(task, 0, target=_BUFFER_XYZ, kind="buffer"),
            _step(task, 1),
            _step(task, 0),
        ]
    if key == ("triple", 1):
        return [_step(task, 2)]
    if key == ("triple", 2):
        return [_step(task, 0), _step(task, 1), _step(task, 2)]
    if key == ("triple", 3):
        return [_step(task, 2), _step(task, 1), _step(task, 0)]
    if key == ("triple", 4):
        return [
            _step(task, 0, target=_BUFFER_XYZ, kind="buffer"),
            _step(task, 2),
            _step(task, 1),
            _step(task, 0),
        ]
    raise KeyError(f"unsupported UR5e cube task: {task.family} task{task.task_id}")
