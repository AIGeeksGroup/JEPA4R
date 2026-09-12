"""Replay LiftCube demonstrations and record exact cube-robot contact labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeContactRecord:
    episode_id: int
    frame_id: np.ndarray
    cube_robot_contact: np.ndarray
    object_pos_error_m: np.ndarray
    tcp_pos_error_m: np.ndarray


def cube_robot_contact_from_contacts(
    contacts: Iterable[Any],
    *,
    robot_actor_names: set[str],
    cube_name: str = "cube",
) -> bool:
    """Return whether an exact cube-to-robot actor contact is present."""
    for contact in contacts:
        actor0 = str(getattr(getattr(contact, "actor0", None), "name", ""))
        actor1 = str(getattr(getattr(contact, "actor1", None), "name", ""))
        if actor0 == cube_name and actor1 in robot_actor_names:
            return True
        if actor1 == cube_name and actor0 in robot_actor_names:
            return True
    return False


def frame_ids_for_actions(actions: np.ndarray) -> np.ndarray:
    """Map reset and post-action observations to compact frame ids."""
    value = np.asarray(actions)
    if value.ndim != 2:
        raise ValueError(f"actions must have shape [A,D], got {value.shape}")
    return np.arange(value.shape[0] + 1, dtype=np.int64)


def alignment_errors(
    replay_object_pos: np.ndarray,
    reference_object_pos: np.ndarray,
    replay_tcp_pos: np.ndarray,
    reference_tcp_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame Euclidean replay errors for object and TCP positions."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            replay_object_pos,
            reference_object_pos,
            replay_tcp_pos,
            reference_tcp_pos,
        )
    ]
    if any(value.ndim != 2 or value.shape[1] != 3 for value in arrays):
        raise ValueError("object and TCP positions must all have shape [F,3]")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("replay and reference position arrays must have equal shapes")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("replay and reference positions must be finite")
    return (
        np.linalg.norm(arrays[0] - arrays[1], axis=1),
        np.linalg.norm(arrays[2] - arrays[3], axis=1),
    )


def validate_alignment(
    object_pos_error_m: np.ndarray,
    tcp_pos_error_m: np.ndarray,
    *,
    tolerance_m: float,
) -> None:
    if tolerance_m < 0:
        raise ValueError("alignment tolerance must be non-negative")
    object_error = np.asarray(object_pos_error_m, dtype=np.float64)
    tcp_error = np.asarray(tcp_pos_error_m, dtype=np.float64)
    if object_error.ndim != 1 or tcp_error.shape != object_error.shape:
        raise ValueError("alignment error arrays must be same-shape vectors")
    combined = np.concatenate([object_error, tcp_error])
    if not np.isfinite(combined).all() or np.any(combined < 0):
        raise ValueError("alignment errors must be finite and non-negative")
    if np.any(combined > tolerance_m):
        raise ValueError(
            f"replay alignment exceeds tolerance {tolerance_m:.3e} m: "
            f"object max={float(object_error.max()):.3e}, "
            f"TCP max={float(tcp_error.max()):.3e}"
        )


def assemble_sidecar(records: list[EpisodeContactRecord]) -> dict[str, np.ndarray]:
    """Validate, flatten, and deterministically sort episode contact records."""
    if not records:
        raise ValueError("at least one episode contact record is required")
    rows: list[tuple[int, int, int, float, float]] = []
    for record in records:
        arrays = [
            np.asarray(record.frame_id),
            np.asarray(record.cube_robot_contact),
            np.asarray(record.object_pos_error_m),
            np.asarray(record.tcp_pos_error_m),
        ]
        if any(value.ndim != 1 for value in arrays) or len(
            {len(value) for value in arrays}
        ) != 1:
            raise ValueError("episode contact arrays must be same-length vectors")
        if not np.isin(arrays[1], (0, 1)).all():
            raise ValueError("contact labels must contain only 0 and 1")
        for frame_id, contact, object_error, tcp_error in zip(*arrays):
            rows.append(
                (
                    int(record.episode_id),
                    int(frame_id),
                    int(contact),
                    float(object_error),
                    float(tcp_error),
                )
            )
    rows.sort(key=lambda row: (row[0], row[1]))
    keys = [(row[0], row[1]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (episode_id, frame_id) in contact records")
    matrix = np.asarray(rows, dtype=np.float64)
    return {
        "episode_id": matrix[:, 0].astype(np.int64),
        "frame_id": matrix[:, 1].astype(np.int64),
        "cube_robot_contact": matrix[:, 2].astype(np.uint8),
        "object_pos_error_m": matrix[:, 3].astype(np.float64),
        "tcp_pos_error_m": matrix[:, 4].astype(np.float64),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _robot_actor_names(base_env: Any) -> set[str]:
    agent = getattr(base_env, "agent", None)
    robot = getattr(agent, "robot", None)
    names = {
        str(getattr(link, "name", ""))
        for link in list(getattr(robot, "get_links", lambda: [])() or [])
    }
    for item in (
        getattr(agent, "finger1_link", None),
        getattr(agent, "finger2_link", None),
        getattr(base_env, "tcp", None),
    ):
        if item is not None:
            names.add(str(getattr(item, "name", "")))
    names.discard("")
    if not names:
        raise RuntimeError("could not identify robot actor names")
    return names


def _scene_contacts(base_env: Any) -> list[Any]:
    scene = getattr(base_env, "_scene", None)
    if scene is None or not hasattr(scene, "get_contacts"):
        raise RuntimeError("environment does not expose SAPIEN scene contacts")
    return list(scene.get_contacts())


def _positions(base_env: Any) -> tuple[np.ndarray, np.ndarray]:
    obj = getattr(base_env, "obj", None)
    tcp = getattr(base_env, "tcp", None)
    if obj is None or tcp is None:
        raise RuntimeError("LiftCube environment is missing obj or tcp actors")
    object_pos = np.asarray(obj.pose.p, dtype=np.float64)
    tcp_pos = np.asarray(tcp.pose.p, dtype=np.float64)
    if object_pos.shape != (3,) or tcp_pos.shape != (3,):
        raise RuntimeError("object/TCP positions must have shape [3]")
    return object_pos.copy(), tcp_pos.copy()


def _episode_index(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("trajectory JSON must contain an episodes list")
    result: dict[int, dict[str, Any]] = {}
    for position, episode in enumerate(episodes):
        episode_id = int(episode.get("episode_id", position))
        if episode_id in result:
            raise ValueError(f"duplicate episode_id {episode_id} in trajectory JSON")
        result[episode_id] = episode
    return result


def _reset_kwargs(episode: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(episode.get("reset_kwargs") or {})
    if "seed" not in kwargs and "episode_seed" in episode:
        kwargs["seed"] = int(episode["episode_seed"])
    return kwargs


def replay_episode(
    env: Any,
    *,
    episode_id: int,
    episode_metadata: dict[str, Any],
    actions: np.ndarray,
    reference_object_pos: np.ndarray,
    reference_tcp_pos: np.ndarray,
    alignment_tolerance_m: float,
) -> EpisodeContactRecord:
    """Replay one action sequence and validate every observation frame."""
    actions = np.asarray(actions, dtype=np.float32)
    frame_id = frame_ids_for_actions(actions)
    expected_shape = (len(frame_id), 3)
    if np.asarray(reference_object_pos).shape != expected_shape:
        raise ValueError(
            f"episode {episode_id}: object reference must be {expected_shape}"
        )
    if np.asarray(reference_tcp_pos).shape != expected_shape:
        raise ValueError(f"episode {episode_id}: TCP reference must be {expected_shape}")
    env.reset(**_reset_kwargs(episode_metadata))
    base = getattr(env, "unwrapped", env)
    robot_names = _robot_actor_names(base)
    object_positions: list[np.ndarray] = []
    tcp_positions: list[np.ndarray] = []
    contacts: list[bool] = []

    def record_frame() -> None:
        object_pos, tcp_pos = _positions(base)
        object_positions.append(object_pos)
        tcp_positions.append(tcp_pos)
        contacts.append(
            cube_robot_contact_from_contacts(
                _scene_contacts(base), robot_actor_names=robot_names
            )
        )

    record_frame()
    for action in actions:
        env.step(action)
        record_frame()
    replay_object = np.stack(object_positions)
    replay_tcp = np.stack(tcp_positions)
    object_error, tcp_error = alignment_errors(
        replay_object,
        reference_object_pos,
        replay_tcp,
        reference_tcp_pos,
    )
    validate_alignment(
        object_error, tcp_error, tolerance_m=alignment_tolerance_m
    )
    return EpisodeContactRecord(
        episode_id=int(episode_id),
        frame_id=frame_id,
        cube_robot_contact=np.asarray(contacts, dtype=np.uint8),
        object_pos_error_m=object_error,
        tcp_pos_error_m=tcp_error,
    )


def _environment_spec(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    env_info = metadata.get("env_info") or {}
    env_id = str(env_info.get("env_id") or metadata.get("env_id") or "LiftCube-v0")
    env_kwargs = dict(env_info.get("env_kwargs") or {})
    env_kwargs["obs_mode"] = "state_dict"
    env_kwargs.setdefault("control_mode", "pd_ee_delta_pose")
    env_kwargs.pop("render_mode", None)
    return env_id, env_kwargs


def generate_sidecar(
    *,
    h5_path: Path,
    json_path: Path,
    compact_root: Path,
    output_dir: Path,
    episode_ids: list[int] | None,
    alignment_tolerance_m: float,
) -> dict[str, object]:
    """Generate a validated sidecar using the installed ManiSkill runtime."""
    import gymnasium as gym
    import mani_skill2  # noqa: F401
    import mani_skill2.envs  # noqa: F401
    import sapien

    protected_outputs = [
        output_dir / "contact_sidecar.npz",
        output_dir / "metadata.json",
    ]
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(f"refusing to overwrite completed output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = _episode_index(metadata)
    selected = sorted(episodes) if episode_ids is None else [int(x) for x in episode_ids]
    unknown = sorted(set(selected) - set(episodes))
    if unknown:
        raise ValueError(f"episode ids absent from trajectory JSON: {unknown}")
    env_id, env_kwargs = _environment_spec(metadata)
    env = gym.make(env_id, **env_kwargs)
    records: list[EpisodeContactRecord] = []
    episode_summaries: list[dict[str, object]] = []
    try:
        with h5py.File(h5_path, "r") as trajectories:
            for episode_id in selected:
                group_name = f"traj_{episode_id}"
                if group_name not in trajectories:
                    raise KeyError(f"missing HDF5 group {group_name}")
                actions = np.asarray(trajectories[group_name]["actions"], dtype=np.float32)
                compact_path = compact_root / f"episode_{episode_id:04d}.npz"
                with np.load(compact_path, allow_pickle=False) as archive:
                    reference_object = np.asarray(archive["object_pos"], dtype=np.float64)
                    compact_tcp = (
                        np.asarray(archive["tcp_pose"][:, :3], dtype=np.float64)
                        if "tcp_pose" in archive.files
                        else None
                    )
                if compact_tcp is not None:
                    reference_tcp = compact_tcp
                elif "obs/agent/tcp_pose" in trajectories[group_name]:
                    reference_tcp = np.asarray(
                        trajectories[group_name]["obs/agent/tcp_pose"][:, :3],
                        dtype=np.float64,
                    )
                elif "obs/extra/tcp_pose" in trajectories[group_name]:
                    reference_tcp = np.asarray(
                        trajectories[group_name]["obs/extra/tcp_pose"][:, :3],
                        dtype=np.float64,
                    )
                else:
                    raise KeyError(
                        f"episode {episode_id}: no compact or HDF5 TCP pose reference"
                    )
                record = replay_episode(
                    env,
                    episode_id=episode_id,
                    episode_metadata=episodes[episode_id],
                    actions=actions,
                    reference_object_pos=reference_object,
                    reference_tcp_pos=reference_tcp,
                    alignment_tolerance_m=alignment_tolerance_m,
                )
                records.append(record)
                episode_summaries.append(
                    {
                        "episode_id": episode_id,
                        "frame_count": int(len(record.frame_id)),
                        "contact_frame_count": int(record.cube_robot_contact.sum()),
                        "max_object_pos_error_m": float(
                            record.object_pos_error_m.max()
                        ),
                        "max_tcp_pos_error_m": float(record.tcp_pos_error_m.max()),
                    }
                )
                print(json.dumps(episode_summaries[-1], sort_keys=True), flush=True)
    finally:
        env.close()

    arrays = assemble_sidecar(records)
    sidecar_path = output_dir / "contact_sidecar.npz"
    _atomic_write_npz(sidecar_path, arrays)
    report: dict[str, object] = {
        "schema_version": 1,
        "task": "LiftCube-v0",
        "h5_path": str(h5_path.resolve()),
        "h5_sha256": _sha256_file(h5_path),
        "json_path": str(json_path.resolve()),
        "json_sha256": _sha256_file(json_path),
        "compact_root": str(compact_root.resolve()),
        "environment": {
            "env_id": env_id,
            "kwargs": env_kwargs,
            "gymnasium_version": getattr(gym, "__version__", "unknown"),
            "maniskill2_version": getattr(mani_skill2, "__version__", "unknown"),
            "sapien_version": getattr(sapien, "__version__", "unknown"),
        },
        "alignment_tolerance_m": float(alignment_tolerance_m),
        "episode_count": len(selected),
        "frame_count": int(len(arrays["frame_id"])),
        "contact_frame_count": int(arrays["cube_robot_contact"].sum()),
        "max_object_pos_error_m": float(arrays["object_pos_error_m"].max()),
        "max_tcp_pos_error_m": float(arrays["tcp_pos_error_m"].max()),
        "episodes": episode_summaries,
    }
    _atomic_write_json(output_dir / "metadata.json", report)
    report["sidecar_sha256"] = _sha256_file(sidecar_path)
    _atomic_write_json(output_dir / "metadata.json", report)
    return report


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument("--alignment-tolerance-m", type=float, default=1e-4)
    args = parser.parse_args(argv)
    report = generate_sidecar(
        h5_path=args.h5,
        json_path=args.json,
        compact_root=args.compact_root,
        output_dir=args.output_dir,
        episode_ids=args.episode_ids,
        alignment_tolerance_m=args.alignment_tolerance_m,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
