"""Opt-in recording of one episode from the actual upstream training rollout.

The hook observes ``UnifiedVectorEnvAPI.step`` inside ``step_dict``, before its
automatic reset. It copies the real XYZ+RGB observation consumed by the policy,
without rendering, stepping, resetting, sampling actions, or using an evaluator.
After the first complete worker-0 episode it removes itself; a CPU thread creates
a fixed-view point-cloud video from the captured observations.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np


_MISSING = object()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _encode_video(path, pointclouds, fps):
    from ms2_lewm.pointcloud_video import encode_pointcloud_video

    return encode_pointcloud_video(path, pointclouds, fps=fps)


def _batch_item(value, position):
    if isinstance(value, dict):
        return {key: _batch_item(item, position) for key, item in value.items()}
    return np.asarray(value)[position]


class TrainingEpisodeRecorder:
    """Read-only observer for worker 0 of an already constructed training rollout."""

    def __init__(self, rollout, output_path, *, fps=20, metadata=None, encoder=None):
        self.vec_env = rollout.vec_env
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(f"Refusing to overwrite training video: {self.output_path}")
        self.fps = fps
        self.encoder = _encode_video if encoder is None else encoder
        self.frames = []
        self.actions, self.rewards, self.terminated, self.truncated = [], [], [], []
        self.active = True
        self.thread = None
        self.original_step = self.vec_env.step
        self.previous_step_attribute = vars(self.vec_env).get("step", _MISSING)
        self.metadata = dict(metadata or {})
        self.metadata.update(
            source="actual PPO training rollout.vec_env.step (inside upstream step_dict)",
            worker_index=0,
            num_training_envs=rollout.num_envs,
            episode_selection="first complete training episode of worker 0",
            policy_phase="initial training; this video does not establish convergence",
            separate_evaluation=False,
            observation="actual policy XYZ+RGB point cloud, before PointNet",
            visualization="fixed 3D view of the captured input points; no extra environment rendering",
            pointcloud_dtype="float32",
            fps=fps,
            video_path=str(self.output_path),
            status="waiting_for_first_training_step",
        )
        self._write_metadata()
        self.vec_env.step = self._observed_step

    def _write_metadata(self):
        target = self.output_path.with_suffix(".json")
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_jsonable(self.metadata), indent=2) + "\n")
        temporary.replace(target)

    def _capture_frame(self, observation, position):
        xyz = np.asarray(observation["xyz"][position])
        rgb = np.asarray(observation["rgb"][position])
        if xyz.ndim != 2 or xyz.shape[-1] != 3 or rgb.shape != xyz.shape:
            raise ValueError(f"Expected matching [N,3] XYZ/RGB inputs, got {xyz.shape}/{rgb.shape}")
        frame = np.concatenate((xyz, rgb), axis=-1).astype(np.float32, copy=True)
        self.frames.append(frame)
        self.metadata["points_per_frame"] = len(xyz)
        if len(self.frames) > 10001:
            raise RuntimeError("No complete episode after 10000 steps; recording disabled")

    def _begin(self):
        self.metadata.update(status="recording", started_unix=time.time())
        try:
            self.metadata["initial_sim_state"] = _jsonable(self.vec_env.get_state(idx=np.array([0], dtype=np.int32)))
        except Exception as error:
            self.metadata["initial_sim_state_unavailable"] = f"{type(error).__name__}: {error}"
        self._capture_frame(self.vec_env.recent_obs, 0)
        self._write_metadata()

    def _observed_step(self, actions, idx=None):
        selected = np.arange(self.vec_env.num_envs) if idx is None else np.asarray(idx).reshape(-1)
        worker_positions = np.flatnonzero(selected == 0)
        position = int(worker_positions[0]) if len(worker_positions) else None
        if self.active and position is not None and not self.frames:
            try:
                self._begin()
            except Exception as error:
                self._fail(error)

        # Exactly one original step, with identical action and index objects.
        # Environment/training exceptions propagate; only recorder errors are caught.
        result = self.original_step(actions, idx=idx)
        if self.active and position is not None:
            try:
                next_obs, rewards, terminated, truncated, infos = result
                self._capture_frame(next_obs, position)  # terminal input, before auto-reset
                self.actions.append(np.asarray(actions[position]).copy())
                self.rewards.append(np.asarray(rewards[position]).copy())
                self.terminated.append(np.asarray(terminated[position]).copy())
                self.truncated.append(np.asarray(truncated[position]).copy())
                if bool(np.any(terminated[position]) or np.any(truncated[position])):
                    self.metadata["final_info"] = _jsonable(_batch_item(infos, position))
                    self._complete()
            except Exception as error:
                self._fail(error)
        return result

    def _detach(self):
        if self.active:
            if self.previous_step_attribute is _MISSING:
                del self.vec_env.step
            else:
                self.vec_env.step = self.previous_step_attribute
            self.active = False

    def _complete(self):
        self._detach()
        self.metadata.update(
            status="encoding", steps=len(self.actions), frames=len(self.frames),
            episode_return=float(np.asarray(self.rewards).sum()),
            finished_unix=time.time(),
            transitions_path=str(self.output_path.with_suffix(".npz")),
        )
        np.savez_compressed(
            self.output_path.with_suffix(".npz"), actions=np.stack(self.actions),
            pointclouds=np.stack(self.frames),
            rewards=np.stack(self.rewards), terminated=np.stack(self.terminated),
            truncated=np.stack(self.truncated),
        )
        self._write_metadata()
        self.thread = threading.Thread(target=self._encode, name="ppo-episode-video-encoder", daemon=True)
        self.thread.start()

    def _encode(self):
        try:
            self.metadata["video_render"] = self.encoder(self.output_path, np.stack(self.frames), self.fps)
            self.metadata["status"] = "complete"
            self._write_metadata()
            print(f"PPO training episode video saved: {self.output_path}", flush=True)
        except Exception as error:
            self._fail(error)
        finally:
            self.frames.clear()

    def _fail(self, error):
        self._detach()
        self.metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            self._write_metadata()
        except Exception:
            pass
        self.frames.clear()
        print(f"PPO video recording disabled; training continues: {error}", flush=True)

    def close(self):
        if self.active:
            self._fail(RuntimeError("Training ended before a complete episode was recorded"))
        if self.thread is not None:
            self.thread.join(timeout=2)


def maybe_record_training_episode(rollout, output_path, *, metadata=None):
    """A failed optional recorder must never prevent the authorized training run."""
    if not output_path:
        return None
    try:
        return TrainingEpisodeRecorder(rollout, output_path, metadata=metadata)
    except Exception as error:
        print(f"PPO video recording could not start; training continues: {error}", flush=True)
        return None
