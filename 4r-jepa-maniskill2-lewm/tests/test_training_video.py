"""Recorder observes real training transitions without rendering or changing them."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ms2_lewm.training_video import TrainingEpisodeRecorder


class FakeVec:
    num_envs = 2

    def __init__(self, terminate=False):
        self.counts = np.zeros(2, dtype=np.int64)
        self.recent_obs = self.obs()
        self.step_calls = 0
        self.terminate = terminate
        self.received_actions = []
        self.received_indices = []

    def obs(self, idx=None):
        counts = self.counts if idx is None else self.counts[idx]
        xyz = np.repeat(counts[:, None, None], 12, axis=1).repeat(3, axis=2).astype(np.float32)
        return {"xyz": xyz, "rgb": np.full_like(xyz, 0.25)}

    def render(self, *args, **kwargs):
        pytest.fail("Recording pointcloud input must not render the environment")

    def get_state(self, idx=None):
        return self.counts[idx].copy()

    def step(self, actions, idx=None):
        self.step_calls += 1
        self.received_actions.append(actions)
        self.received_indices.append(idx)
        selected = np.arange(2) if idx is None else idx
        self.counts[selected] += 1
        next_obs = self.obs(selected)
        self.recent_obs = self.obs()
        reward = np.random.uniform(size=(len(selected), 1)).astype(np.float32)
        done = (self.counts[selected] >= 3)[:, None]
        terminated = done if self.terminate else np.zeros_like(done)
        truncated = np.zeros_like(done) if self.terminate else done
        return next_obs, reward, terminated, truncated, {"success": terminated.copy()}

    def step_dict(self, actions, idx=None):
        result = self.step(actions, idx=idx)
        selected = np.arange(2) if idx is None else idx
        done = (result[2] | result[3]).reshape(-1)
        self.counts[selected[done]] = 0
        self.recent_obs = self.obs()
        return result


def rollout(vec):
    return SimpleNamespace(vec_env=vec, num_envs=vec.num_envs)


@pytest.mark.parametrize("terminate", [False, True])
def test_real_transitions_rng_and_terminal_points_unchanged(tmp_path, terminate):
    actions = [np.full((2, 7), step / 10, dtype=np.float32) for step in range(5)]
    np.random.seed(120)
    baseline = FakeVec(terminate)
    expected = [baseline.step_dict(action) for action in actions]
    baseline_rng_next = np.random.random()

    np.random.seed(120)
    vec = FakeVec(terminate)
    encoded = {}

    def encoder(path, pointclouds, fps):
        encoded.update(points=pointclouds.copy(), fps=fps)
        path.write_bytes(b"FAKE VIDEO; UNIT TEST ONLY")
        return {"frames": len(pointclouds), "test_double": True}

    path = tmp_path / "episode.mp4"
    recorder = TrainingEpisodeRecorder(rollout(vec), path, encoder=encoder)
    actual = [vec.step_dict(action) for action in actions]
    recorder.close()
    assert not recorder.active and "step" not in vars(vec)
    assert vec.step_calls == len(actions)
    assert all(received is action for received, action in zip(vec.received_actions, actions))
    assert all(index is None for index in vec.received_indices)
    for got, want in zip(actual, expected):
        for key in got[0]:
            np.testing.assert_array_equal(got[0][key], want[0][key])
        for i in range(1, 4):
            np.testing.assert_array_equal(got[i], want[i])
    assert np.random.random() == baseline_rng_next
    # Initial frame plus three transitions, including terminal before auto-reset.
    np.testing.assert_array_equal(encoded["points"][:, 0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(encoded["points"][..., 3:], 0.25)
    assert encoded["points"].dtype == np.float32
    assert encoded["fps"] == 20
    saved = np.load(path.with_suffix(".npz"))
    np.testing.assert_array_equal(saved["pointclouds"], encoded["points"])
    np.testing.assert_array_equal(saved["actions"], np.stack(actions[:3])[:, 0])
    report = json.loads(path.with_suffix(".json").read_text())
    assert report["steps"] == 3 and report["frames"] == 4
    assert report["status"] == "complete" and not report["separate_evaluation"]


def test_nonzero_worker_excluded_and_partial_index_position_correct(tmp_path):
    vec = FakeVec()
    encoded = {}

    def encoder(path, points, fps):
        encoded["points"] = points.copy()
        return {}

    recorder = TrainingEpisodeRecorder(rollout(vec), tmp_path / "partial.mp4", encoder=encoder)
    vec.step_dict(np.zeros((1, 7)), idx=np.array([1]))
    assert not recorder.frames
    indices = np.array([1, 0])
    actions = np.array([[1] * 7, [2] * 7])
    for _ in range(3):
        vec.step_dict(actions, idx=indices)
    recorder.close()
    np.testing.assert_array_equal(encoded["points"][:, 0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(recorder.actions, np.full((3, 7), 2))
    assert vec.received_indices[-1] is indices


def test_recorder_capture_failure_does_not_duplicate_or_prevent_real_step(tmp_path):
    vec = FakeVec()
    del vec.recent_obs["xyz"]
    recorder = TrainingEpisodeRecorder(rollout(vec), tmp_path / "broken.mp4")
    result = vec.step_dict(np.zeros((2, 7)))
    assert vec.step_calls == 1
    assert not recorder.active and "step" not in vars(vec)
    assert result[0]["xyz"][0, 0, 0] == 1
    report = json.loads((tmp_path / "broken.json").read_text())
    assert report["status"] == "failed" and "KeyError" in report["error"]


def test_encoder_failure_only_fails_recording(tmp_path):
    vec = FakeVec()

    def broken_encoder(*args):
        raise RuntimeError("test encoder failure")

    recorder = TrainingEpisodeRecorder(rollout(vec), tmp_path / "encode.mp4", encoder=broken_encoder)
    for _ in range(4):
        vec.step_dict(np.zeros((2, 7)))
    recorder.close()
    assert vec.step_calls == 4 and not recorder.active
    report = json.loads((tmp_path / "encode.json").read_text())
    assert report["status"] == "failed" and "test encoder failure" in report["error"]


def test_existing_instance_step_hook_restored(tmp_path):
    vec = FakeVec()
    old_step = vec.step
    vec.step = old_step
    recorder = TrainingEpisodeRecorder(rollout(vec), tmp_path / "restore.mp4", encoder=lambda *args: {})
    for _ in range(3):
        vec.step_dict(np.zeros((2, 7)))
    recorder.close()
    assert vars(vec)["step"] is old_step


def test_original_environment_exceptions_propagate(tmp_path):
    vec = FakeVec()

    def broken_step(actions, idx=None):
        raise RuntimeError("real environment failure")

    vec.step = broken_step
    recorder = TrainingEpisodeRecorder(rollout(vec), tmp_path / "env.mp4")
    with pytest.raises(RuntimeError, match="real environment failure"):
        vec.step_dict(np.zeros((2, 7)))
    recorder.close()
