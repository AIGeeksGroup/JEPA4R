"""Replay/episode boundary checks using array-only Gym envs, no simulator."""
import gymnasium as gym
import numpy as np
import pytest

from maniskill2_learn.env.wrappers import ExtendedEnv, build_wrapper
from ms2_lewm.history import PointCloudHistory, PointCloudHistoryBuffer
from ms2_lewm.official import lewm_config


class ReusingPointEnv(gym.Env):
    """Mutate the same arrays on every step to expose accidental aliasing."""

    def __init__(self, base=0, terminated_at=None, truncated_at=None):
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "xyz": gym.spaces.Box(-np.inf, np.inf, (8, 3), np.float32),
            "rgb": gym.spaces.Box(0, 1, (8, 3), np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (30,), np.float32),
        })
        self.observation = {key: np.zeros(space.shape, space.dtype) for key, space in self.observation_space.spaces.items()}
        self.base = base
        self.terminated_at = terminated_at
        self.truncated_at = truncated_at
        self.actions = []
        self.reset_calls = 0

    def _obs(self):
        self.observation["xyz"].fill(self.base + self.t)
        self.observation["rgb"].fill(self.t / 100)
        # Every dimension has a distinct value and each control step/episode
        # changes the whole vector, exposing stale history or reordering.
        self.observation["state"][:] = current_state(self.base, self.t)
        return self.observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        if options and "base" in options:
            self.base = options["base"]
        self.t = 0
        return self._obs(), {"seed": seed}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.t += 1
        return self._obs(), self.t / 10, self.t == self.terminated_at, self.t == self.truncated_at, {"step": self.t}


def values(obs):
    return obs["xyz"][:, 0, 0].tolist()


def current_state(base, t):
    return np.arange(30, dtype=np.float32) + np.float32(1000 * (base + t))


def test_official_registration_config_defaults_and_gym_contract():
    raw = ReusingPointEnv()
    env = build_wrapper(dict(type="PointCloudHistory", env=raw, env_name="PickCube-v0"))
    cfg = lewm_config()
    assert env.history_size == cfg.history_size == 3
    assert env.history_stride == 1
    # Vendored source remains unchanged; current training explicitly uses 1.
    assert cfg.data.dataset.frameskip == 5
    assert env.history.raw_history_size == 3
    assert raw.reset_calls == 0
    assert env.observation_space["xyz"].shape == (3, 8, 3)
    assert env.observation_space["rgb"].shape == (3, 8, 3)
    assert env.observation_space["state"].shape == (30,)
    obs, info = env.reset(seed=71)
    assert info == {"seed": 71}
    assert set(obs) == {"xyz", "rgb", "state"}
    assert env.observation_space.contains(obs)
    assert env.observation_space["state"].shape == (30,)
    np.testing.assert_array_equal(obs["state"], current_state(0, 0))
    assert values(obs) == [0, 0, 0]
    action = np.arange(7, dtype=np.float32) / 10
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert values(next_obs) == [0, 0, 1]
    np.testing.assert_array_equal(next_obs["state"], current_state(0, 1))
    assert next_obs["state"].dtype == raw.observation["state"].dtype
    assert (reward, terminated, truncated, info) == (0.1, False, False, {"step": 1})
    assert len(raw.actions) == 1
    np.testing.assert_array_equal(raw.actions[0], action)


def test_consecutive_history_at_every_control_step_without_action_repeat_or_future_leak():
    raw = ReusingPointEnv()
    env = PointCloudHistory(raw)
    first, _ = env.reset()
    for t in range(1, 19):
        obs, *_ = env.step(np.full(7, t, np.float32))
        expected = [max(0, t - 2), max(0, t - 1), t]
        assert values(obs) == expected
        np.testing.assert_allclose(obs["rgb"][:, 0, 0], np.array(expected) / 100)
        np.testing.assert_array_equal(obs["state"], raw.observation["state"])
        np.testing.assert_array_equal(obs["state"], current_state(0, t))
        assert obs["state"].shape == (30,)
        assert not np.shares_memory(obs["state"], raw.observation["state"])
        assert raw.t == len(raw.actions) == t
    assert values(first) == [0, 0, 0]
    np.testing.assert_array_equal(first["state"], current_state(0, 0))


@pytest.mark.parametrize("boundary", ["terminated_at", "truncated_at", "outer_time_limit"])
def test_terminal_next_observation_and_reset_keep_episode_boundaries(boundary):
    raw = ReusingPointEnv(**({boundary: 6} if boundary != "outer_time_limit" else {}))
    history_env = PointCloudHistory(raw)
    env = gym.wrappers.TimeLimit(history_env, 6) if boundary == "outer_time_limit" else history_env
    # Match the actual outermost official wrapper used by PPO.
    env = ExtendedEnv(env, reward_scale=1.0, use_cost=False)
    obs = env.reset(seed=13)
    for t in range(6):
        terminal_obs, _, terminated, truncated, _ = env.step(np.zeros(7, np.float32))
    assert bool(terminated) == (boundary == "terminated_at")
    assert bool(truncated) == (boundary != "terminated_at")
    assert values(terminal_obs) == [4, 5, 6]
    np.testing.assert_array_equal(terminal_obs["state"], current_state(0, 6))
    reset_obs = env.reset(options={"base": 100})
    assert values(reset_obs) == [100, 100, 100]
    np.testing.assert_array_equal(reset_obs["state"], current_state(100, 0))
    assert values(terminal_obs) == [4, 5, 6]
    np.testing.assert_array_equal(terminal_obs["state"], current_state(0, 6))
    assert values(obs) == [0, 0, 0]
    np.testing.assert_array_equal(obs["state"], current_state(0, 0))
    next_obs = env.step(np.zeros(7, np.float32))[0]
    assert values(next_obs) == [100, 100, 101]
    np.testing.assert_array_equal(next_obs["state"], current_state(100, 1))


def test_vector_workers_and_caller_mutations_do_not_share_history_arrays():
    a, b = PointCloudHistory(ReusingPointEnv()), PointCloudHistory(ReusingPointEnv(base=500))
    old_a, _ = a.reset()
    old_b, _ = b.reset()
    for _ in range(12):
        a.step(np.zeros(7))
    assert values(a.get_obs()) == [10, 11, 12]
    assert values(b.get_obs()) == [500, 500, 500]
    returned = a.get_obs()
    returned["xyz"].fill(-99)
    returned["state"].fill(-99)
    a.env.observation["state"].fill(-100)
    assert values(a.get_obs()) == [10, 11, 12]
    np.testing.assert_array_equal(a.get_obs()["state"], current_state(0, 12))
    np.testing.assert_array_equal(b.get_obs()["state"], current_state(500, 0))
    assert values(old_a) == [0, 0, 0]
    assert values(old_b) == [500, 500, 500]
    np.testing.assert_array_equal(old_a["state"], current_state(0, 0))
    np.testing.assert_array_equal(old_b["state"], current_state(500, 0))


def test_saved_replay_histories_are_independent_of_sampling_order_and_later_resets():
    env = PointCloudHistory(ReusingPointEnv())
    obs, _ = env.reset()
    transitions = []
    for _ in range(17):
        next_obs, *_ = env.step(np.zeros(7))
        # No test-side copy: each wrapper output must already own its arrays.
        transitions.append((obs, next_obs))
        obs = next_obs
    permutation = np.random.default_rng(31).permutation(len(transitions))
    batch = np.stack([transitions[i][0]["xyz"] for i in permutation])
    next_batch = np.stack([transitions[i][1]["xyz"] for i in permutation])
    state_batch = np.stack([transitions[i][0]["state"] for i in permutation])
    next_state_batch = np.stack([transitions[i][1]["state"] for i in permutation])
    env.reset(options={"base": 900})
    for j, t in enumerate(permutation):
        np.testing.assert_array_equal(batch[j, :, 0, 0], [max(0, t - 2), max(0, t - 1), t])
        np.testing.assert_array_equal(next_batch[j, :, 0, 0], [max(0, t - 1), t, t + 1])
        np.testing.assert_array_equal(transitions[t][0]["xyz"], batch[j])
        np.testing.assert_array_equal(state_batch[j], current_state(0, t))
        np.testing.assert_array_equal(next_state_batch[j], current_state(0, t + 1))
        np.testing.assert_array_equal(transitions[t][0]["state"], state_batch[j])


def test_current_nonpoint_fields_and_state_dtype_are_preserved_without_aliasing():
    observation = {
        "xyz": np.zeros((2, 3), dtype=np.float32),
        "rgb": np.zeros((2, 3), dtype=np.float32),
        "state": np.arange(30, dtype=np.float64) / 7 - 2,
        "auxiliary": np.array([7, 8], dtype=np.int16),
    }
    buffer = PointCloudHistoryBuffer()
    original = buffer.reset(observation)
    expected_initial_state = observation["state"].copy()
    observation["state"] += 100
    observation["auxiliary"][:] = [9, 10]
    current = buffer.append(observation)
    np.testing.assert_array_equal(original["state"], expected_initial_state)
    np.testing.assert_array_equal(original["auxiliary"], [7, 8])
    np.testing.assert_array_equal(current["state"], observation["state"])
    np.testing.assert_array_equal(current["auxiliary"], [9, 10])
    assert current["state"].dtype == np.float64
    assert current["auxiliary"].shape == (2,)
    assert current["auxiliary"].dtype == np.int16
    current["auxiliary"][:] = -1
    observation["auxiliary"][:] = -2
    np.testing.assert_array_equal(buffer.snapshot()["auxiliary"], [9, 10])


def test_single_frame_and_invalid_history_inputs_fail_explicitly():
    env = PointCloudHistory(ReusingPointEnv(), history_size=1, history_stride=1)
    with pytest.raises(RuntimeError, match="Reset"):
        env.get_obs()
    assert values(env.reset()[0]) == [0]
    assert values(env.step(np.zeros(7))[0]) == [1]
    for kwargs in ({"history_size": 0}, {"history_stride": -2}, {"history_size": 1.5}, {"history_stride": True}):
        with pytest.raises(ValueError, match="positive integer"):
            PointCloudHistoryBuffer(**kwargs)
    with pytest.raises(ValueError, match="consecutive control-step frames"):
        PointCloudHistoryBuffer(history_stride=5)
    with pytest.raises(TypeError, match="frameskip"):
        PointCloudHistory(ReusingPointEnv(), frameskip=5)
    buffer = PointCloudHistoryBuffer()
    with pytest.raises(ValueError, match="both xyz and rgb"):
        buffer.reset({"xyz": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="same point count"):
        buffer.reset({"xyz": np.zeros((2, 3)), "rgb": np.zeros((3, 3))})
    points = {"xyz": np.zeros((2, 3)), "rgb": np.zeros((2, 3))}
    with pytest.raises(ValueError, match="original current state"):
        buffer.reset(points)
    for state in (np.array(0), np.zeros(0), np.zeros((3, 30))):
        with pytest.raises(ValueError, match="current state shape"):
            buffer.reset({**points, "state": state})
    buffer.reset({**points, "state": np.zeros(30)})
    with pytest.raises(ValueError, match="original current state"):
        buffer.append(points)
    for state in (np.zeros(29), np.zeros(30, dtype=np.float32)):
        with pytest.raises(ValueError, match="state shape and dtype must stay constant"):
            buffer.append({**points, "state": state})
    with pytest.raises(ValueError, match="stay constant"):
        buffer.append({"xyz": np.zeros((3, 3)), "rgb": np.zeros((3, 3)), "state": np.zeros(30)})
