import copy

import numpy as np
import pytest

from bench.cortexbench.adroit.convert_demos_3d import convert_pen_demo


class FakeEnv:
    def __init__(self, observations):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.index = 0

    def set_env_state(self, state):
        del state
        self.index = 0

    def get_obs(self):
        return self.observations[self.index]

    def step(self, action):
        del action
        self.index += 1
        return self.observations[self.index], 0.0, False, {}


def _demo():
    observations = np.arange(12, dtype=np.float32).reshape(3, 4)
    return {
        "actions": np.ones((3, 2), dtype=np.float32),
        "observations": np.zeros((3, 7), dtype=np.float64),
        "state_obs": observations.astype(np.float64),
        "images": np.zeros((3, 256, 256, 3), dtype=np.uint8),
        "env_infos": {"proprio": observations.copy()},
        "init_state_dict": {"qpos": np.zeros(2), "qvel": np.zeros(2)},
    }


def test_convert_pen_demo_replaces_images_with_aligned_xyzrgb_and_keeps_contract():
    demo = _demo()
    env = FakeEnv(demo["state_obs"])

    def capture(_env, rng, config):
        del rng, config
        xyz = np.full((1200, 3), env.index, dtype=np.float32)
        rgb = np.full((1200, 3), 17 + env.index, dtype=np.uint8)
        return None, xyz, rgb, {"finite": True}

    converted, report = convert_pen_demo(
        copy.deepcopy(demo),
        env,
        capture_fn=capture,
        rng=np.random.default_rng(4),
        replay_atol=0.0,
    )

    assert converted["images"].shape == (3, 1200, 6)
    assert converted["images"].dtype == np.float32
    np.testing.assert_array_equal(converted["images"][2, :, :3], 2.0)
    np.testing.assert_array_equal(converted["images"][1, :, 3:], 18.0)
    np.testing.assert_array_equal(converted["actions"], demo["actions"])
    np.testing.assert_array_equal(converted["env_infos"]["proprio"], demo["env_infos"]["proprio"])
    assert report["max_state_replay_error"] == 0.0
    assert report["frames"] == 3


def test_convert_pen_demo_rejects_misaligned_replay_state():
    demo = _demo()
    demo["state_obs"][1, 0] += 1.0
    env = FakeEnv(_demo()["state_obs"])

    with pytest.raises(ValueError, match="state replay mismatch"):
        convert_pen_demo(
            demo,
            env,
            capture_fn=lambda *_: (
                None,
                np.zeros((1200, 3), dtype=np.float32),
                np.zeros((1200, 3), dtype=np.uint8),
                {},
            ),
            rng=np.random.default_rng(4),
            replay_atol=0.0,
        )
