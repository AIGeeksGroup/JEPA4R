import numpy as np
import pytest

from bench.cortexbench.adroit.pcd_runtime import get_adroit_pcd, seed_adroit_pcd


class FakeEnv:
    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)
        return [seed]


class FakeWrapper:
    def __init__(self, env_id="pen-v0"):
        self.env = FakeEnv()
        self.spec = type("Spec", (), {"id": env_id})()


def test_seed_and_get_adroit_pcd_keep_a_deterministic_point_cloud_contract():
    wrapper = FakeWrapper()

    def capture(env, rng, config):
        assert env is wrapper.env
        assert config.num_points == 1200
        value = rng.integers(0, 255)
        return None, np.full((1200, 3), value, dtype=np.float32), np.full((1200, 3), 7, dtype=np.uint8), {}

    seed_adroit_pcd(wrapper, 17)
    first = get_adroit_pcd(wrapper, capture_fn=capture)
    seed_adroit_pcd(wrapper, 17)
    second = get_adroit_pcd(wrapper, capture_fn=capture)

    assert wrapper.env.seeds == [17, 17]
    assert first.shape == (1200, 6)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_get_adroit_pcd_rejects_non_pen_tasks():
    with pytest.raises(RuntimeError, match="pen-v0"):
        get_adroit_pcd(FakeWrapper("relocate-v0"), capture_fn=lambda *_: None)
