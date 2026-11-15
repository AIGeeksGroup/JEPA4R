"""Cube-count/input projection only; no sampling, reward or learning math here."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CubeInputs:
    cubes: int
    latent_dim: int = 192

    def __post_init__(self):
        if self.cubes not in (1, 2, 3, 4) or self.latent_dim < 1:
            raise ValueError("Supported Cube counts are 1,2,3,4; latent_dim must be positive")

    @property
    def state_dim(self):
        return 19 + 9 * self.cubes

    @property
    def env_id(self):
        return "cube-" + ("single", "double", "triple", "quadruple")[self.cubes - 1] + "-v0"

    def check(self, features, states):
        features, states = np.asarray(features), np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("Wrong official Cube state dimension")
        if features.shape != (len(states), self.latent_dim) or features.dtype != np.float32:
            raise ValueError("Features must be aligned FP32 rows with the configured latent dimension")
        return features, states

    def current(self, features, states, oracle=False):
        features, states = self.check(features, states)
        prefix = np.zeros_like(features) if oracle else features
        return np.concatenate((prefix, states if oracle else states[:, :19]), axis=-1).astype(np.float32)

    def goal(self, features, states, oracle=False):
        features, states = self.check(features, states)
        xyz = np.concatenate([states[:, 19 + 9 * i : 22 + 9 * i] for i in range(self.cubes)], axis=-1)
        return np.concatenate((np.zeros_like(features) if oracle else features, xyz), axis=-1).astype(np.float32)


def selftest():
    for count in (1, 2, 3, 4):
        contract = CubeInputs(count)
        state = np.arange(2 * contract.state_dim, dtype=np.float32).reshape(2, -1)
        latent = np.ones((2, 192), np.float32)
        assert contract.current(latent, state).shape == (2, 211)
        assert contract.goal(latent, state).shape == (2, 192 + 3 * count)
        assert contract.current(latent, state, oracle=True).shape == (2, 192 + contract.state_dim)
        assert not contract.goal(latent, state, oracle=True)[:, :192].any()
        np.testing.assert_array_equal(contract.current(latent, state)[:, 192:], state[:, :19])
        expected = np.concatenate([state[:, 19 + 9 * i : 22 + 9 * i] for i in range(count)], axis=-1)
        np.testing.assert_array_equal(contract.goal(latent, state)[:, 192:], expected)
    return {"counts": [1, 2, 3, 4], "current_visual_dim": 211, "triple_goal_dim": 201}


if __name__ == "__main__":
    print(selftest())
