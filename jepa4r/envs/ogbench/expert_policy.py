"""``ExpertPolicy`` that also accepts ``swm/OGBCubePCD-v0``.

Upstream ``set_env`` whitelists the env id ('swm/OGBCube-v0' / 'swm/OGBScene-v0').
Everything else (oracle construction, per-episode noise, sub-task switching)
is inherited unchanged, so the collection policy is identical.
"""

import numpy as np
from stable_worldmodel.envs.ogbench import ExpertPolicy


class PCDExpertPolicy(ExpertPolicy):
    def set_env(self, env):
        self.env = env
        single_env = self.env.envs[0].unwrapped
        assert "Cube" in single_env.spec.id or "Scene" in single_env.spec.id, (
            "PCDExpertPolicy can only be used with OGBench manipulation environments."
        )
        self._set_oracle_agents()
        self._p_stack = np.zeros(self.env.num_envs)
        self._xi = np.zeros(self.env.num_envs)
        self._agents = [None] * self.env.num_envs
