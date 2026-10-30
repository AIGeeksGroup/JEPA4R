"""Thin input adapter around the unmodified official GCDataset sampler."""
from dataclasses import dataclass
import numpy as np
from utils.datasets import GCDataset
from cube_inputs import CubeInputs


@dataclass
class CachedInputs(GCDataset):
    features: object = None
    cubes: int = 3
    oracle: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.config['frame_stack'] is not None or self.config['p_aug'] != 0:
            raise ValueError('This frozen-feature interface uses the official state profile without augmentation/stacking')
        self.input_contract = CubeInputs(self.cubes, self.features.shape[1])
        if len(self.features) != self.size:
            raise ValueError('Cache rows must match the same compact dataset raw IDs')
        self.input_contract.check(self.features[:1], self.dataset['observations'][:1])

    def sample_goals(self, *args, **kwargs):
        ids = super().sample_goals(*args, **kwargs)
        self._goal_ids.append(ids.copy())
        return ids

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        self._goal_ids = []
        batch = super().sample(batch_size, idxs=idxs, evaluation=evaluation)
        value_ids, actor_ids = self._goal_ids
        state = self.dataset['observations']
        contract = self.input_contract
        batch['observations'] = contract.current(self.features[idxs], state[idxs], self.oracle)
        batch['next_observations'] = contract.current(self.features[idxs+1], state[idxs+1], self.oracle)
        batch['value_goals'] = contract.goal(self.features[value_ids], state[value_ids], self.oracle)
        batch['actor_goals'] = contract.goal(self.features[actor_ids], state[actor_ids], self.oracle)
        self.last_ids = dict(current=idxs.copy(), next=idxs+1, value_goal=value_ids, actor_goal=actor_ids)
        return batch
