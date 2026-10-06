"""Official compact-boundary goals over immutable Cube-derived RGB shards."""
from collections import OrderedDict
import copy
import json
from pathlib import Path

import numpy as np
import torch


class OfficialGoalSampler:
    """Locked official actor_p_cur=0/traj=1/random=0 uniform rounding stream."""
    def __init__(self, episodes, transitions_per_episode=1000, seed=0):
        self.episodes, self.transitions = int(episodes), int(transitions_per_episode)
        self.size = self.episodes * self.transitions
        self.rng = np.random.RandomState(seed)

    def sample(self, compact_indices):
        idx = np.asarray(compact_indices, dtype=np.int64)
        if np.any((idx < 0) | (idx >= self.size)): raise ValueError('Current transition outside split')
        # Official sample_goals consumes unused random goals and branch draws.
        self.rng.randint(self.size, size=len(idx))
        episode, step = idx // self.transitions, idx % self.transitions
        raw = episode * (self.transitions + 1) + step
        final = episode * (self.transitions + 1) + self.transitions - 1
        distance = self.rng.rand(len(idx))
        goal_raw = np.round(np.minimum(raw + 1, final) * distance + final * (1 - distance)).astype(np.int64)
        self.rng.rand(len(idx)); self.rng.rand(len(idx))
        # Rounding must happen in original raw-index space, before compact map.
        return episode * self.transitions + goal_raw - episode * (self.transitions + 1)


class LogicalSampler:
    """Parent-owned exact RNG/plans, including goal plans prefetched by workers."""
    def __init__(self, size, micro_batch=64, accumulation=16, seed=0, transitions_per_episode=1000):
        if size % transitions_per_episode: raise ValueError('Whole compact episodes required')
        self.size, self.micro, self.accumulation = int(size), int(micro_batch), int(accumulation)
        self.logical = self.micro * self.accumulation
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).numpy()
        self.goal_sampler = OfficialGoalSampler(size // transitions_per_episode, transitions_per_episode, seed)
        self.committed = 0
        self.plans = {}

    def __iter__(self):
        for start in range(self.committed, self.size, self.logical):
            if start not in self.plans:
                current = self.order[start:min(start + self.logical, self.size)]
                self.plans[start] = list(zip(current.tolist(), self.goal_sampler.sample(current).tolist()))
            plan = self.plans[start]
            for micro in range(0, len(plan), self.micro): yield plan[micro:micro + self.micro]

    def __len__(self):
        return (self.size - self.committed + self.micro - 1) // self.micro

    def commit_logical_batch(self, samples):
        expected = min(self.logical, self.size - self.committed)
        if samples != expected or self.committed not in self.plans: raise ValueError('Logical optimizer boundary/count differs')
        self.plans.pop(self.committed)
        self.committed += samples

    def next_epoch(self):
        if self.committed != self.size: raise ValueError('Cannot advance incomplete epoch')
        self.order = torch.randperm(self.size, generator=self.generator).numpy()
        self.committed, self.plans = 0, {}
        # Goal RNG continues; no epoch seed reset or per-index substitute RNG.

    def state_dict(self):
        return copy.deepcopy(dict(size=self.size, micro=self.micro, accumulation=self.accumulation,
             order=self.order, committed=self.committed, plans=self.plans, generator=self.generator.get_state(),
             transitions=self.goal_sampler.transitions, goal_rng=self.goal_sampler.rng.get_state()))

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['size'], state['micro'], state['accumulation'], transitions_per_episode=state['transitions'])
        obj.order = state['order'].copy(); obj.committed = state['committed']; obj.plans = copy.deepcopy(state['plans'])
        obj.generator.set_state(state['generator']); obj.goal_sampler.rng.set_state(state['goal_rng'])
        return obj


class IdentityAction:
    def __call__(self, action): return action


class CubeOfficialDataset(torch.utils.data.Dataset):
    """Single real frame/goal, unchanged raw normalized Box actions, no time axis."""
    def __init__(self, index_path, split, preprocessor, open_files=16, expected_rows=1001):
        if split not in ('train', 'val'): raise ValueError('Train/val must be explicit independent split')
        index = json.loads(Path(index_path).read_text())
        if index['status'] != 'passed' or index['exit_code'] != 0: raise ValueError('Full data index has not passed')
        self.entries = index[split]
        if [entry['episode'] for entry in self.entries] != list(range(len(self.entries))): raise ValueError('Source episode indices differ')
        self.split, self.preprocessor = split, preprocessor
        self.rows, self.transitions, self.open_files = expected_rows, expected_rows - 1, open_files
        self.readers = OrderedDict()

    def __len__(self): return len(self.entries) * self.transitions

    def __getstate__(self):
        state = self.__dict__.copy(); state['readers'] = OrderedDict(); return state

    def _reader(self, episode):
        from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
        from .transforms import DataTransform
        if episode in self.readers: self.readers.move_to_end(episode); return self.readers[episode]
        reader = HDF5Dataset(path=self.entries[episode]['path'], num_steps=1, frameskip=1,
            keys_to_load=['pixels', 'action', 'source_row', 'source_episode', 'source_step', 'source_terminal'],
            transform=DataTransform('pixels', self.preprocessor, IdentityAction()))
        if reader.lengths.tolist() != [self.rows] or reader.offsets.tolist() != [0]: raise ValueError('Original local episode bounds differ')
        self.readers[episode] = reader
        if len(self.readers) > self.open_files:
            _, retired = self.readers.popitem(last=False)
            if retired.h5_file is not None: retired.h5_file.close()
        return reader

    def __getitem__(self, pair):
        current, goal = map(int, pair)
        if not 0 <= current < len(self) or not 0 <= goal < len(self): raise ValueError('Transition index outside split')
        episode, step = divmod(current, self.transitions)
        goal_episode, goal_step = divmod(goal, self.transitions)
        if goal_episode != episode or goal_step < min(step + 1, self.transitions - 1): raise ValueError('Official trajectory goal crosses episode or is past')
        reader = self._reader(episode)
        cur = reader._load_slice(0, step, step + 1); target = reader._load_slice(0, goal_step, goal_step + 1)
        if cur['source_terminal'].any() or target['source_terminal'].any(): raise ValueError('Raw terminal observation/action may not enter compact sample')
        result = dict(current=cur['pixels'][0], goal=target['pixels'][0], action=cur['action'][0],
                      source_row=cur['source_row'][0], goal_source_row=target['source_row'][0],
                      source_episode=cur['source_episode'][0], goal_source_episode=target['source_episode'][0])
        if result['action'].shape != (5,) or result['action'].dtype != torch.float32 or not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError('Real single-frame/raw-action contract invalid')
        return result
