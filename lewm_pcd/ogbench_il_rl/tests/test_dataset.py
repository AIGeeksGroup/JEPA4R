"""Synthetic data boundaries plus an actual pinned official sampler reference."""
import numpy as np
import pytest
import torch

from ogbench_il_rl.data import dataset as module


def test_new_dataset_interface_exists():
    assert module is not None, 'New official single-frame data interface missing'


def test_goal_rounding_and_terminal_self_goal():
    sampler = module.OfficialGoalSampler(episodes=2, transitions_per_episode=4, seed=0)
    current = np.array([0, 1, 2, 3, 4, 6, 7])
    goal = sampler.sample(current)
    assert np.all(goal // 4 == current // 4)
    assert np.all(goal >= np.minimum(current + 1, (current // 4 + 1) * 4 - 1))
    assert goal[3] == 3 and goal[-1] == 7


def test_prefetched_logical_plan_resume_is_exact():
    sampler = module.LogicalSampler(size=29, micro_batch=2, accumulation=4, seed=0, transitions_per_episode=29)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(4)]
    assert sum(map(len, consumed)) == 8
    ahead = [next(iterator) for _ in range(2)]  # model real worker prefetch.
    sampler.commit_logical_batch(8)
    state = sampler.state_dict()
    restored = module.LogicalSampler.from_state_dict(state)
    resumed = iter(restored)
    assert [next(resumed) for _ in range(2)] == ahead
    remaining = [pair for batch in resumed for pair in batch]
    assert len(remaining) == 17
    all_current = [cur for batch in consumed + ahead for cur, _ in batch] + [cur for cur, _ in remaining]
    assert sorted(all_current) == list(range(29))


def test_sampler_matches_actual_official_source():
    import os, sys, importlib.util
    root = os.environ.get('OFFICIAL_OGBENCH_SOURCE')
    assert root, 'CPU parity gate must specify actual pinned official source'
    path = __import__('pathlib').Path(root) / 'impls/utils/datasets.py'
    spec = importlib.util.spec_from_file_location('actual_official_dataset_reference', path)
    reference = importlib.util.module_from_spec(spec); sys.modules[spec.name] = reference; spec.loader.exec_module(reference)
    # Actual compact representation has two terminals at each raw episode end,
    # and valid_idxs exclude only the last raw terminal observation.
    raw_terms = np.zeros(15, np.float32); raw_terms[[4, 9, 14]] = 1
    terms = np.minimum(raw_terms + np.r_[raw_terms[1:], 1.], 1).astype(np.float32)
    data = reference.Dataset.create(observations=np.arange(15), actions=np.zeros((15, 5), np.float32), terminals=terms, valids=1 - raw_terms)
    config = dict(discount=.99, frame_stack=None, value_p_curgoal=0., value_p_trajgoal=1., value_p_randomgoal=0.,
                  actor_p_curgoal=0., actor_p_trajgoal=1., actor_p_randomgoal=0.)
    actual = reference.GCDataset(data, config)
    compact = np.array([0, 1, 2, 3, 4, 6, 7, 8, 11])
    raw = (compact // 4) * 5 + compact % 4
    np.random.seed(73)
    official = actual.sample_goals(raw, 0., 1., 0., False)
    candidate = module.OfficialGoalSampler(3, 4, seed=73).sample(compact)
    mapped = (candidate // 4) * 5 + candidate % 4
    np.testing.assert_array_equal(mapped, official)
