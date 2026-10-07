"""CPU task contracts; these tests do not establish native replay or CEM scores."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def module():
    spec = importlib.util.spec_from_file_location('relocate_fragment_contract', ROOT / 'envs/adroit/fragment.py')
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    assert hasattr(out, 'RelocateFragment'), 'thin Relocate fragment adapter is missing'
    return out


class Native:
    def __init__(self):
        self.initial = dict(qpos=np.zeros(36), qvel=np.zeros(36), hand_qpos=np.zeros(30),
                            obj_pos=np.array([.01, .02, .03]), palm_pos=np.zeros(3),
                            target_pos=np.array([.1, .2, .3]))
        self.state = {key: value.copy() for key, value in self.initial.items()}
        self.actions, self.done = [], False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.state = {key: value.copy() for key, value in self.initial.items()}
        self.actions = []
        return self.get_obs()

    def get_env_state(self):
        return {key: value.copy() for key, value in self.state.items()}

    def set_env_state(self, state):
        raise AssertionError('Relocate fragment must restore by seeded reset and exact prefix')

    def get_obs(self):
        obs = np.zeros(39)
        obs[:30] = self.state['qpos'][:30]
        obs[36:39] = self.state['qpos'][30:33] - self.state['target_pos']
        return obs

    def step(self, action):
        self.actions.append(action.copy())
        self.state['qpos'][:30] += action
        self.state['qvel'][:30] = action
        self.state['hand_qpos'] = self.state['qpos'][:30].copy()
        return self.get_obs(), -2., self.done, dict(goal_achieved=True)


def row(native, actions):
    initial, initial_obs = native.get_env_state(), native.get_obs()
    for action in actions:
        native.step(action)
    out = dict(native.get_env_state(), native_observation=native.get_obs(), step_idx=len(actions),
               native_step_idx=len(actions), native_replay_trace=dict(initial_state=initial,
                   initial_observation=initial_obs, actions=actions, valid=np.ones(len(actions), np.int64)))
    native.reset()
    return out


def test_seeded_reset_exact_prefix_future_position_and_no_midstate_setter():
    m, native = module(), Native()
    actions = np.stack([np.linspace(-.2, .2, 30), np.linspace(.1, -.1, 30)])
    start = row(native, actions)
    future = native.get_obs()
    future[36:39] = [.04, -.02, -.25]
    env = m.RelocateFragment(native)
    obs, info = env.reset(seed=61, options=env.reset_options_from_dataset(start, dict(goal_native_observation=future)))
    np.testing.assert_array_equal(native.actions, actions)
    np.testing.assert_array_equal(obs, start['native_observation'])
    np.testing.assert_allclose(info['fragment_goal_position'], [.14, .18, .05])
    assert env.proprio.shape == (30,) and info['restore_checked'] == 1 and native.seed_value == 61
    assert env._t == 0


def test_relocate_distance_strict_boundary_and_low_height_is_not_pen_drop():
    m, native = module(), Native()
    env = m.RelocateFragment(native)
    env.set_goal_state([.1, 0., 0.])
    assert env.step(np.zeros(30))[4]['fragment_success'] == 0
    env.set_goal_state([.099, 0., 0.])
    _, _, terminated, truncated, info = env.step(np.zeros(30))
    assert terminated and not truncated and info['fragment_success'] == 1
    assert 'fragment_axis_dot' not in info and 'fragment_goal_axis' not in info
    native.done = True
    assert env.step(np.zeros(30))[2:4] == (False, True)


def test_relocate_clips_actual_controls_and_native_goal_does_not_override_fragment():
    m, native = module(), Native()
    env = m.RelocateFragment(native)
    env.set_goal_state([1., 0., 0.])
    action = np.linspace(-2., 2., 30)
    _, _, terminated, _, info = env.step(action)
    assert not terminated and info['native_goal_achieved'] == 1
    np.testing.assert_array_equal(info['executed_action'], np.clip(action, -1., 1.))
    with pytest.raises(ValueError, match='finite'):
        env.step(np.zeros(24))


def test_initial_state_and_endpoint_mismatch_are_hold():
    m, native = module(), Native()
    start = row(native, np.zeros((2, 30)))
    env = m.RelocateFragment(native)
    options = env.reset_options_from_dataset(start, dict(native_observation=native.get_obs()))
    options['initial_state']['target_pos'][0] += .01
    with pytest.raises(RuntimeError, match='HOLD.*initial.*state'):
        env.reset(seed=61, options=options)
    start['qvel'][0] += .1
    options = env.reset_options_from_dataset(start, dict(native_observation=native.get_obs()))
    with pytest.raises(RuntimeError, match='HOLD.*state restore'):
        env.reset(seed=61, options=options)


class Reader:
    lengths, offsets = np.array([3, 4]), np.array([0, 3])
    episode_column_names = []

    def __init__(self):
        self.data = dict(step_idx=np.array([0, 1, 2, 0, 1, 2, 3]),
                         seed=np.array([61, 61, 61, 62, 62, 62, 62]),
                         ep_idx=np.array([0, 0, 0, 1, 1, 1, 1]), action=np.arange(210).reshape(7, 30) / 1000.,
                         qpos=np.zeros((7, 36)), qvel=np.zeros((7, 36)), observation=np.zeros((7, 39)))
        self.column_names, self.h5_file = list(self.data), self.data

    def _open(self):
        pass

    def get_col_data(self, key):
        return self.data[key]

    def get_row_data(self, rows):
        return {key: value[rows] for key, value in self.data.items()}

    def load_chunk(self, episodes, starts, ends):
        return [{key: value[self.offsets[e]+s:self.offsets[e]+end] for key, value in self.data.items()}
                for e, s, end in zip(episodes, starts, ends)]


def initials(reader):
    out = dict(seed=np.array([61, 62]), original_episode_index=np.arange(2),
               reset_observation=reader.data['observation'][reader.offsets].copy())
    for key, size in [('qpos', 36), ('qvel', 36), ('hand_qpos', 30), ('obj_pos', 3), ('palm_pos', 3), ('target_pos', 3)]:
        out[key] = np.zeros((2, size))
    return out


def test_readonly_pre_action_trace_aliases_and_initial_npz(tmp_path):
    m, reader = module(), Reader()
    path = tmp_path / 'initials.npz'
    np.savez(path, **initials(reader))
    view = m.EpisodeAliases(reader, task='relocate-v0', initial_states=path)
    traces = view.get_episode_data([1, 0, 1])['native_replay_trace']
    np.testing.assert_array_equal(traces[0]['actions'], reader.data['action'][3:7])
    np.testing.assert_array_equal(traces[1]['actions'], reader.data['action'][:3])
    rows = view.get_row_data([1, 4])
    np.testing.assert_array_equal(rows['native_step_idx'], [1, 1])
    np.testing.assert_array_equal(rows['native_observation'], reader.data['observation'][[1, 4]])
    np.testing.assert_array_equal(rows['action'], reader.data['action'][[1, 4]])
    assert 'native_replay_trace' not in view.column_names and 'native_observation' in view.column_names
    assert set(reader.data) == set(reader.column_names)


class PixelTensor:
    """Layout-only CPU double, as in existing Pen tests; native Torch remains a smoke gate."""
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def new_tensor(self, value):
        return PixelTensor(value)

    def __getitem__(self, key):
        return PixelTensor(self.value[key])


def test_sparse_rgb_uses_real_requested_rows_and_missing_frame_holds(tmp_path, monkeypatch):
    if importlib.util.find_spec('torch') is None:
        monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(from_numpy=PixelTensor))
    m, reader = module(), Reader()
    path = tmp_path / 'rgb.h5'
    pixels = np.zeros((3, 256, 256, 3), np.uint8)
    pixels[:, :, :, 0] = np.array([14, 13, 15])[:, None, None]
    with h5py.File(path, 'w') as target:
        target['row_index'] = [4, 3, 5]
        target['pixels'] = pixels
    view = m.EpisodeAliases(reader, task='relocate-v0', initial_states=initials(reader), rgb_frames=path)
    chunk = view.load_chunk([1], [0], [3])[0]
    assert chunk['pixels'].shape == (3, 3, 224, 224)
    np.testing.assert_array_equal(chunk['pixels'][:, 0, 0, 0].numpy(), [13, 14, 15])
    assert view.get_row_data(4)['pixels'][0, 0, 0] == 14
    with pytest.raises(RuntimeError, match='HOLD.*RGB.*row'):
        view.get_row_data(6)


def test_initial_dataset_identity_and_action_bounds_are_checked():
    m, reader = module(), Reader()
    packed = initials(reader)
    packed['seed'][0] += 1
    with pytest.raises(ValueError, match='initial.*seed'):
        m.EpisodeAliases(reader, task='relocate-v0', initial_states=packed)
    packed['seed'][0] -= 1
    reader.data['action'][0, 0] = 2.
    view = m.EpisodeAliases(reader, task='relocate-v0', initial_states=packed)
    with pytest.raises(RuntimeError, match='HOLD.*controls'):
        view.get_episode_data([0])
