import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def module(relative):
    path = ROOT / relative
    assert path.is_file(), 'new Pen fragment implementation is missing'
    spec = importlib.util.spec_from_file_location('pen_fragment_test_' + path.stem, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def observation(pos=(0., 0., .2), axis=(0., 0., 1.)):
    obs = np.zeros(45)
    obs[24:27], obs[33:36] = pos, axis
    return obs


class Native:
    def __init__(self):
        self.state = dict(qpos=np.zeros(30), qvel=np.zeros(30), desired_orien=np.array([1., 0., 0., 0.]))
        self.obs = observation()
        self.done = False
        self.action = None

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return self.get_obs()

    def get_obs(self):
        return self.obs.copy()

    def get_env_state(self):
        return {key: value.copy() for key, value in self.state.items()}

    def set_env_state(self, state):
        self.state = {key: value.copy() for key, value in state.items()}

    def step(self, action):
        self.action = action.copy()
        return self.get_obs(), -3., self.done, dict(goal_achieved=True)


def with_prefix(row, initial_observation=None):
    row = dict(row, step_idx=0, native_step_idx=0)
    row['native_replay_trace'] = dict(initial_state={k: row[k].copy() for k in ('qpos', 'qvel', 'desired_orien')},
                                    initial_observation=(row['native_observation'] if initial_observation is None
                                                         else initial_observation).copy(),
                                    actions=np.empty((0, 24)), valid=np.empty(0, dtype=np.int64))
    return row


def test_position_and_axis_both_required_and_strict_boundaries():
    m = module('envs/adroit/fragment.py')
    pos, axis = np.array([0., 0., .2]), np.array([0., 0., 1.])
    assert m.fragment_status(observation(), pos, axis)['fragment_success'] == 1
    assert m.fragment_status(observation(pos=(.075, 0., .2)), pos, axis)['fragment_success'] == 0
    assert m.fragment_status(observation(axis=(np.sqrt(1-.95**2), 0., .95)), pos, axis)['fragment_success'] == 0


def test_drop_is_failure_even_when_at_fragment_goal_and_native_goal_true():
    m = module('envs/adroit/fragment.py')
    native = Native()
    native.obs = observation(pos=(0., 0., .05))
    env = m.PenFragment(native)
    env.set_goal_state((0., 0., .05), (0., 0., 1.))
    _, _, terminated, truncated, info = env.step(np.zeros(24))
    assert not terminated and truncated and info['fragment_success'] == 0
    assert info['fragment_dropped'] == 1


def test_native_done_never_counts_as_success_and_action_is_actual_clipped():
    m = module('envs/adroit/fragment.py')
    native = Native()
    native.done = True
    env = m.PenFragment(native)
    env.set_goal_state((0., 0., .2), (0., 0., 1.))
    proposed = np.linspace(-2, 2, 24)
    _, _, terminated, truncated, info = env.step(proposed)
    assert not terminated and truncated
    np.testing.assert_array_equal(native.action, np.clip(proposed, -1., 1.))
    np.testing.assert_array_equal(info['executed_action'], native.action)
    with pytest.raises(ValueError, match='finite'):
        env.step(np.full(24, np.nan))


def test_restore_all_state_and_goal_uses_future_object_not_native_target():
    m = module('envs/adroit/fragment.py')
    native = Native()
    env = m.PenFragment(native)
    initial = dict(native.get_env_state(), native_observation=native.get_obs(), episode_seed=37)
    initial['qvel'][:] = .1
    initial = with_prefix(initial)
    future = dict(native_observation=observation(pos=(.02, 0., .21), axis=(1., 0., 0.)))
    options = env.reset_options_from_dataset(initial, future)
    _, info = env.reset(seed=37, options=options)
    np.testing.assert_array_equal(env.get_env_state()['qvel'], initial['qvel'])
    np.testing.assert_array_equal(info['fragment_goal_position'], [.02, 0., .21])
    np.testing.assert_array_equal(info['fragment_goal_axis'], [1., 0., 0.])
    assert info['restore_checked'] == 1
    assert native.seed_value == 37


def test_restore_mismatch_is_hold_not_silent_acceptance():
    m = module('envs/adroit/fragment.py')
    native = Native()
    env = m.PenFragment(native)
    row = with_prefix(dict(native.get_env_state(), native_observation=observation(pos=(.01, 0., .2))), native.get_obs())
    goal = dict(native_observation=observation())
    with pytest.raises(RuntimeError, match='HOLD.*restore'):
        env.reset(seed=3, options=env.reset_options_from_dataset(row, goal))


class Reader:
    lengths = np.array([3, 4])
    offsets = np.array([0, 3])
    column_names = ['step_idx', 'episode_seed', 'action']
    episode_column_names = []

    def __init__(self):
        self.data = dict(step_idx=np.array([0, 1, 2, 0, 1, 2, 3]),
                         episode_seed=np.array([11, 11, 11, 12, 12, 12, 12]),
                         action=np.arange(7*24).reshape(7, 24))

    def get_col_data(self, key):
        return self.data[key]

    def get_row_data(self, rows):
        return {key: value[rows] for key, value in self.data.items()}

    def load_chunk(self, episodes, starts, ends):
        return [{key: value[self.offsets[e]+s:self.offsets[e]+end] for key, value in self.data.items()}
                for e, s, end in zip(episodes, starts, ends)]


def test_readonly_aliases_preserve_rows_actions_and_source_columns():
    m = module('envs/adroit/fragment.py')
    original = Reader()
    view = m.EpisodeAliases(original)
    np.testing.assert_array_equal(view.get_col_data('ep_idx'), [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(view.get_col_data('seed'), original.data['episode_seed'])
    rows = view.get_row_data([1, 4])
    np.testing.assert_array_equal(rows['ep_idx'], [0, 1])
    np.testing.assert_array_equal(rows['action'], original.data['action'][[1, 4]])
    chunks = view.load_chunk([1], [1], [4])
    np.testing.assert_array_equal(chunks[0]['ep_idx'], [1, 1, 1])
    np.testing.assert_array_equal(chunks[0]['seed'], [12, 12, 12])
    assert original.column_names == ['step_idx', 'episode_seed', 'action']
    assert set(original.data) == set(original.column_names)


def test_config_locks_mainline_protocol_and_separates_training_stats(tmp_path):
    m = module('scripts/checks/run_adroit_fragment_cem.py')
    cfg = dict(seed=42, policy=str(tmp_path / 'weights.pt'), dataset=dict(stats=str(tmp_path / 'train.h5')),
               eval=dict(dataset_name=str(tmp_path / 'test.h5'), goal_offset_steps=25, eval_budget=50, num_eval=50, img_size=224),
               world=dict(width=256, height=256), plan_config=dict(horizon=5, receding_horizon=5, action_block=5),
               solver=dict(num_samples=300, topk=30, n_steps=10))
    m.validate_config(cfg)
    cfg['dataset']['stats'] = cfg['eval']['dataset_name']
    with pytest.raises(ValueError, match='train.*test'):
        m.validate_config(cfg)
    cfg['dataset']['stats'] = str(tmp_path / 'train.h5')
    cfg['eval']['eval_budget'] = 25
    with pytest.raises(ValueError, match='eval_budget'):
        m.validate_config(cfg)


def test_configs_keep_history_default_one_and_pen_training_affine():
    pix = ROOT / 'config/eval/adroit_pen_pix_fragment.yaml'
    pcd = ROOT / 'config/eval/adroit_pen_pcd_fragment.yaml'
    assert pix.is_file() and pcd.is_file(), 'new Pen fragment configs are missing'
    a, b = yaml.safe_load(pix.read_text()), yaml.safe_load(pcd.read_text())
    assert a['plan_config'] == dict(horizon=5, receding_horizon=5, action_block=5)
    assert a['eval']['goal_offset_steps'] == 25 and a['eval']['eval_budget'] == 50
    assert a['solver']['n_steps'] == 10
    assert b['pcd_process'] == dict(center=[.075, -.02, .225], scale=4., use_rgb=True)
    assert not a['video'] and not b['video']


class LinearNative(Native):
    def __init__(self, done_at=None):
        super().__init__()
        self.history, self.done_at = [], done_at

    def set_env_state(self, state):
        super().set_env_state(state)
        self.obs[:24] = state['qpos'][:24]

    def step(self, action):
        self.history.append(action.copy())
        self.state['qpos'][:24] += action
        self.state['qvel'][:24] = action
        self.obs[:24] = self.state['qpos'][:24]
        return self.get_obs(), -3., len(self.history) == self.done_at, dict(goal_achieved=True)


def replay_row(native, actions):
    initial = native.get_env_state()
    initial_obs = native.get_obs()
    state = {key: value.copy() for key, value in initial.items()}
    state['qpos'][:24] += actions.sum(axis=0)
    if len(actions):
        state['qvel'][:24] = actions[-1]
    obs = initial_obs.copy()
    obs[:24] = state['qpos'][:24]
    return dict(state, native_observation=obs, step_idx=len(actions), native_step_idx=len(actions),
                native_replay_trace=dict(initial_state=initial, initial_observation=initial_obs,
                                         actions=actions, valid=np.ones(len(actions), dtype=np.int64)))


def test_exact_prefix_calls_native_actual_actions_and_checks_endpoint():
    m = module('envs/adroit/fragment.py')
    native = LinearNative()
    actions = np.stack([np.linspace(-.2, .2, 24), np.linspace(.1, -.1, 24)])
    row = replay_row(native, actions)
    env = m.PenFragment(native)
    obs, info = env.reset(seed=41, options=env.reset_options_from_dataset(row, dict(native_observation=observation())))
    np.testing.assert_array_equal(np.asarray(native.history), actions)
    np.testing.assert_array_equal(obs, row['native_observation'])
    np.testing.assert_array_equal(env.get_env_state()['qvel'], row['qvel'])
    assert info['restore_checked'] == 1 and env._t == 0


def test_missing_prefix_is_hold_even_when_direct_set_would_match():
    m = module('envs/adroit/fragment.py')
    native = LinearNative()
    row = replay_row(native, np.zeros((2, 24)))
    del row['native_replay_trace']
    with pytest.raises(RuntimeError, match='HOLD.*prefix'):
        m.PenFragment(native).reset_options_from_dataset(row, dict(native_observation=observation()))


def test_prefix_done_before_sampled_start_is_hold():
    m = module('envs/adroit/fragment.py')
    native = LinearNative(done_at=1)
    row = replay_row(native, np.full((3, 24), .01))
    env = m.PenFragment(native)
    with pytest.raises(RuntimeError, match='HOLD.*done'):
        env.reset(seed=8, options=env.reset_options_from_dataset(row, dict(native_observation=observation())))


class ReplayReader(Reader):
    def __init__(self):
        super().__init__()
        n = len(self.data['step_idx'])
        for key, size in (('qpos', 30), ('qvel', 30), ('desired_orien', 4)):
            self.data['initial_' + key] = np.zeros((n, size))
        self.data['initial_desired_orien'][:, 0] = 1.
        self.data['initial_native_observation'] = np.tile(observation(), (n, 1))
        self.data['executed_action'] = np.arange(n*24).reshape(n, 24) / 1000.
        self.data['executed_action_valid'] = (self.data['step_idx'] > 0).astype(np.int64)
        self.column_names = list(self.data)
        self.h5_file = self.data

    def _open(self):
        pass


def test_episode_scoped_trace_uses_incoming_actions_order_and_duplicate_requests():
    m = module('envs/adroit/fragment.py')
    reader = ReplayReader()
    view = m.EpisodeAliases(reader)
    assert 'native_replay_trace' in view.episode_column_names
    assert 'native_replay_trace' not in view.column_names
    traces = view.get_episode_data([1, 0, 1])['native_replay_trace']
    assert len(traces) == 3
    np.testing.assert_array_equal(traces[0]['actions'], reader.data['executed_action'][4:7])
    np.testing.assert_array_equal(traces[1]['actions'], reader.data['executed_action'][1:3])
    np.testing.assert_array_equal(traces[2]['actions'], traces[0]['actions'])
    np.testing.assert_array_equal(traces[0]['valid'], [1, 1, 1])


def test_pixel_row_view_matches_original_pil_bilinear_and_leaves_source_bytes():
    m = module('envs/adroit/fragment.py')
    reader = Reader()
    pixels = np.arange(7*256*256*3, dtype=np.uint8).reshape(7, 256, 256, 3)
    reader.data['pixels'] = pixels
    reader.column_names = [*reader.column_names, 'pixels']
    before = pixels.tobytes()
    view = m.EpisodeAliases(reader)
    out = view.get_row_data([1, 4])
    assert out['pixels'].shape == (2, 224, 224, 3) and out['pixels'].dtype == np.uint8
    for index, row in enumerate((1, 4)):
        expected = np.array(Image.fromarray(pixels[row]).resize((224, 224), Image.BILINEAR))
        np.testing.assert_array_equal(out['pixels'][index], expected)
    np.testing.assert_array_equal(out['action'], reader.data['action'][[1, 4]])
    assert pixels.tobytes() == before
    assert view.get_row_data(0)['pixels'].shape == (224, 224, 3)


class PixelTensor:
    """CPU layout double; actual HDF5 load_chunk returns uint8 Torch TCHW."""
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def new_tensor(self, array):
        return PixelTensor(np.array(array, copy=True))


class PixelChunkReader(Reader):
    def __init__(self):
        super().__init__()
        self.pixels = np.arange(3*256*256*3, dtype=np.uint8).reshape(3, 256, 256, 3)
        self.column_names = [*self.column_names, 'pixels', 'pcd_xyz', 'pcd_rgb']

    def load_chunk(self, episodes, starts, ends):
        rows = super().load_chunk(episodes, starts, ends)
        for row in rows:
            row['pixels'] = PixelTensor(self.pixels.transpose(0, 3, 1, 2))
            row['pcd_xyz'], row['pcd_rgb'] = np.ones((3, 1200, 3), np.float32), np.ones((3, 1200, 3), np.uint8)
        self.last_chunks = rows
        return rows


def test_pixel_chunk_view_preserves_torch_layout_and_pcd_arrays():
    m = module('envs/adroit/fragment.py')
    reader = PixelChunkReader()
    before = reader.pixels.tobytes()
    chunk = m.EpisodeAliases(reader).load_chunk([0], [0], [3])[0]
    assert isinstance(chunk['pixels'], PixelTensor)
    assert chunk['pixels'].array.shape == (3, 3, 224, 224)
    assert chunk['pixels'].array.dtype == np.uint8
    expected = np.stack([np.array(Image.fromarray(frame).resize((224, 224), Image.BILINEAR)) for frame in reader.pixels])
    np.testing.assert_array_equal(chunk['pixels'].array.transpose(0, 2, 3, 1), expected)
    assert reader.pixels.tobytes() == before
    assert chunk['pcd_xyz'] is reader.last_chunks[0]['pcd_xyz']
    assert chunk['pcd_rgb'] is reader.last_chunks[0]['pcd_rgb']
