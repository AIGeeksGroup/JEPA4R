"""CPU task contracts; these tests do not establish native replay or CEM scores."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import copy
import hashlib
import json
import shutil
import types

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
    """Layout-only CPU double; native Torch remains a separate runtime gate."""
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


def load_check(name):
    path = ROOT / 'scripts/checks' / (name + '.py')
    spec = importlib.util.spec_from_file_location('relocate_' + name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_fragment_product_is_relocate_only_and_has_generic_shared_transition():
    m = module()
    assert hasattr(m, 'NativeFragment')
    assert not hasattr(m, 'PenFragment')
    with pytest.raises(ValueError, match='Relocate|unsupported'):
        m.EpisodeAliases(Reader(), task='pen-v0')


def test_only_final_relocate_collection_and_fragment_are_registered(monkeypatch):
    calls = []
    registration = types.ModuleType('gymnasium.envs.registration')
    registration.registry = {}
    registration.register = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, registration.__name__, registration)
    spec = importlib.util.spec_from_file_location('final_adroit_registrations', ROOT / 'envs/adroit/__init__.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert [call['id'] for call in calls] == ['swm/AdroitRelocate-v0', 'swm/AdroitRelocateFragment-v0']
    assert 'PenFragmentEnv' not in loaded.__all__


def test_cem_cli_rejects_pen_and_full_budget_is_unchanged():
    entry = load_check('run_adroit_fragment_cem')
    parser = entry.make_parser()
    supplied = ['--code-root', '/code', '--route', 'pcd', '--initials', '/initial.npz',
                '--train', '/train.h5', '--test', '/test.h5', '--checkpoint', '/weights.pt', '--out', '/fresh']
    args = parser.parse_args(supplied)
    assert args.task == 'relocate-v0' and args.mode == 'full'
    with pytest.raises(SystemExit):
        parser.parse_args(supplied + ['--task', 'pen-v0'])
    assert entry.budget('full') == dict(num_eval=50, eval_budget=50, num_samples=300,
                                      topk=30, n_steps=10, train_episodes=10000, test_episodes=150)
    assert entry.budget('smoke')['num_eval'] == 1
    assert entry.budget('smoke')['eval_budget'] >= 25
    with pytest.raises(ValueError):
        entry.budget('unknown')


def test_relocate_config_validation_distinguishes_diagnostic_from_full_protocol(tmp_path):
    entry = load_check('run_adroit_fragment_cem')
    cfg = dict(seed=42, policy=str(tmp_path/'weights.pt'), dataset=dict(stats=str(tmp_path/'train.h5')),
               eval=dict(goal_offset_steps=25, eval_budget=50, num_eval=50, img_size=224, dataset_name=str(tmp_path/'test.h5')),
               plan_config=dict(horizon=5, receding_horizon=5, action_block=5),
               solver=dict(num_samples=300, topk=30, n_steps=10),
               world=dict(env_name='envs.adroit:swm/AdroitRelocateFragment-v0', width=256, height=256))
    entry.validate_config(cfg)
    diagnostic = copy.deepcopy(cfg)
    entry.apply_budget(diagnostic, 'smoke')
    entry.validate_config(diagnostic, 'smoke')
    with pytest.raises(ValueError):
        entry.validate_config(diagnostic, 'full')
    cfg['world']['env_name'] = 'envs.adroit:swm/AdroitPenFragment-v0'
    with pytest.raises(ValueError):
        entry.validate_config(cfg)


def test_rgb_windows_use_the_same_full_sampling_and_explicit_one_fragment_smoke():
    helper = load_check('prepare_relocate_fragment_rgb')
    lengths = np.full(150, 200)
    steps = np.tile(np.arange(200), 150)
    maximum = np.repeat(lengths - 26, lengths)
    valid = np.flatnonzero(steps <= maximum)
    expected = np.sort(valid[np.random.default_rng(42).choice(len(valid)-1, size=50, replace=False)])
    np.testing.assert_array_equal(helper.selected_rows(lengths, steps), expected)
    smoke = helper.selected_rows(lengths[:1], steps[:200], mode='smoke')
    assert smoke.shape == (1,) and 0 <= smoke[0] <= 174
    with pytest.raises(ValueError):
        helper.selected_rows(lengths, steps, mode='invalid')


def test_rgb_cache_producer_binds_current_dataset_initials_mode_and_payload(tmp_path):
    entry = load_check('run_adroit_fragment_cem')
    data,initials = tmp_path/'current_test.h5',tmp_path/'current_initials.npz'
    with h5py.File(data,'w') as handle:handle['value']=np.arange(3)
    initials.write_bytes(b'initial state content')
    cache_dir = tmp_path/'moved-cache';cache_dir.mkdir()
    cache=cache_dir/'renamed_cache.h5';cache.write_bytes(b'cache content fixture')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(status='COMPLETE',exit_code=0,mode='smoke',cache_sha256=digest(cache),
                inputs={'/former/location/test.h5':digest(data),'/former/location/init.npz':digest(initials)},
                input_dependencies={'test':entry.hdf_dependencies(data)})
    (cache_dir/'report.json').write_text(json.dumps(report))
    # Old absolute paths and cache filename are contextual; equal bytes travel.
    assert entry.validate_rgb_cache(cache,data,initials,'smoke')['passed']
    for key,value in (('status','RUNNING'),('exit_code',1)):
        invalid=dict(report);invalid[key]=value
        (cache_dir/'report.json').write_text(json.dumps(invalid))
        with pytest.raises(ValueError,match='producer'):
            entry.validate_rgb_cache(cache,data,initials,'smoke')
    (cache_dir/'report.json').write_text(json.dumps(dict(report,input_dependencies={})))
    with pytest.raises(ValueError,match='dependencies'):
        entry.validate_rgb_cache(cache,data,initials,'smoke')
    (cache_dir/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='mode'):
        entry.validate_rgb_cache(cache,data,initials,'full')
    old=data.read_bytes()
    with h5py.File(data,'r+') as handle:handle['value'][0]=99
    with pytest.raises(ValueError,match='test|dataset'):
        entry.validate_rgb_cache(cache,data,initials,'smoke')
    data.write_bytes(old)
    old=initials.read_bytes();initials.write_bytes(b'other initial states')
    with pytest.raises(ValueError,match='initial'):
        entry.validate_rgb_cache(cache,data,initials,'smoke')
    initials.write_bytes(old);cache.write_bytes(b'tampered cache')
    with pytest.raises(ValueError,match='cache'):
        entry.validate_rgb_cache(cache,data,initials,'smoke')


def test_rgb_cache_rejects_changed_external_backing_with_identical_header(tmp_path):
    entry=load_check('run_adroit_fragment_cem')
    backing=tmp_path/'test_pcd.h5';header=tmp_path/'test_nonvisual.h5'
    with h5py.File(backing,'w') as handle:handle['action']=np.zeros((2,30))
    with h5py.File(header,'w') as handle:handle['action']=h5py.ExternalLink(backing.name,'/action')
    initials=tmp_path/'initials.npz';initials.write_bytes(b'initial state fixture')
    cache_dir=tmp_path/'cache';cache_dir.mkdir();cache=cache_dir/'rgb_windows.h5';cache.write_bytes(b'RGB fixture')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(status='COMPLETE',exit_code=0,mode='smoke',cache_sha256=digest(cache),
        inputs={str(header):digest(header),str(initials):digest(initials)},
        input_dependencies={'test':entry.hdf_dependencies(header)})
    (cache_dir/'report.json').write_text(json.dumps(report))
    assert entry.validate_rgb_cache(cache,header,initials,'smoke')['passed']
    original_header=header.read_bytes()
    with h5py.File(backing,'r+') as handle:handle['action'][0,0]=.5
    assert header.read_bytes()==original_header
    with pytest.raises(ValueError,match='dependencies|backing'):
        entry.validate_rgb_cache(cache,header,initials,'smoke')


def test_hdf_binding_handles_visible_vds_and_rejects_dependency_cycles(tmp_path):
    entry=load_check('run_adroit_fragment_cem')
    backing=tmp_path/'backing.h5';virtual=tmp_path/'virtual.h5'
    with h5py.File(backing,'w') as handle:handle['x']=np.arange(3)
    layout=h5py.VirtualLayout(shape=(3,),dtype='int64')
    layout[:]=h5py.VirtualSource(backing.name,'x',shape=(3,))
    with h5py.File(virtual,'w',libver='latest') as handle:handle.create_virtual_dataset('x',layout)
    first=entry.hdf_dependencies(virtual)
    assert first['file_count']==2
    with h5py.File(backing,'r+') as handle:handle['x'][1]=99
    assert entry.hdf_dependencies(virtual)['content_sha256']!=first['content_sha256']
    a,b=tmp_path/'a.h5',tmp_path/'b.h5'
    with h5py.File(a,'w') as handle:handle['loop']=h5py.ExternalLink(b.name,'/')
    with h5py.File(b,'w') as handle:handle['loop']=h5py.ExternalLink(a.name,'/')
    with pytest.raises(ValueError,match='cycle'):
        entry.hdf_dependencies(a)


def test_hdf_binding_is_portable_but_does_not_confuse_swapped_backing_roles(tmp_path):
    entry=load_check('run_adroit_fragment_cem')
    original=tmp_path/'original';original.mkdir()
    a,b=original/'a.h5',original/'b.h5';header=original/'header.h5'
    for path,value in ((a,1),(b,2)):
        with h5py.File(path,'w') as handle:handle['x']=[value]
    with h5py.File(header,'w') as handle:
        handle['left']=h5py.ExternalLink(a.name,'/x')
        handle['right']=h5py.ExternalLink(b.name,'/x')
    bound=entry.hdf_dependencies(header)
    moved=tmp_path/'moved';shutil.copytree(original,moved)
    assert entry.hdf_dependencies(moved/'header.h5')['content_sha256']==bound['content_sha256']
    first,second=a.read_bytes(),b.read_bytes();a.write_bytes(second);b.write_bytes(first)
    changed=entry.hdf_dependencies(header)
    assert {row['sha256'] for row in changed['files']}=={row['sha256'] for row in bound['files']}
    assert changed['content_sha256']!=bound['content_sha256']


def test_hdf_binding_deduplicates_files_and_fails_closed_on_soft_links(tmp_path):
    entry=load_check('run_adroit_fragment_cem')
    backing,header=tmp_path/'backing.h5',tmp_path/'header.h5'
    with h5py.File(backing,'w') as handle:handle['x']=[1]
    with h5py.File(header,'w') as handle:
        handle['one']=h5py.ExternalLink(backing.name,'/x')
        handle['two']=h5py.ExternalLink(backing.name,'/x')
    assert entry.hdf_dependencies(header)['file_count']==2
    with h5py.File(header,'r+') as handle:handle['alias']=h5py.SoftLink('/one')
    with pytest.raises(ValueError,match='unsupported'):
        entry.hdf_dependencies(header)
