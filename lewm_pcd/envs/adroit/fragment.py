"""Adroit fragment extensions; pure NumPy contracts, no expert or simulator import."""
from pathlib import Path
import numpy as np


def vector(value, size, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError('%s must be a finite (%d,) vector' % (name, size))
    return out.copy()



class NativeFragment:
    """Shared exact-reset and transition boundary; task-specific methods live below."""
    def __init__(self, native_env, width=256, height=256, camera='vil_camera'):
        self.native_env = native_env
        self.width, self.height, self.camera = int(width), int(height), camera
        self._goal, self._t = None, 0
        self._restore_checked, self._restore_error = 0, 0.

    def get_obs(self):
        return vector(self.native_env.get_obs(), self.observation_dim, 'native_observation')

    def get_env_state(self):
        return self._state(self.native_env.get_env_state())

    def set_env_state(self, state):
        self.native_env.set_env_state(self._state(state))

    @property
    def proprio(self):
        return self.get_obs()[:self.action_dim].copy()

    def reset_options_from_dataset(self, init_row, goal_row):
        if 'native_replay_trace' not in init_row:
            raise RuntimeError('HOLD: missing exact native replay prefix')
        trace = init_row['native_replay_trace']
        n = int(init_row['native_step_idx'])
        actions = np.asarray(trace['actions'], dtype=np.float64)
        valid = np.asarray(trace['valid']).reshape(-1)
        if n != int(init_row['step_idx']) or not 0 <= n <= len(actions):
            raise RuntimeError('HOLD: native replay prefix step indices disagree')
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or len(valid) != len(actions):
            raise RuntimeError('HOLD: native replay prefix has invalid shape')
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.) or np.any(valid[:n] != 1):
            raise RuntimeError('HOLD: native replay prefix actions must be actual finite valid controls')
        initial_state = self._state(trace['initial_state'])
        return dict(state=self._state(init_row), goal=self._goal_from_row(goal_row, initial_state),
                    initial_state=initial_state,
                    initial_observation=vector(trace['initial_observation'], self.observation_dim, 'initial observation'),
                    prefix_actions=actions[:n].copy(),
                    expected_observation=vector(init_row['native_observation'], self.observation_dim, 'native_observation'))

    def reset(self, *, seed=None, options=None):
        self._goal, self._t = None, 0
        self._restore_checked, self._restore_error = 0, 0.
        if seed is not None:
            self.native_env.seed(int(seed))
        self.native_env.reset()
        if options:
            state = self._state(options['state'])
            self._restore_initial(options['initial_state'])
            if not np.allclose(self.get_obs(), options['initial_observation'], atol=1e-6, rtol=1e-5):
                raise RuntimeError('HOLD: initial native observation restore mismatch')
            for action in options['prefix_actions']:
                _, _, done, _ = self.native_env.step(action)
                if done:
                    raise RuntimeError('HOLD: native replay prefix reached done before a valid fragment start')
            actual = self.get_env_state()
            if any(not np.allclose(actual[key], state[key], atol=1e-6, rtol=1e-5) for key in state):
                raise RuntimeError('HOLD: replayed native state restore differs in ' + '/'.join(state))
            obs = self.get_obs()
            expected = vector(options['expected_observation'], self.observation_dim, 'expected observation')
            self._restore_error = float(np.max(np.abs(obs - expected)))
            if not np.allclose(obs, expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError('HOLD: native observation restore mismatch max_abs=%g' % self._restore_error)
            self.set_goal_state(*options['goal'])
            self._restore_checked = 1
        return self.get_obs(), self._info()

    def _info(self):
        obs = self.get_obs()
        info = dict(self.get_env_state(), native_observation=obs, proprio=obs[:self.action_dim].copy(),
                    restore_checked=np.int64(self._restore_checked), restore_obs_max_abs=np.float64(self._restore_error))
        if self._goal is not None:
            info.update(self._status(obs))
            info.update(self._goal_info())
        return info

    def step(self, action):
        proposed = vector(action, self.action_dim, 'action')
        # Native Adroit.step clips normalized controls before its actuator mapping.
        actual = np.clip(proposed, -1., 1.)
        obs, reward, native_done, native_info = self.native_env.step(actual)
        self._t += 1
        obs = vector(obs, self.observation_dim, 'native_observation')
        info = self._info()
        dropped = self._failure(obs, native_done)
        status = self._status(obs, native_done=native_done) if self._goal is not None else {}
        info.update(status, native_done=np.int64(native_done),
                    native_goal_achieved=np.int64(bool(native_info.get('goal_achieved', False))),
                    executed_action=actual.copy(), action_clip_max_abs=np.float64(np.max(np.abs(actual-proposed))))
        success = bool(status.get('fragment_success', 0))
        # World.evaluate counts terminated as success. Drop must only end as failure.
        return obs, float(reward), success, bool(dropped or (self._t >= 100 and not success)), info


RELOCATE_STATE_FIELDS = (('qpos', 36), ('qvel', 36), ('hand_qpos', 30),
                         ('obj_pos', 3), ('palm_pos', 3), ('target_pos', 3))


class RelocateFragment(NativeFragment):
    """Future-object-position fragment; single-frame <.1 is separate from native >25-frame success."""
    task = 'relocate-v0'
    action_dim, observation_dim, qpos_dim = 30, 39, 36

    def _state(self, row):
        return {key: vector(row[key], size, key) for key, size in RELOCATE_STATE_FIELDS}

    def _goal_from_row(self, goal_row, initial_state):
        obs = vector(goal_row['goal_native_observation'] if 'goal_native_observation' in goal_row
                     else goal_row['native_observation'], 39, 'goal native observation')
        # Native obs ends in obj world position - native target position; state.obj_pos is the model base.
        return (obs[36:39] + initial_state['target_pos'],)

    def _restore_initial(self, state):
        actual = self.get_env_state()
        if any(not np.allclose(actual[key], value, atol=1e-6, rtol=1e-5) for key, value in state.items()):
            raise RuntimeError('HOLD: seeded reset initial native state restore mismatch')

    def set_goal_state(self, position):
        self._goal = (vector(position, 3, 'goal position'),)

    def _status(self, obs, native_done=False):
        position = obs[36:39] + self.get_env_state()['target_pos']
        distance = float(np.linalg.norm(position - self._goal[0]))
        return dict(fragment_distance=np.float64(distance), fragment_failed=np.int64(native_done),
                    fragment_success=np.int64(not native_done and distance < .1))

    def _goal_info(self):
        return dict(fragment_goal_position=self._goal[0].copy())

    def _failure(self, obs, native_done):
        return bool(native_done)


class EpisodeAliases:
    """Read-only view of the frozen HF columns; values/actions remain untouched."""
    def __init__(self, dataset, pixel_shape=(224, 224), task='relocate-v0', initial_states=None, rgb_frames=None):
        self.dataset = dataset
        if task != 'relocate-v0':
            raise ValueError('unsupported Adroit fragment task: ' + task)
        self.task, self.rgb_frames = task, rgb_frames
        self.pixel_shape = tuple(int(v) for v in pixel_shape)
        lengths, offsets = np.asarray(dataset.lengths, np.int64), np.asarray(dataset.offsets, np.int64)
        expected = np.r_[0, np.cumsum(lengths)[:-1]]
        if np.any(lengths <= 0) or not np.array_equal(offsets, expected):
            raise ValueError('episode offsets/lengths must cover contiguous nonempty episodes')
        self._episode_idx = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        if task == 'relocate-v0':
            if isinstance(initial_states, (str, Path)):
                with np.load(initial_states, allow_pickle=False) as packed:
                    initial_states = {key: np.array(packed[key], copy=True) for key in packed.files}
            if initial_states is None:
                raise ValueError('Relocate fragment requires verified initial states')
            self._initial_states = {key: np.array(value, copy=True) for key, value in initial_states.items()}
            if not np.array_equal(self._initial_states['original_episode_index'], np.arange(len(lengths))):
                raise ValueError('initial states must match dataset episode order')
            for key, size in (*RELOCATE_STATE_FIELDS, ('reset_observation', 39)):
                value = self._initial_states[key]
                if value.shape != (len(lengths), size) or not np.isfinite(value).all():
                    raise ValueError('initial states have invalid ' + key)
            first = dataset.get_row_data(offsets.tolist())
            for source, target in [('seed', 'seed'), ('qpos', 'qpos'), ('qvel', 'qvel'), ('observation', 'reset_observation')]:
                if not np.array_equal(np.asarray(first[source]), self._initial_states[target]):
                    raise ValueError('initial states disagree with dataset ' + source)
        if rgb_frames is not None:
            import h5py
            with h5py.File(rgb_frames, 'r') as frames:
                rows = np.asarray(frames['row_index'])
                if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows) or np.any(rows < 0):
                    raise ValueError('RGB cache row_index must contain unique nonnegative integers')
                if frames['pixels'].shape != (len(rows), 256, 256, 3) or frames['pixels'].dtype != np.uint8:
                    raise ValueError('RGB cache requires uint8 (N,256,256,3) pixels')
            self._rgb_index = {int(row): index for index, row in enumerate(rows)}

    def __getattr__(self, key):
        return getattr(self.dataset, key)

    @property
    def column_names(self):
        extra = ['native_observation', 'native_step_idx', 'obj_pos', 'palm_pos', 'target_pos', 'hand_qpos']
        if self.rgb_frames is not None:
            extra.append('pixels')
        return list(dict.fromkeys([*self.dataset.column_names, 'ep_idx', 'seed', *extra]))

    @property
    def episode_column_names(self):
        return [*self.dataset.episode_column_names, 'native_replay_trace']

    def get_episode_data(self, episodes_idx=None):
        episodes = range(len(self.dataset.lengths)) if episodes_idx is None else episodes_idx
        out = self.dataset.get_episode_data(episodes) if self.dataset.episode_column_names else {}
        self.dataset._open()
        h5 = self.dataset.h5_file
        traces = []
        for episode in episodes:
            offset, length = int(self.dataset.offsets[episode]), int(self.dataset.lengths[episode])
            actions = np.array(h5['action'][offset:offset+length], copy=True)
            if actions.shape != (length, 30) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.):
                raise RuntimeError('HOLD: Relocate replay controls must be actual finite bounded actions')
            traces.append(dict(initial_state={key: self._initial_states[key][episode].copy() for key, _ in RELOCATE_STATE_FIELDS},
                               initial_observation=self._initial_states['reset_observation'][episode].copy(),
                               actions=actions, valid=np.ones(length, np.int64)))
        out['native_replay_trace'] = traces
        return out

    def get_col_data(self, key):
        if key == 'ep_idx':
            return self._episode_idx.copy()
        return self.dataset.get_col_data(key)

    def get_row_data(self, rows):
        out = dict(self.dataset.get_row_data(rows))
        out.update(ep_idx=self._episode_idx[rows], seed=np.array(out['seed'], copy=True))
        self._relocate_columns(out, out['ep_idx'])
        if self.rgb_frames is not None:
            out['pixels'] = self._rgb(rows)
        if 'pixels' in out:
            out['pixels'] = self._pixel_view(out['pixels'])
        return out

    def _relocate_columns(self, row, episodes):
        row['native_observation'], row['native_step_idx'] = row['observation'], row['step_idx']
        row['hand_qpos'] = row['qpos'][..., :30]
        for key in ('obj_pos', 'palm_pos', 'target_pos'):
            row[key] = self._initial_states[key][episodes].copy()

    def _rgb(self, rows):
        import h5py
        values = np.asarray(rows)
        indices = []
        for row in values.reshape(-1):
            if int(row) not in self._rgb_index:
                raise RuntimeError('HOLD: missing real RGB cache row %d' % row)
            indices.append(self._rgb_index[int(row)])
        with h5py.File(self.rgb_frames, 'r') as frames:
            pixels = np.stack([frames['pixels'][index] for index in indices])
        return pixels[0] if values.ndim == 0 else pixels

    def _pixel_view(self, value):
        # Same three operations as SWM AddPixelsWrapper._process_img; no WM transform here.
        from PIL import Image
        tensor = hasattr(value, 'detach')
        array = value.detach().cpu().numpy().transpose(0, 2, 3, 1) if tensor else np.asarray(value)
        single = array.ndim == 3
        if single:
            array = array[None]
        if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError('pixels view requires uint8 RGB frames')
        height, width = self.pixel_shape
        resized = np.stack([np.array(Image.fromarray(frame).resize((width, height), Image.BILINEAR))
                            for frame in array])
        if tensor:
            return value.new_tensor(resized.transpose(0, 3, 1, 2))
        return resized[0] if single else resized

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        out = []
        for episode, first, last, chunk in zip(episodes_idx, start, end, chunks):
            row = dict(chunk)
            if self.rgb_frames is not None:
                import torch
                offset = int(self.dataset.offsets[episode])
                row['pixels'] = torch.from_numpy(self._rgb(np.arange(offset+first, offset+last)).transpose(0, 3, 1, 2).copy())
            if 'pixels' in row:
                row['pixels'] = self._pixel_view(row['pixels'])
            row['seed'] = np.array(row['seed'], copy=True)
            row['ep_idx'] = np.full(len(row['seed']), int(episode), np.int64)
            self._relocate_columns(row, row['ep_idx'])
            out.append(row)
        return out
