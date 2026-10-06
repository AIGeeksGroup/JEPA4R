"""Pen-only fragment extension; pure NumPy contracts, no expert or simulator import."""
import numpy as np


def vector(value, size, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError('%s must be a finite (%d,) vector' % (name, size))
    return out.copy()


def native_state_from_row(row):
    return {key: vector(row[key], size, key)
            for key, size in (('qpos', 30), ('qvel', 30), ('desired_orien', 4))}


def pen_pose(obs):
    obs = vector(obs, 45, 'native_observation')
    # Pinned PenEnvV0.get_obs: hand24, object position3, velocity6, axis3.
    return obs[24:27].copy(), obs[33:36].copy()


def fragment_status(obs, goal_position, goal_axis, native_done=False):
    position, axis = pen_pose(obs)
    goal_position, goal_axis = vector(goal_position, 3, 'goal position'), vector(goal_axis, 3, 'goal axis')
    if not np.isclose(np.linalg.norm(goal_axis), 1., atol=1e-3, rtol=0):
        raise ValueError('goal axis must be unit length')
    distance, dot = float(np.linalg.norm(position - goal_position)), float(np.dot(axis, goal_axis))
    dropped = bool(native_done or position[2] < .075)
    return dict(fragment_distance=np.float64(distance), fragment_axis_dot=np.float64(dot),
                fragment_dropped=np.int64(dropped),
                fragment_success=np.int64(not dropped and distance < .075 and dot > .95))


class PenFragment:
    """Native transition/reset contract; Gymnasium subclass adds registration/spaces."""
    task = 'pen-v0'
    action_dim, observation_dim, qpos_dim = 24, 45, 30

    def __init__(self, native_env, width=256, height=256, camera='vil_camera'):
        self.native_env = native_env
        self.width, self.height, self.camera = int(width), int(height), camera
        self._goal, self._t = None, 0
        self._restore_checked, self._restore_error = 0, 0.

    def get_obs(self):
        return vector(self.native_env.get_obs(), 45, 'native_observation')

    def get_env_state(self):
        return native_state_from_row(self.native_env.get_env_state())

    def set_env_state(self, state):
        self.native_env.set_env_state(native_state_from_row(state))

    @property
    def proprio(self):
        return self.get_obs()[:24].copy()

    def set_goal_state(self, position, axis):
        # Validate the target without moving native task geometry or changing physics.
        position, axis = vector(position, 3, 'goal position'), vector(axis, 3, 'goal axis')
        fragment_status(self.get_obs(), position, axis)
        self._goal = (position, axis)

    def reset_options_from_dataset(self, init_row, goal_row):
        if 'native_replay_trace' not in init_row:
            raise RuntimeError('HOLD: missing exact native replay prefix')
        trace = init_row['native_replay_trace']
        n = int(init_row['native_step_idx'])
        actions = np.asarray(trace['actions'], dtype=np.float64)
        valid = np.asarray(trace['valid']).reshape(-1)
        if n != int(init_row['step_idx']) or not 0 <= n <= len(actions):
            raise RuntimeError('HOLD: native replay prefix step indices disagree')
        if actions.ndim != 2 or actions.shape[1] != 24 or len(valid) != len(actions):
            raise RuntimeError('HOLD: native replay prefix has invalid shape')
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.) or np.any(valid[:n] != 1):
            raise RuntimeError('HOLD: native replay prefix actions must be actual finite valid controls')
        position, axis = pen_pose(goal_row['goal_native_observation'] if 'goal_native_observation' in goal_row
                                  else goal_row['native_observation'])
        return dict(state=native_state_from_row(init_row), goal=(position, axis),
                    initial_state=native_state_from_row(trace['initial_state']),
                    initial_observation=vector(trace['initial_observation'], 45, 'initial observation'),
                    prefix_actions=actions[:n].copy(),
                    expected_observation=vector(init_row['native_observation'], 45, 'native_observation'))

    def reset(self, *, seed=None, options=None):
        self._goal, self._t = None, 0
        self._restore_checked, self._restore_error = 0, 0.
        if seed is not None:
            self.native_env.seed(int(seed))
        self.native_env.reset()
        if options:
            state = native_state_from_row(options['state'])
            self.set_env_state(options['initial_state'])
            if not np.allclose(self.get_obs(), options['initial_observation'], atol=1e-6, rtol=1e-5):
                raise RuntimeError('HOLD: initial native observation restore mismatch')
            for action in options['prefix_actions']:
                _, _, done, _ = self.native_env.step(action)
                if done:
                    raise RuntimeError('HOLD: native replay prefix reached done before a valid fragment start')
            actual = self.get_env_state()
            if any(not np.allclose(actual[key], state[key], atol=1e-6, rtol=1e-5) for key in state):
                raise RuntimeError('HOLD: replayed native state restore differs in qpos/qvel/desired_orien')
            obs = self.get_obs()
            expected = vector(options['expected_observation'], 45, 'expected observation')
            self._restore_error = float(np.max(np.abs(obs - expected)))
            if not np.allclose(obs, expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError('HOLD: native observation restore mismatch max_abs=%g' % self._restore_error)
            self.set_goal_state(*options['goal'])
            self._restore_checked = 1
        return self.get_obs(), self._info()

    def _info(self):
        obs = self.get_obs()
        info = dict(self.get_env_state(), native_observation=obs, proprio=obs[:24].copy(),
                    restore_checked=np.int64(self._restore_checked), restore_obs_max_abs=np.float64(self._restore_error))
        if self._goal is not None:
            info.update(fragment_status(obs, *self._goal))
            info.update(fragment_goal_position=self._goal[0].copy(), fragment_goal_axis=self._goal[1].copy())
        return info

    def step(self, action):
        proposed = vector(action, 24, 'action')
        # Native Pen.step clips normalized controls before its actuator mapping.
        actual = np.clip(proposed, -1., 1.)
        obs, reward, native_done, native_info = self.native_env.step(actual)
        self._t += 1
        obs = vector(obs, 45, 'native_observation')
        info = self._info()
        dropped = bool(native_done or obs[26] < .075)
        status = fragment_status(obs, *self._goal, native_done=native_done) if self._goal is not None else {}
        info.update(status, native_done=np.int64(native_done),
                    native_goal_achieved=np.int64(bool(native_info.get('goal_achieved', False))),
                    executed_action=actual.copy(), action_clip_max_abs=np.float64(np.max(np.abs(actual-proposed))))
        success = bool(status.get('fragment_success', 0))
        # World.evaluate counts terminated as success. Drop must only end as failure.
        return obs, float(reward), success, bool(dropped or (self._t >= 100 and not success)), info


class EpisodeAliases:
    """Read-only view of the frozen HF columns; values/actions remain untouched."""
    def __init__(self, dataset, pixel_shape=(224, 224)):
        self.dataset = dataset
        self.pixel_shape = tuple(int(v) for v in pixel_shape)
        lengths, offsets = np.asarray(dataset.lengths, np.int64), np.asarray(dataset.offsets, np.int64)
        expected = np.r_[0, np.cumsum(lengths)[:-1]]
        if np.any(lengths <= 0) or not np.array_equal(offsets, expected):
            raise ValueError('episode offsets/lengths must cover contiguous nonempty episodes')
        self._episode_idx = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)

    def __getattr__(self, key):
        return getattr(self.dataset, key)

    @property
    def column_names(self):
        return list(dict.fromkeys([*self.dataset.column_names, 'ep_idx', 'seed']))

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
            traces.append(dict(initial_state={key: np.array(h5['initial_' + key][offset], copy=True)
                                              for key in ('qpos', 'qvel', 'desired_orien')},
                               initial_observation=np.array(h5['initial_native_observation'][offset], copy=True),
                               actions=np.array(h5['executed_action'][offset+1:offset+length], copy=True),
                               valid=np.array(h5['executed_action_valid'][offset+1:offset+length], copy=True)))
        out['native_replay_trace'] = traces
        return out

    def get_col_data(self, key):
        if key == 'ep_idx':
            return self._episode_idx.copy()
        return self.dataset.get_col_data('episode_seed' if key == 'seed' else key)

    def get_row_data(self, rows):
        out = dict(self.dataset.get_row_data(rows))
        out.update(ep_idx=self._episode_idx[rows], seed=np.array(out['episode_seed'], copy=True))
        if 'pixels' in out:
            out['pixels'] = self._pixel_view(out['pixels'])
        return out

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
        for episode, chunk in zip(episodes_idx, chunks):
            row = dict(chunk)
            if 'pixels' in row:
                row['pixels'] = self._pixel_view(row['pixels'])
            row['seed'] = np.array(row['episode_seed'], copy=True)
            row['ep_idx'] = np.full(len(row['episode_seed']), int(episode), np.int64)
            out.append(row)
        return out
