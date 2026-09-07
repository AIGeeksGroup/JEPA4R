"""Real spawned workers, without a simulator, renderer, or timing benchmark."""

from functools import partial
import multiprocessing as mp
import os

import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.world.env_pool import EnvPool

from ms2_lewm.process_env_pool import ProcessEnvPool, ProcessWorld


class _FakeEnv(gym.Env):
    """Tiny stateful environment exposing the same collector RPC boundaries."""

    _env_type = "single"
    _arm_sampling_bounds = np.asarray([[-1, -1, 0], [1, 1, 1]])

    def __init__(self, index, barrier=None, fail_step=False, fail_init=False):
        if fail_init:
            raise ValueError("deliberate constructor failure")
        self.index = index
        self.barrier = barrier
        self.fail_step = fail_step
        self.action_space = gym.spaces.Box(-10, 10, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-1e6, 1e6, (3,), np.float32)
        self.episode_seed = 0
        self.n = 0
        self.offset = 0
        self.target = 0.0
        self.last_action = np.zeros(2, np.float32)

    def _info(self):
        return {
            "seed": self.episode_seed,
            "step_idx": self.n,
            "value": np.asarray([self.index, self.n, self.offset], np.float32),
            "action": self.last_action.copy(),
            "label": f"env-{self.index}",
            "success": self.n >= 2,
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.episode_seed = seed
        self.n = 0
        self.offset = (options or {}).get("offset", 0)
        self.target = float(self.offset)
        self.last_action[:] = 0
        info = self._info()
        return info["value"].copy(), info

    def step(self, action):
        if self.fail_step:
            raise ValueError("deliberate step failure")
        if self.barrier is not None:
            # Both workers must enter step before either may finish. A serial
            # send/receive loop fails this test regardless of machine speed.
            self.barrier.wait()
        self.n += 1
        self.last_action = np.asarray(action, np.float32).copy()
        info = self._info()
        return (
            info["value"].copy(),
            float(self.index * 100 + self.n + self.last_action.sum()),
            self.n >= 2 and self.index == 0,
            self.n >= 2 and self.index != 0,
            info,
        )

    def oracle_info(self):
        return {
            "pid": os.getpid(),
            "index": self.index,
            "step": self.n,
            "target": self.target,
            "privileged/target_task": "cube",
        }

    def action_adapter_state(self):
        return {
            "rotation": np.eye(3),
            "qpos": float(self.n) * 0.001,
            "lower": -0.01,
            "upper": 0.04,
        }

    def set_new_target(self, p_stack=0.0):
        self.target = 100.0 + self.index + p_stack
        return self._info()["value"].copy(), self.oracle_info()

    def set_sim_state(self, state):
        self.n = int(np.asarray(state).reshape(-1)[0])

    def set_goal_cube_position(self, target_pos):
        self.target = float(np.asarray(target_pos).reshape(-1)[0])

    def close(self):
        pass


def _make_factories(**kwargs):
    return [partial(_FakeEnv, index=i, **kwargs) for i in range(2)]


def _assert_infos_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def _assert_result_equal(left, right):
    assert left[0] is right[0] is None
    for left_array, right_array in zip(left[1:-1], right[1:-1]):
        np.testing.assert_array_equal(left_array, right_array)
    _assert_infos_equal(left[-1], right[-1])


def _wait_for_stop(ready, stop):
    ready.set()
    stop.wait(60)


def test_world_keeps_official_collection_evaluation_and_episode_loop():
    from stable_worldmodel.world.world import World

    assert ProcessWorld.collect is World.collect
    assert ProcessWorld.evaluate is World.evaluate
    assert ProcessWorld._run_iter is World._run_iter


class _ExpertFakeEnv(_FakeEnv):
    def __init__(self, index):
        super().__init__(index)
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
        self.last_action = np.zeros(7, np.float32)


class _StatefulOracle:
    """Exercises the original expert's ordered RNG and retarget branches."""

    def __init__(self, index, call_log):
        self.index = index
        self.call_log = call_log
        self.done = False

    def reset(self, observation, info):
        self.done = False
        self.steps = 0
        self.offset = float(np.random.uniform(-0.05, 0.05))
        self.call_log.append((os.getpid(), self.index, "reset"))

    def select_action(self, observation, info):
        self.steps += 1
        self.done = self.steps == 2
        self.call_log.append((os.getpid(), self.index, "action"))
        return np.asarray([0.1 + self.offset, -0.2, 0.3, 0.15, -0.1])


def test_workers_have_distinct_pids_and_steps_are_dispatched_concurrently():
    context = mp.get_context("spawn")
    barrier = context.Barrier(2, timeout=15)
    pool = ProcessEnvPool(_make_factories(barrier=barrier), timeout=30)
    try:
        pool.reset(seed=3072)
        pids = [env.unwrapped.oracle_info()["pid"] for env in pool.envs]
        assert len(set(pids)) == 2
        assert os.getpid() not in pids
        actions = np.asarray([[1, 2], [3, 4]], np.float32)
        _, rewards, _, _, infos = pool.step(actions)
        np.testing.assert_array_equal(rewards, [4, 108])
        np.testing.assert_array_equal(infos["action"][:, 0], actions)
        assert [env.oracle_info()["index"] for env in pool.envs] == [0, 1]
        assert [env.oracle_info()["step"] for env in pool.envs] == [1, 1]
    finally:
        pool.close()


def test_seeds_order_mask_resets_and_terminal_observations_match_official_pool():
    serial = EnvPool(_make_factories())
    pool = ProcessEnvPool(_make_factories(), timeout=30)
    try:
        expected = serial.reset(seed=3072, options={"offset": 7})
        actual = pool.reset(seed=3072, options={"offset": 7})
        _assert_result_equal(actual, expected)
        np.testing.assert_array_equal(pool.seeds, [3072, 3073])
        actions = np.asarray([[0.25, -0.5], [1.25, -1.5]], np.float32)
        _assert_result_equal(pool.step(actions), serial.step(actions))
        final = pool.step(actions)
        _assert_result_equal(final, serial.step(actions))
        np.testing.assert_array_equal(final[2], [True, False])
        np.testing.assert_array_equal(final[3], [False, True])
        # No implicit autoreset: terminal observations remain at step 2.
        np.testing.assert_array_equal(final[-1]["step_idx"], [[2], [2]])

        kwargs = {"seed": [88, 99], "options": [{"offset": 8}, {"offset": 9}],
                  "mask": np.asarray([False, True])}
        _assert_result_equal(pool.reset(**kwargs), serial.reset(**kwargs))
        np.testing.assert_array_equal(pool.seeds, [3072, 99])
        mask = np.asarray([False, True])
        actual = pool.step(actions, mask=mask)
        _assert_result_equal(actual, serial.step(actions, mask=mask))
        assert actual[1][0] == 0
        assert not actual[2][0] and not actual[3][0]
        np.testing.assert_array_equal(actual[-1]["step_idx"], [[2], [1]])

        empty_mask = np.asarray([False, False])
        _assert_result_equal(pool.step(actions, mask=empty_mask),
                             serial.step(actions, mask=empty_mask))
        _assert_result_equal(pool.reset(seed=None, mask=empty_mask),
                             serial.reset(seed=None, mask=empty_mask))
        np.testing.assert_array_equal(pool.seeds, serial.seeds)
        # The batched action space remains cached and seedable like upstream.
        assert pool.action_space is pool.action_space
        serial.action_space.seed(19)
        pool.action_space.seed(19)
        for _ in range(2):
            np.testing.assert_array_equal(pool.action_space.sample(), serial.action_space.sample())
    finally:
        pool.close()
        serial.close()


def test_retarget_and_state_rpcs_refresh_oracle_and_action_snapshots():
    pool = ProcessEnvPool(_make_factories(), timeout=30)
    try:
        pool.reset(seed=[17, 23])
        remote = pool.envs[1].unwrapped
        assert remote is pool.envs[1]
        assert remote._env_type == "single"
        np.testing.assert_array_equal(remote._arm_sampling_bounds, _FakeEnv._arm_sampling_bounds)
        obs, info = remote.set_new_target(p_stack=0.25)
        np.testing.assert_array_equal(obs, [1, 0, 0])
        assert info["target"] == 101.25
        assert remote.oracle_info()["target"] == 101.25
        pool.step(np.zeros((2, 2), np.float32))
        assert remote.action_adapter_state()["qpos"] == pytest.approx(0.001)
        assert remote.oracle_info()["target"] == 101.25
        remote.set_sim_state(np.asarray([9]))
        assert remote.oracle_info()["step"] == 9
        assert remote.action_adapter_state()["qpos"] == pytest.approx(0.009)
        remote.set_goal_cube_position(np.asarray([0.3, 0.1, 0.2]))
        assert remote.oracle_info()["target"] == 0.3
        assert pool.envs[0].oracle_info()["target"] == 0.0
    finally:
        pool.close()


def test_expert_noise_rng_action_order_and_retarget_stay_in_parent():
    from ms2_lewm.expert import ManiSkillNoisyExpert, capture_numpy_rng, restore_numpy_rng

    factories = [partial(_ExpertFakeEnv, index=i) for i in range(2)]
    serial = EnvPool(factories)
    pool = ProcessEnvPool(factories, timeout=30)
    prior_rng = capture_numpy_rng()
    try:
        policies, logs = [], [[], []]
        for index, env_pool in enumerate([serial, pool]):
            policy = ManiSkillNoisyExpert(seed=3072)
            policy.set_env(env_pool)
            policy._oracle_agents = {"cube": [_StatefulOracle(i, logs[index]) for i in range(2)]}
            policies.append(policy)
        _, serial_info = serial.reset(seed=3072)
        _, parallel_info = pool.reset(seed=3072)
        np.random.seed(729)
        for step in range(6):
            before = capture_numpy_rng()
            expected = policies[0].get_action(serial_info)
            after = capture_numpy_rng()
            restore_numpy_rng(before)
            actual = policies[1].get_action(parallel_info)
            assert capture_numpy_rng() == after
            assert policies[0].rng.bit_generator.state == policies[1].rng.bit_generator.state
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(policies[0].last_canonical_actions,
                                          policies[1].last_canonical_actions)
            assert logs[0] == logs[1]
            assert all(pid == os.getpid() for pid, _, _ in logs[1])
            # A selective reset also exercises per-episode xi and parent RNG.
            if step == 3:
                mask = np.asarray([True, False])
                _, serial_info = serial.reset(seed=[97, None], mask=mask)
                _, parallel_info = pool.reset(seed=[97, None], mask=mask)
            else:
                serial_info = serial.step(expected)[-1]
                parallel_info = pool.step(actual)[-1]
            _assert_infos_equal(parallel_info, serial_info)
        np.testing.assert_array_equal(policies[0].episode_noise_scales,
                                      policies[1].episode_noise_scales)
        assert any(event == "reset" for _, _, event in logs[1][4:])
    finally:
        restore_numpy_rng(prior_rng)
        pool.close()
        serial.close()


def test_worker_exception_is_reported_and_all_owned_workers_are_reaped():
    pool = ProcessEnvPool(_make_factories(fail_step=True), timeout=10)
    try:
        pool.reset(seed=42)
        pids = {env.oracle_info()["pid"] for env in pool.envs}
        with pytest.raises(RuntimeError, match="deliberate step failure"):
            pool.step(np.zeros((2, 2), np.float32))
    finally:
        pool.close()
    assert pids.isdisjoint({child.pid for child in mp.active_children()})
    pool.close()  # Idempotent cleanup after an exception.


def test_constructor_exception_does_not_leak_earlier_workers():
    previous = {child.pid for child in mp.active_children()}
    factories = [partial(_FakeEnv, index=0), partial(_FakeEnv, index=1, fail_init=True)]
    with pytest.raises(RuntimeError, match="deliberate constructor failure"):
        ProcessEnvPool(factories, timeout=15)
    assert {child.pid for child in mp.active_children()} <= previous


def test_close_leaves_an_unrelated_process_running():
    context = mp.get_context("spawn")
    ready, stop = context.Event(), context.Event()
    unrelated = context.Process(target=_wait_for_stop, args=(ready, stop))
    unrelated.start()
    pool = None
    try:
        assert ready.wait(20)
        pool = ProcessEnvPool(_make_factories(), timeout=30)
        pool.reset(seed=42)
        worker_pids = {env.oracle_info()["pid"] for env in pool.envs}
        pool.close()
        assert unrelated.is_alive()
        assert worker_pids.isdisjoint({child.pid for child in mp.active_children()})
    finally:
        if pool is not None:
            pool.close()
        stop.set()
        unrelated.join(20)
        if unrelated.is_alive():
            unrelated.terminate()
            unrelated.join(5)
