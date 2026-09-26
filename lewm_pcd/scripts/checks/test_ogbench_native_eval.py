import gymnasium as gym
import numpy as np

from eval_ogbench import NativeBatchedPolicy, evaluate_native, summarize_results


class FakeActionSpace(gym.spaces.Box):
    def __init__(self):
        super().__init__(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class FakeNativeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.reset_calls = []
        self._task_id = None

    def reset(self, *, seed=None, options=None):
        self._task_id = options["task_id"]
        self.reset_calls.append((seed, dict(options)))
        return np.array([self._task_id], dtype=np.float32), {"success": False}

    def step(self, action):
        return (
            np.array([self._task_id], dtype=np.float32),
            0.0,
            self._task_id == 1,
            self._task_id == 2,
            {"success": self._task_id == 1},
        )

    def close(self):
        pass


class RecordingPolicy:
    def __init__(self, env):
        self.env = env
        self.reset_count = 0
        self.goal_seen = []

    def reset(self):
        self.reset_count += 1

    def act(self, obs, info):
        if "goal_pcd_xyz" in info:
            self.goal_seen.append(info["goal_pcd_xyz"].copy())
        return self.env.action_space.sample()


class FakePlannerPolicy:
    def __init__(self):
        self.set_env_calls = 0
        self.infos = []

    def set_env(self, env):
        self.env = env
        self.set_env_calls += 1

    def get_action(self, info):
        self.infos.append(info)
        return np.ones((1, 2), dtype=np.float32)


def test_native_loop_uses_task_reset_and_records_terminal_flags():
    env = FakeNativeEnv()
    policy = RecordingPolicy(env)
    events = []

    records = evaluate_native(
        env,
        policy,
        task_ids=(1, 2),
        episodes_per_task=1,
        seed_manifest={1: [101], 2: [202]},
        on_reset=lambda **event: events.append(("reset", event["task_id"])),
        on_step=lambda **event: events.append(
            ("step", event["task_id"], event["steps"])
        ),
        on_episode_end=lambda **event: events.append(
            ("end", event["record"]["task_id"])
        ),
    )

    assert [r["task_id"] for r in records] == [1, 2]
    assert [r["seed"] for r in records] == [101, 202]
    assert records[0]["terminated"] is True
    assert records[0]["truncated"] is False
    assert records[1]["terminated"] is False
    assert records[1]["truncated"] is True
    assert policy.reset_count == 2
    assert [call[1]["task_id"] for call in env.reset_calls] == [1, 2]
    assert all(call[1]["render_goal"] is True for call in env.reset_calls)
    assert events == [
        ("reset", 1),
        ("step", 1, 1),
        ("end", 1),
        ("reset", 2),
        ("step", 2, 1),
        ("end", 2),
    ]


def test_summary_reports_per_task_and_mean_success_rate():
    summary = summarize_results(
        [
            {"task_id": 1, "success": True},
            {"task_id": 1, "success": False},
            {"task_id": 2, "success": True},
        ]
    )

    assert summary["task1_sr"] == 0.5
    assert summary["task2_sr"] == 1.0
    assert summary["mean_sr"] == 0.75


def test_native_loop_preserves_goal_observation_for_every_policy_call():
    env = FakeNativeEnv()
    step_count = 0

    def step(action):
        nonlocal step_count
        step_count += 1
        return (
            np.array([env._task_id], dtype=np.float32),
            0.0,
            False,
            step_count >= 3,
            {"success": False},
        )

    env.step = step
    policy = RecordingPolicy(env)

    goal = np.arange(6, dtype=np.float32).reshape(2, 3)
    env.reset = lambda *, seed=None, options=None: (
        np.array([1], dtype=np.float32),
        {"success": False, "goal_pcd_xyz": goal},
    )
    evaluate_native(env, policy, task_ids=(1,), episodes_per_task=1)

    assert len(policy.goal_seen) == 3
    assert all(np.array_equal(seen, goal) for seen in policy.goal_seen)


def test_native_batched_policy_resets_and_batches_current_and_goal_pcd():
    env = FakeNativeEnv()
    planner = FakePlannerPolicy()
    policy = NativeBatchedPolicy(planner, env)
    pcd = np.zeros((1200, 3), dtype=np.float32)
    rgb = np.zeros((1200, 3), dtype=np.uint8)

    policy.reset()
    action = policy.act(
        None,
        {
            "pcd_xyz": pcd,
            "pcd_rgb": rgb,
            "goal_pcd_xyz": pcd,
            "goal_pcd_rgb": rgb,
        },
    )
    policy.reset()

    assert planner.set_env_calls == 2
    assert planner.env.envs == [env]
    assert action.shape == (2,)
    assert planner.infos[0]["pcd_xyz"].shape == (1, 1, 1200, 3)
    assert planner.infos[0]["goal_pcd_xyz"].shape == (1, 1, 1200, 3)
    assert planner.infos[0]["action"].shape == (1, 1, 2)
    assert np.isnan(planner.infos[0]["action"]).all()
    assert planner.infos[0]["step_idx"].tolist() == [[0]]
