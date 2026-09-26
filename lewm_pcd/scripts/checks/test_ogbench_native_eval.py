import numpy as np

from eval_ogbench import evaluate_native, summarize_results


class FakeActionSpace:
    shape = (2,)

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


def test_native_loop_uses_task_reset_and_records_terminal_flags():
    env = FakeNativeEnv()
    policy = RecordingPolicy(env)

    records = evaluate_native(
        env,
        policy,
        task_ids=(1, 2),
        episodes_per_task=1,
        seed_manifest={1: [101], 2: [202]},
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
