from types import SimpleNamespace

import numpy as np
import pytest

import envs.fixed_task_expert_policy as policy_module
from envs.fixed_task_expert_policy import FixedTaskCubePlanPolicy


class FakeActionSpace:
    shape = (1, 5)


class FakeCubeEnv:
    def __init__(self, family="double", task_id=4, mode="task"):
        self.unwrapped = self
        self._env_type = family
        self._mode = mode
        self.cur_task_id = task_id
        self.targets = []

    def set_target_pos(self, block_id, target_xyz):
        self.targets.append((block_id, np.asarray(target_xyz).copy()))


class FakeVectorEnv:
    def __init__(self, single):
        self.envs = [single]
        self.num_envs = 1
        self.action_space = FakeActionSpace()


class FakeOracle:
    instances = []

    def __init__(self, env, noise, noise_smoothing):
        self.done = False
        self.resets = []
        FakeOracle.instances.append(self)

    def reset(self, _ob, info):
        self.done = False
        self.resets.append(info)

    def select_action(self, _ob, _info):
        self.done = True
        return np.full(5, len(FakeOracle.instances), dtype=np.float32)


def make_info(step_idx):
    values = {
        "step_idx": np.array([[step_idx]]),
        "time": np.array([[step_idx * 0.05]]),
        "proprio/effector_pos": np.zeros((1, 1, 3)),
        "proprio/effector_yaw": np.zeros((1, 1, 1)),
        "proprio/gripper_opening": np.zeros((1, 1, 1)),
    }
    for block_id in range(3):
        values[f"privileged/block_{block_id}_pos"] = np.zeros((1, 1, 3))
        values[f"privileged/block_{block_id}_yaw"] = np.zeros((1, 1, 1))
    return values


def test_swap_policy_executes_buffer_then_dependency_order(monkeypatch):
    FakeOracle.instances.clear()
    monkeypatch.setattr(policy_module, "CubePlanOracle", FakeOracle)
    single = FakeCubeEnv("double", 4)
    policy = FixedTaskCubePlanPolicy("double", 4)
    policy.set_env(FakeVectorEnv(single))

    for step in range(3):
        action = policy.get_action(make_info(step))
        assert action.shape == (1, 5)

    blocks = [block for block, _ in single.targets]
    assert blocks == [0, 1, 0]
    np.testing.assert_allclose(single.targets[-1][1], [0.425, 0.1, 0.02])
    assert policy.finished.tolist() == [True]
    np.testing.assert_array_equal(policy.get_action(make_info(4)), np.zeros((1, 5)))


def test_oracle_receives_synthesized_single_cube_target_info(monkeypatch):
    FakeOracle.instances.clear()
    monkeypatch.setattr(policy_module, "CubePlanOracle", FakeOracle)
    single = FakeCubeEnv("triple", 3)
    policy = FixedTaskCubePlanPolicy("triple", 3)
    policy.set_env(FakeVectorEnv(single))
    policy.get_action(make_info(0))

    reset_info = FakeOracle.instances[0].resets[0]
    assert reset_info["privileged/target_task"] == "cube"
    assert reset_info["privileged/target_block"] == 2
    np.testing.assert_allclose(reset_info["privileged/target_block_pos"], [0.50, 0.0, 0.02])
    np.testing.assert_allclose(reset_info["privileged/target_block_yaw"], [0.0])


@pytest.mark.parametrize(
    "single,match",
    [
        (FakeCubeEnv("single", 1), "env_type"),
        (FakeCubeEnv("double", 4, mode="data_collection"), "task mode"),
        (FakeCubeEnv("double", 2), "task_id"),
    ],
)
def test_policy_rejects_wrong_environment_contract(single, match):
    policy = FixedTaskCubePlanPolicy("double", 4)
    with pytest.raises(ValueError, match=match):
        policy.set_env(FakeVectorEnv(single))
