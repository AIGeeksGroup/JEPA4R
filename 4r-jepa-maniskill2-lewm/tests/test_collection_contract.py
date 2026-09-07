"""Meaningful checks for official noise equality and the zero-success gate."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from ms2_lewm.expert import ExpertPolicy, ManiSkillNoisyExpert, gripper_target_action, official_noise_config


class ConstantOracle:
    done = False

    def reset(self, observation, info):
        pass

    def select_action(self, observation, info):
        return np.array([0.15, -0.2, 0.1, 0.3, -0.7], dtype=np.float32)


def test_noisy_expert_matches_upstream_actions_and_rng_exactly():
    pose = SimpleNamespace(to_transformation_matrix=lambda: np.eye(4))
    robot = SimpleNamespace(get_qpos=lambda: np.array([0.04, 0.04]))
    bounds = gym.spaces.Box(np.array([-0.01], dtype=np.float32), np.array([0.04], dtype=np.float32))
    controller = SimpleNamespace(controllers={"gripper": SimpleNamespace(_action_space=bounds)})
    raw = SimpleNamespace(_env_type="single", simulator=SimpleNamespace(
        tcp=SimpleNamespace(pose=pose), agent=SimpleNamespace(robot=robot, controller=controller)
    ))
    pool = SimpleNamespace(
        num_envs=1,
        envs=[SimpleNamespace(unwrapped=raw)],
        action_space=gym.spaces.Box(-1.0, 1.0, (1, 7), np.float32),
    )
    adapted = ManiSkillNoisyExpert(seed=3072)
    adapted.set_env(pool)
    adapted._oracle_agents = {"cube": [ConstantOracle()]}
    original = ExpertPolicy(seed=3072)
    original.env = adapted.env
    original._p_stack = np.zeros(1)
    original._xi = np.zeros(1)
    original._agents = [None]
    original._oracle_agents = {"cube": [ConstantOracle()]}
    for step in list(range(20)) + list(range(20)):
        info = {
            "step_idx": np.array([[step]]),
            "privileged/target_task": np.array([["cube"]]),
        }
        expected = original.get_action(info)
        actual = adapted.get_action(info)
        np.testing.assert_array_equal(adapted.last_canonical_actions, expected)
        np.testing.assert_allclose(actual[:, :3], expected[:, :3] * 0.5, rtol=1e-6)
        np.testing.assert_allclose(actual[:, 5], np.clip(expected[:, 3] * 3.0, -1, 1), rtol=1e-6)
        np.testing.assert_allclose(actual[:, 6], 1.0 - 2.0 * np.clip(expected[:, 4], 0, 1), rtol=1e-6)
        np.testing.assert_array_equal(actual[:, [3, 4]], 0)
        assert original.rng.bit_generator.state == adapted.rng.bit_generator.state
    assert official_noise_config() == {
        "policy_type": "markov_oracle", "action_noise": 0.1,
        "p_random_action": 0.0, "noise_smoothing": 0.5, "min_norm": 0.4,
    }


def test_zero_success_preflight_never_starts_collection(monkeypatch, tmp_path):
    import ms2_lewm.environment as environment
    import ms2_lewm.expert as expert

    constructed = []

    class FakeWorld:
        def set_policy(self, policy):
            pass

        def evaluate(self, **kwargs):
            return {"episode_successes": np.zeros(kwargs["episodes"]), "success_rate": 0.0}

        def collect(self, **kwargs):
            pytest.fail("Collector was reached despite zero successes")

        def close(self):
            pass

    def make_world(**kwargs):
        constructed.append(kwargs)
        return FakeWorld()

    monkeypatch.setattr(environment, "make_world", make_world)
    monkeypatch.setattr(expert, "ManiSkillNoisyExpert", lambda: SimpleNamespace(
        rng=np.random.default_rng(3072), episode_noise_scales=[0.01, 0.02, 0.03]
    ))
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("collect_wm_under_test", root / "scripts/collect_wm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "forbidden.h5"
    monkeypatch.setattr(sys, "argv", ["collect_wm.py", "--output", str(output), "--smoke", "--reserve-gib", "0"])
    with pytest.raises(RuntimeError, match="ZERO preflight successes"):
        module.main()
    assert len(constructed) == 1  # Only the small preflight world exists.
    assert not output.exists()
    assert not output.with_suffix(".lance").exists()
    assert output.with_suffix(".h5.preflight.json").exists()


def test_busy_gpu_is_rejected_before_any_renderer(monkeypatch, tmp_path):
    import ms2_lewm.environment as environment
    import stable_worldmodel as swm

    monkeypatch.setenv("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    monkeypatch.setattr(environment, "TASK_ROOT", tmp_path)
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="0, 90, 31856, 256\n"
    ))
    monkeypatch.setattr(swm, "World", lambda **kwargs: pytest.fail("GPU renderer construction was reached"))
    with pytest.raises(RuntimeError, match="protect existing tasks"):
        environment.make_world(num_envs=1, max_episode_steps=200)
    import json
    report = json.loads((tmp_path / "artifacts/last_renderer_resource_check.json").read_text())
    assert report["devices"][0]["free_mib"] == 256
    assert report["required_free_mib"] == 2048
    assert report["passed"] is False


@pytest.mark.parametrize("qpos", [-0.01, 0.0, 0.015, 0.02, 0.04])
def test_gripper_zero_delta_preserves_native_joint_position(qpos):
    lower, upper = -0.01, 0.04
    normalized = gripper_target_action(qpos, 0.0, lower, upper)
    restored = lower + 0.5 * (normalized + 1.0) * (upper - lower)
    assert restored == pytest.approx(qpos)
    assert gripper_target_action(qpos, 1.0, lower, upper) == pytest.approx(-1.0)
    assert gripper_target_action(qpos, -1.0, lower, upper) == pytest.approx(1.0)
