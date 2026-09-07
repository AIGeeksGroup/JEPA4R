"""The settling adapter preserves upstream transit and noisy-action semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from ms2_lewm.expert import ManiSkillNoisyExpert
from ms2_lewm.pickcube_oracle import CubeMarkovOracle, PickCubeHoldOracle


def make_oracle(cls=PickCubeHoldOracle, **kwargs):
    env = SimpleNamespace(_arm_sampling_bounds=np.array([[-0.1, -0.2, 0.2], [0.2, 0.2, 0.35]]))
    env.unwrapped = env
    oracle = cls(env=env, min_norm=0.4, **kwargs)
    oracle.reset(None, {})
    return oracle


def info(*, distance=0.01, grasped=True):
    return {
        "privileged/target_block": np.int64(0),
        "privileged/block_0_pos": np.array([0.0, 0.0, 0.2]),
        "privileged/block_0_yaw": np.array([0.0]),
        "privileged/target_block_pos": np.array([distance, 0.0, 0.2]),
        "privileged/target_block_yaw": np.array([0.0]),
        "proprio/effector_pos": np.array([0.0, 0.0, 0.2]),
        "proprio/effector_yaw": np.array([0.0]),
        "proprio/gripper_opening": np.array([0.5]),
        "proprio/gripper_contact": np.array([float(grasped)]),
    }


@pytest.mark.parametrize("distance,grasped", [(0.03, True), (0.4, True), (0.03, False), (0.2, False)])
def test_approach_grasp_transport_actions_remain_upstream(distance, grasped):
    upstream, adapted = make_oracle(CubeMarkovOracle), make_oracle()
    adapted._final_pos = upstream._final_pos.copy()
    adapted._final_yaw = upstream._final_yaw
    state = info(distance=distance, grasped=grasped)
    np.testing.assert_array_equal(adapted.select_action(None, state), upstream.select_action(None, state))


def test_short_hold_closes_gripper_without_minimum_norm_boost_then_releases():
    oracle = make_oracle(hold_steps=3)
    phases = []
    oracle.print_phase = phases.append
    for _ in range(3):
        np.testing.assert_allclose(oracle.select_action(None, info(distance=0.001)), [0.005, 0, 0, 0, 1])
    assert phases == ["6b: Hold at PickCube target"] * 3
    assert oracle._step == 3
    released = oracle.select_action(None, info(distance=0.001))
    assert released[4] == -1
    assert phases[-1] == "7: Release"
    oracle.reset(None, {})
    assert oracle.select_action(None, info(distance=0.001))[4] == 1


def test_drop_during_hold_returns_to_original_recovery():
    oracle = make_oracle()
    oracle.select_action(None, info())
    upstream = make_oracle(CubeMarkovOracle)
    oracle._final_pos = upstream._final_pos.copy()
    oracle._final_yaw = upstream._final_yaw
    displaced = info(distance=0.1, grasped=False)
    np.testing.assert_array_equal(oracle.select_action(None, displaced), upstream.select_action(None, displaced))
    assert oracle._hold_remaining == 0
    assert not oracle._hold_completed


def test_hold_preserves_subtask_timeout():
    oracle = make_oracle()
    oracle._step = oracle._max_step - 1
    oracle.select_action(None, info())
    assert oracle.done


def test_explicit_legacy_oracle_is_exact_upstream_class():
    raw = SimpleNamespace(unwrapped=None)
    raw.unwrapped = raw
    pool = SimpleNamespace(envs=[raw], num_envs=1)
    legacy = ManiSkillNoisyExpert(seed=3072, oracle_mode="legacy")
    legacy.set_env(pool)
    assert type(legacy._oracle_agents["cube"][0]) is CubeMarkovOracle
    adapted = ManiSkillNoisyExpert(seed=3072)
    adapted.set_env(pool)
    assert type(adapted._oracle_agents["cube"][0]) is PickCubeHoldOracle
    assert legacy.rng.bit_generator.state == adapted.rng.bit_generator.state


def test_unknown_oracle_mode_is_rejected():
    with pytest.raises(ValueError, match="oracle_mode"):
        ManiSkillNoisyExpert(oracle_mode="noiseless")
