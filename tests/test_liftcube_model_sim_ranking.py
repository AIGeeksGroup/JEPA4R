from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from planning.liftcube_goal import LiftCubeCostWeights
from scripts.eval_liftcube_model_sim_ranking import (
    RealTrajectory,
    SimulatorSnapshot,
    compute_real_liftcube_cost,
    summarize_candidate_ranking,
)


class TinyEnv:
    def __init__(self):
        self.unwrapped = self
        self.step_count = 3
        self.obj = SimpleNamespace(pose=SimpleNamespace(p=np.array([0.0, 0.0, 0.02])))
        self.np_random = np.random.default_rng(17)

    def get_state(self):
        return {
            "step": self.step_count,
            "pose": self.obj.pose.p.copy(),
        }

    def set_state(self, state):
        self.step_count = int(state["step"])
        self.obj.pose.p = np.asarray(state["pose"], dtype=np.float64).copy()


def test_snapshot_round_trip_restores_state_and_digest():
    env = TinyEnv()
    snapshot = SimulatorSnapshot.capture(env)
    env.step_count = 99
    env.obj.pose.p[:] = [1.0, 2.0, 3.0]

    snapshot.restore(env)

    assert env.step_count == 3
    np.testing.assert_array_equal(env.obj.pose.p, [0.0, 0.0, 0.02])
    assert snapshot.digest == SimulatorSnapshot.capture(env).digest


def test_candidate_ranking_reports_perfect_order_and_topk_regret():
    result = summarize_candidate_ranking(
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        top_k=(1, 2),
    )

    assert result["valid"] is True
    assert result["spearman"] == pytest.approx(-1.0)
    assert result["kendall_tau_b"] == pytest.approx(-1.0)
    assert result["top_k"]["1"]["overlap"] == 0
    assert result["top_k"]["2"]["overlap"] == 0
    assert result["regret_at_1"] == pytest.approx(0.3)


def test_candidate_ranking_marks_constant_real_cost_low_variance():
    result = summarize_candidate_ranking(
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 1.0, 1.0]),
        top_k=(1,),
    )

    assert result["valid"] is False
    assert result["reason"] == "low_real_cost_variance"
    assert result["spearman"] is None
    assert result["kendall_tau_b"] is None


def test_real_cost_uses_two_chunk_endpoints_and_dense_trace():
    actions = np.zeros((2, 10, 7), dtype=np.float32)
    actions[0, :, -1] = 1.0
    actions[1, :, -1] = -1.0
    trajectory = RealTrajectory(
        actions=actions,
        object_pos=np.array([[0.02, 0.0, 0.02], [0.02, 0.0, 0.02]]),
        tcp_xyz=np.array([[0.02, 0.0, 0.03], [0.02, 0.0, 0.03]]),
        heights=np.full(20, 0.02),
        tcp_object_distances=np.full(20, 0.01),
        gripper_gap=np.full(20, 0.04),
        contact_steps=np.zeros(20, dtype=bool),
        penetrations=np.zeros(20),
    )

    result = compute_real_liftcube_cost(
        trajectory,
        initial_cube_height=0.02,
        lift_threshold=0.22,
        hold_steps=10,
        height_tolerance=0.01,
        approach_radius=0.04,
    )

    assert result["total"] == pytest.approx(0.08)
    assert result["object_pos_endpoint_count"] == 2
    assert result["physical_success"] is False


def test_real_cost_includes_endpoint_gripper_and_robot_contact():
    actions = np.zeros((2, 10, 7), dtype=np.float32)
    actions[..., -1] = -1.0
    trajectory = RealTrajectory(
        actions=actions,
        object_pos=np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.04]]),
        tcp_xyz=np.array([[0.01, 0.0, 0.03], [0.01, 0.0, 0.04]]),
        heights=np.linspace(0.02, 0.04, 20),
        tcp_object_distances=np.full(20, 0.01),
        gripper_gap=np.full(20, 0.04),
        contact_steps=np.zeros(20, dtype=bool),
        penetrations=np.zeros(20),
        cube_robot_contact=np.zeros(20, dtype=bool),
    )
    weights = LiftCubeCostWeights(
        height=0.0,
        hold=0.0,
        stability=0.0,
        latent=0.0,
        approach=0.0,
        early_close=0.0,
        gripper=1.0,
        contact=1.0,
    )

    result = compute_real_liftcube_cost(
        trajectory,
        initial_cube_height=0.02,
        lift_threshold=0.22,
        weights=weights,
    )

    assert result["gripper_cost"] == pytest.approx(0.0004)
    assert result["contact_cost"] == pytest.approx(1.0)
    assert result["total"] == pytest.approx(1.0004)
