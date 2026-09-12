from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from planning.liftcube_goal import (
    GoalSpec,
    LiftCubeCostWeights,
    liftcube_trajectory_cost,
)
from planning.latent_cem import (
    AuxiliaryRollout,
    CEMConfig,
    LatentCEMPlanner,
    latent_trajectory_cost,
    normalized_latent_mse,
    rollout_from_latent,
    rollout_with_aux,
)


class FakeModel:
    def __init__(self, *, chunk_size: int = 2):
        self.cfg = SimpleNamespace(T=3, K=1, D=2, skip_frame=chunk_size)
        self.seen_actions: list[torch.Tensor] = []

    def _predict_sequence(self, latent, actions, env):
        assert env == "maniskill2"
        self.seen_actions.append(actions.detach().cpu().clone())
        batch = latent.shape[0]
        chunks = actions.reshape(batch, self.cfg.T, self.cfg.skip_frame, -1)
        arm_delta = chunks[..., 0].mean(dim=2)
        prediction = latent.clone()
        prediction[..., 1] += arm_delta.unsqueeze(-1)
        return prediction, latent


def _history() -> torch.Tensor:
    history = torch.zeros(1, 3, 1, 2)
    history[..., 0] = 1.0
    return history


def test_normalized_latent_mse_is_scale_invariant() -> None:
    prediction = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    goal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    torch.testing.assert_close(normalized_latent_mse(prediction, goal), torch.zeros(1))


def test_rollout_shifts_latent_and_atomic_action_history() -> None:
    model = FakeModel()
    context_actions = torch.tensor(
        [[[0.1, 1.0], [0.2, 1.0], [0.3, -1.0], [0.4, -1.0]]]
    )
    candidates = torch.tensor(
        [
            [
                [[0.5, 1.0], [0.7, 1.0]],
                [[0.9, -1.0], [1.0, -1.0]],
            ],
            [
                [[-0.5, 1.0], [-0.7, 1.0]],
                [[-0.9, -1.0], [-1.0, -1.0]],
            ],
        ]
    )

    rollout = rollout_from_latent(
        model, _history(), context_actions, candidates, "maniskill2"
    )

    assert rollout.shape == (2, 2, 1, 2)
    torch.testing.assert_close(rollout[:, 0, 0, 1], torch.tensor([0.6, -0.6]))
    torch.testing.assert_close(rollout[:, 1, 0, 1], torch.tensor([1.55, -1.55]))
    first, second = model.seen_actions
    torch.testing.assert_close(first[:, :4], context_actions.expand(2, -1, -1))
    torch.testing.assert_close(second[:, :2], context_actions[:, 2:].expand(2, -1, -1))
    torch.testing.assert_close(second[:, 2:4], candidates[:, 0])


def test_cem_plan_is_seeded_bounded_binary_and_reduces_goal_cost() -> None:
    config = CEMConfig(
        horizon=2,
        chunk_size=2,
        action_dim=2,
        candidates=256,
        iterations=5,
        elites=24,
        min_std=0.03,
    )
    model = FakeModel(chunk_size=2)
    history = _history()
    context_actions = torch.zeros(1, 4, 2)
    goal = torch.tensor([[[0.0, 1.0]]])
    action_mean = torch.tensor([0.0, -0.1])
    action_std = torch.tensor([0.5, 0.99])
    planner = LatentCEMPlanner(config, action_mean, action_std)

    first = planner.plan(
        model, history, context_actions, goal, env="maniskill2", seed=17
    )
    second = planner.plan(
        model, history, context_actions, goal, env="maniskill2", seed=17
    )

    assert first.shape == (2, 2, 2)
    torch.testing.assert_close(first, second)
    assert torch.all(first >= -1.0) and torch.all(first <= 1.0)
    assert torch.all((first[..., -1] == -1) | (first[..., -1] == 1))
    assert torch.all(first[:, :, -1] == first[:, :1, -1])
    zero = torch.zeros_like(first)
    zero[..., -1] = 1.0
    planned_cost = latent_trajectory_cost(
        rollout_from_latent(
            model, history, context_actions, first.unsqueeze(0), "maniskill2"
        ),
        goal,
        trajectory_weight=config.trajectory_weight,
    )
    zero_cost = latent_trajectory_cost(
        rollout_from_latent(
            model, history, context_actions, zero.unsqueeze(0), "maniskill2"
        ),
        goal,
        trajectory_weight=config.trajectory_weight,
    )
    assert planned_cost.item() < zero_cost.item()
    assert planner.last_cost == planned_cost.item()


def test_cem_microbatches_population_before_model_forward() -> None:
    config = CEMConfig(
        horizon=2,
        chunk_size=2,
        action_dim=2,
        candidates=128,
        iterations=1,
        elites=16,
        model_batch_size=32,
    )
    model = FakeModel(chunk_size=2)
    planner = LatentCEMPlanner(config, torch.zeros(2), torch.ones(2))

    planner.plan(
        model,
        _history(),
        torch.zeros(1, 4, 2),
        torch.tensor([[[0.0, 1.0]]]),
        env="maniskill2",
        seed=17,
    )

    batches = [actions.shape[0] for actions in model.seen_actions]
    assert max(batches) <= 32
    assert 32 in batches


def test_cem_rejects_nonpositive_model_batch_size() -> None:
    with pytest.raises(ValueError, match="model_batch_size"):
        CEMConfig(model_batch_size=0)


class RecordingAuxModel:
    def __init__(self):
        self.cfg = SimpleNamespace(
            T=3,
            K=1,
            D=2,
            skip_frame=2,
            proprio=SimpleNamespace(dim=25),
        )
        self.proprio_inputs: list[torch.Tensor] = []
        self.calls = 0

    def predict_transition(
        self, latent_history, proprio_history_normalized, actions, env=None
    ):
        assert env == "maniskill2"
        self.proprio_inputs.append(proprio_history_normalized.detach().clone())
        batch = latent_history.shape[0]
        next_value = 7.0 if self.calls == 0 else 9.0
        self.calls += 1
        next_normalized = torch.full(
            (batch, 25),
            next_value,
            dtype=latent_history.dtype,
            device=latent_history.device,
        )
        return SimpleNamespace(
            scene_latent=latent_history[:, -1] + 1.0,
            next_proprio_normalized=next_normalized,
            next_proprio=next_normalized + 100.0,
            object_pos=torch.stack(
                [
                    torch.zeros(batch, device=latent_history.device),
                    torch.zeros(batch, device=latent_history.device),
                    torch.full(
                        (batch,), 0.23, device=latent_history.device
                    ),
                ],
                dim=-1,
            ),
            contact_probability=torch.full(
                (batch,),
                0.2 if self.calls == 1 else 0.8,
                device=latent_history.device,
            ),
        )


def test_rollout_second_step_uses_first_predicted_proprio() -> None:
    model = RecordingAuxModel()
    initial = torch.zeros(1, 3, 25)
    candidates = torch.zeros(1, 2, 2, 2)

    result = rollout_with_aux(
        model,
        _history(),
        initial,
        torch.zeros(1, 4, 2),
        candidates,
        "maniskill2",
    )

    torch.testing.assert_close(
        model.proprio_inputs[1][:, -1], torch.full((1, 25), 7.0)
    )
    torch.testing.assert_close(
        result.proprio_normalized[:, 1], torch.full((1, 25), 9.0)
    )
    torch.testing.assert_close(result.proprio[:, 1], torch.full((1, 25), 109.0))
    assert result.scene_latent.shape == (1, 2, 1, 2)
    assert result.object_pos.shape == (1, 2, 3)
    torch.testing.assert_close(
        result.contact_probability, torch.tensor([[0.2, 0.8]])
    )


def _aux_rollout(
    heights, *, tcp_distance=0.0, gripper_gap=0.0, contact_probability=None
) -> AuxiliaryRollout:
    height = torch.as_tensor(heights, dtype=torch.float32).view(1, -1)
    horizon = height.shape[1]
    proprio = torch.zeros(1, horizon, 25)
    object_pos = torch.zeros(1, horizon, 3)
    object_pos[..., 2] = height
    proprio[..., 18:21] = object_pos
    proprio[..., 18] += float(tcp_distance)
    proprio[..., 7:9] = float(gripper_gap) / 2.0
    contact = (
        None
        if contact_probability is None
        else torch.as_tensor(contact_probability, dtype=torch.float32).view(1, horizon)
    )
    return AuxiliaryRollout(
        scene_latent=torch.ones(1, horizon, 1, 2),
        proprio_normalized=torch.zeros(1, horizon, 25),
        proprio=proprio,
        object_pos=object_pos,
        contact_probability=contact,
    )


def _candidate_actions(horizon=2, *, gripper=1.0) -> torch.Tensor:
    actions = torch.zeros(1, horizon, 10, 7)
    actions[..., -1] = gripper
    return actions


def _goal(**overrides) -> GoalSpec:
    values = {
        "initial_cube_height": 0.02,
        "lift_threshold": 0.22,
        "hold_control_steps": 10,
        "height_tolerance": 0.01,
    }
    values.update(overrides)
    return GoalSpec(**values)


def test_below_threshold_has_height_penalty() -> None:
    cost = liftcube_trajectory_cost(
        _aux_rollout([0.10, 0.15]),
        _candidate_actions(),
        _goal(),
        chunk_size=10,
    )

    assert cost.components["height"].item() > 0


def test_one_peak_then_fall_fails_hold_and_costs_more_than_stable_hold() -> None:
    transient = liftcube_trajectory_cost(
        _aux_rollout([0.24, 0.10]),
        _candidate_actions(),
        _goal(),
        chunk_size=10,
    )
    stable = liftcube_trajectory_cost(
        _aux_rollout([0.23, 0.23]),
        _candidate_actions(),
        _goal(),
        chunk_size=10,
    )

    assert transient.components["hold"].item() > 0
    assert transient.components["stability"].item() > 0
    assert transient.total.item() > stable.total.item()


def test_two_chunk_boundaries_encode_ten_control_step_hold() -> None:
    result = liftcube_trajectory_cost(
        _aux_rollout([0.23, 0.23]),
        _candidate_actions(),
        _goal(hold_control_steps=10),
        chunk_size=10,
    )

    assert result.components["hold"].item() == 0
    assert result.components["stability"].item() == 0


def test_approach_and_early_close_only_shape_pre_lift_nodes() -> None:
    weights = LiftCubeCostWeights(
        height=0.0,
        hold=0.0,
        stability=0.0,
        latent=0.0,
        approach=1.0,
        early_close=1.0,
    )
    pre_lift_open = liftcube_trajectory_cost(
        _aux_rollout([0.10, 0.10], tcp_distance=0.20),
        _candidate_actions(gripper=1.0),
        _goal(weights=weights),
        chunk_size=10,
    )
    pre_lift_closed = liftcube_trajectory_cost(
        _aux_rollout([0.10, 0.10], tcp_distance=0.20),
        _candidate_actions(gripper=-1.0),
        _goal(weights=weights),
        chunk_size=10,
    )
    lifted_closed = liftcube_trajectory_cost(
        _aux_rollout([0.23, 0.23], tcp_distance=0.20),
        _candidate_actions(gripper=-1.0),
        _goal(weights=weights),
        chunk_size=10,
    )

    assert pre_lift_open.components["approach"].item() > 0
    assert pre_lift_open.components["early_close"].item() == 0
    assert pre_lift_closed.components["early_close"].item() > 0
    assert lifted_closed.components["approach"].item() == 0
    assert lifted_closed.components["early_close"].item() == 0


def test_gripper_and_contact_cost_use_predicted_physical_state() -> None:
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
    bad = liftcube_trajectory_cost(
        _aux_rollout(
            [0.03, 0.04],
            tcp_distance=0.01,
            gripper_gap=0.04,
            contact_probability=[0.1, 0.1],
        ),
        _candidate_actions(gripper=-1.0),
        _goal(weights=weights),
        chunk_size=10,
    )
    good = liftcube_trajectory_cost(
        _aux_rollout(
            [0.03, 0.04],
            tcp_distance=0.01,
            gripper_gap=0.0,
            contact_probability=[1.0, 1.0],
        ),
        _candidate_actions(gripper=-1.0),
        _goal(weights=weights),
        chunk_size=10,
    )

    assert bad.components["gripper"].item() > 0
    assert bad.components["contact"].item() > 0
    assert good.components["gripper"].item() == 0
    assert good.components["contact"].item() == 0
    assert bad.total.item() > good.total.item()


def test_contact_weight_requires_contact_prediction() -> None:
    weights = LiftCubeCostWeights(contact=1.0)
    with pytest.raises(ValueError, match="contact_probability"):
        liftcube_trajectory_cost(
            _aux_rollout([0.03, 0.04], contact_probability=None),
            _candidate_actions(),
            _goal(weights=weights),
            chunk_size=10,
        )


def test_nan_empty_and_short_rollouts_are_handled_explicitly() -> None:
    with pytest.raises(ValueError, match="horizon"):
        liftcube_trajectory_cost(
            _aux_rollout([]),
            _candidate_actions(horizon=0),
            _goal(),
            chunk_size=10,
        )
    with pytest.raises(ValueError, match="hold"):
        liftcube_trajectory_cost(
            _aux_rollout([0.23]),
            _candidate_actions(horizon=1),
            _goal(),
            chunk_size=10,
        )
    result = liftcube_trajectory_cost(
        _aux_rollout([0.23, float("nan")]),
        _candidate_actions(),
        _goal(),
        chunk_size=10,
    )
    assert torch.isinf(result.total).all()


def test_goal_aware_cem_uses_structured_rollout_and_microbatches() -> None:
    class GoalModel(RecordingAuxModel):
        def predict_transition(
            self, latent_history, proprio_history_normalized, actions, env=None
        ):
            result = super().predict_transition(
                latent_history, proprio_history_normalized, actions, env=env
            )
            result.object_pos[:, 2] = 0.23
            return result

    config = CEMConfig(
        horizon=2,
        chunk_size=2,
        action_dim=2,
        candidates=8,
        iterations=1,
        elites=2,
        model_batch_size=3,
    )
    model = GoalModel()
    planner = LatentCEMPlanner(config, torch.zeros(2), torch.ones(2))

    plan = planner.plan_with_goal(
        model,
        _history(),
        torch.zeros(1, 3, 25),
        torch.zeros(1, 4, 2),
        _goal(hold_control_steps=2),
        env="maniskill2",
        seed=19,
    )

    assert plan.shape == (2, 2, 2)
    assert max(value.shape[0] for value in model.proprio_inputs) <= 3
    assert torch.isfinite(torch.tensor(planner.last_cost))
    assert set(planner.last_cost_components) == {
        "height",
        "hold",
        "stability",
        "latent",
            "approach",
            "early_close",
            "gripper",
            "contact",
        }
