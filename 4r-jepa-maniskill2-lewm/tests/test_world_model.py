"""Behavioral checks for the requested latent boundaries and time indexing.

Pure temporal/loss checks run with ``-k 'temporal or loss'`` in the LeWM
environment.  The full file requires the real ManiSkill2-Learn dependencies.
No upstream component is replaced with a test double.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ms2_lewm.model import LeWMLoss, TemporalEncoder, WorldModel
from ms2_lewm.official import lewm_config, lewm_jepa, lewm_module
from ms2_lewm.temporal import TemporalEncoder as IndependentTemporalEncoder, resolve_temporal_config


def test_world_model_module_exposes_our_independent_full_width_temporal_encoder():
    torch.manual_seed(71)
    assert TemporalEncoder is IndependentTemporalEncoder
    temporal = TemporalEncoder(512)
    assert temporal.K == 1
    assert temporal.num_frames == 3
    assert temporal.config == resolve_temporal_config()
    assert len(temporal.layers) == 2
    for layer in temporal.layers:
        assert type(layer) is nn.TransformerEncoderLayer
        assert layer.self_attn.embed_dim == 512
        assert layer.self_attn.num_heads == 8
        assert layer.linear1.in_features == layer.linear2.out_features == 512
    assert not any(isinstance(module, lewm_module.Transformer) for module in temporal.modules())


def test_temporal_k1_view_and_short_initial_history():
    torch.manual_seed(72)
    temporal = TemporalEncoder(512).eval()
    points = torch.randn(2, temporal.num_frames, 512)
    torch.testing.assert_close(temporal(points), temporal(points.unsqueeze(2)))
    assert temporal(points[:, :1]).shape == (2, 512)
    with pytest.raises(ValueError, match="K=1"):
        temporal(points.unsqueeze(2).expand(-1, -1, 2, -1))


def test_loss_is_final_target_mse_with_no_target_stop_gradient():
    torch.manual_seed(73)
    loss_fn = LeWMLoss()
    latent1 = torch.randn(3, loss_fn.history_size + 1, 16, requires_grad=True)
    prediction = torch.randn(3, 16, requires_grad=True)
    output = loss_fn(prediction, latent1)
    torch.testing.assert_close(output["pred_loss"], F.mse_loss(prediction, latent1[:, -1]))
    torch.testing.assert_close(
        output["loss"], output["pred_loss"] + lewm_config().loss.sigreg.weight * output["sigreg_loss"]
    )
    output["pred_loss"].backward()
    torch.testing.assert_close(latent1.grad[:, :-1], torch.zeros_like(latent1.grad[:, :-1]))
    assert latent1.grad[:, -1].abs().sum() > 0
    assert prediction.grad.abs().sum() > 0


def test_loss_sigreg_is_per_frame_across_batch_including_the_target():
    torch.manual_seed(74)
    loss_fn = LeWMLoss()
    assert type(loss_fn.sigreg) is lewm_module.SIGReg
    config = lewm_config()
    assert loss_fn.sigreg.num_proj == config.loss.sigreg.kwargs.num_proj
    assert loss_fn.sigreg.t.numel() == config.loss.sigreg.kwargs.knots
    latent1 = torch.randn(4, loss_fn.history_size + 1, 16, requires_grad=True)
    # Give frames different means so treating time as extra batch samples
    # changes the statistic and is detectable.
    latent1 = latent1 + torch.arange(latent1.shape[1])[None, :, None] * 2.0
    latent1.retain_grad()
    random_state = torch.get_rng_state()
    output = loss_fn(torch.zeros(4, 16), latent1)
    per_frame = []
    for t in range(latent1.shape[1]):
        torch.set_rng_state(random_state)
        per_frame.append(loss_fn.sigreg(latent1[:, t].unsqueeze(0)))
    torch.testing.assert_close(output["sigreg_loss"], torch.stack(per_frame).mean())
    torch.set_rng_state(random_state)
    merged_time_batch = loss_fn.sigreg(latent1.reshape(1, -1, latent1.shape[-1]))
    assert not torch.isclose(output["sigreg_loss"], merged_time_batch)
    output["sigreg_loss"].backward()
    assert all(latent1.grad[:, t].abs().sum() > 0 for t in range(latent1.shape[1]))


@pytest.fixture
def world_model():
    torch.manual_seed(75)
    return WorldModel(action_dim=7).eval()


def _training_input(model, batch=2, points=24):
    return {
        "pointcloud": torch.randn(batch, model.history_size + 1, points, 6),
        "action": torch.randn(batch, model.history_size + 1, model.action_dim),
    }


def test_world_forward_uses_only_context_and_the_departure_action(world_model):
    model = world_model
    info = _training_input(model)
    original = model(info)
    modified = {key: value.clone() for key, value in info.items()}
    modified["pointcloud"][:, -1] += 20.0
    # Earlier actions never enter the action-free time encoder; extra action
    # at T+1 must not be mistaken for the action departing context frame T.
    modified["action"][:, : model.history_size - 1] += 30.0
    modified["action"][:, model.history_size :] += 40.0
    changed_future = model(modified)
    torch.testing.assert_close(original["prediction"], changed_future["prediction"], rtol=0, atol=0)
    torch.testing.assert_close(original["latent2"], changed_future["latent2"], rtol=0, atol=0)
    assert not torch.allclose(original["target"], changed_future["target"])

    modified["action"][:, model.history_size - 1] += 50.0
    changed_action = model(modified)
    assert not torch.allclose(original["prediction"], changed_action["prediction"])
    assert original["latent1"].shape == (2, model.history_size + 1, 512)
    assert original["latent2"].shape == original["prediction"].shape == (2, 512)


def test_world_prediction_mse_reaches_the_shared_target_encoder(world_model):
    model = world_model
    output = model(_training_input(model))
    latent1 = output["latent1"]
    latent1.retain_grad()
    output["pred_loss"].backward()
    assert latent1.grad[:, -1].abs().sum() > 0
    assert all(latent1.grad[:, t].abs().sum() > 0 for t in range(model.history_size))
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.frame_encoder.parameters())
    expected_target_gradient = 2.0 * (output["target"] - output["prediction"]) / output["prediction"].numel()
    torch.testing.assert_close(latent1.grad[:, -1], expected_target_gradient)


@pytest.mark.parametrize("initial_history", [1, 3])
def test_world_rollout_matches_official_algorithm_and_manual_action_alignment(world_model, initial_history):
    model = world_model
    batch, samples, horizon = 2, 2, 5
    # The official algorithm assumes common observation history across candidates.
    context = torch.randn(batch, 1, initial_history, 24, 6).expand(-1, samples, -1, -1, -1)
    actions = torch.randn(batch, samples, horizon, model.action_dim)
    rollout = model.rollout({"pointcloud": context}, actions)
    upstream = lewm_jepa.JEPA.rollout(
        model, {"pixels": context}, actions, history_size=model.history_size
    )
    torch.testing.assert_close(rollout["predicted_emb"], upstream["predicted_emb"])

    encoded = model.frame_encoder(context[:, 0])
    manual = encoded[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, initial_history, 512)
    for t in range(initial_history - 1, horizon):
        latent2 = model.temporal_encoder(manual[:, -model.history_size :])
        action_t = actions[:, :, t].reshape(batch * samples, model.action_dim)
        prediction = model.predictor(latent2, action_t)
        manual = torch.cat((manual, prediction[:, None]), dim=1)
    manual = manual.reshape(batch, samples, horizon + 1, 512)
    assert rollout["predicted_emb"].shape == (batch, samples, horizon + 1, 512)
    torch.testing.assert_close(rollout["predicted_emb"], manual)


def test_world_planning_cost_keeps_official_final_state_latent_mse(world_model):
    model = world_model
    batch, samples, horizon = 2, 3, 4
    context = torch.randn(batch, 1, 1, 24, 6).expand(-1, samples, -1, -1, -1)
    goal = torch.randn(batch, 1, 1, 24, 6).expand(-1, samples, -1, -1, -1)
    actions = torch.randn(batch, samples, horizon, model.action_dim)
    with torch.no_grad():
        actual = model.get_cost({"pointcloud": context, "goal": goal}, actions)
        prediction = model.rollout({"pointcloud": context}, actions)["predicted_emb"]
        goal_latent = model.frame_encoder(goal[:, 0])[:, -1]
        expected = (prediction[:, :, -1] - goal_latent[:, None]).square().sum(-1)
    assert actual.shape == (batch, samples)
    torch.testing.assert_close(actual, expected)


def test_flat_storage_forward_matches_shaped_1200_point_clouds(world_model):
    model = world_model
    info = _training_input(model, batch=1, points=1200)
    flat = {"pointcloud": info["pointcloud"].flatten(-2), "action": info["action"]}
    assert flat["pointcloud"].shape == (1, model.history_size + 1, 7200)
    random_state = torch.get_rng_state()
    with torch.no_grad():
        shaped_output = model(info)
        torch.set_rng_state(random_state)
        flat_output = model(flat)
    for key in ("latent1", "latent2", "prediction", "target", "pred_loss", "sigreg_loss", "loss"):
        torch.testing.assert_close(flat_output[key], shaped_output[key], rtol=0, atol=0)


def test_flat_storage_encode_matches_shaped_1200_point_clouds(world_model):
    model = world_model
    points = torch.randn(2, model.history_size, 1200, 6)
    with torch.no_grad():
        shaped_output = model.encode({"pointcloud": points})
        flat_output = model.encode({"pointcloud": points.flatten(-2)})
        direct_flat_output = model.encode(points.flatten(-2))
    for key in ("emb", "latent1", "latent2"):
        torch.testing.assert_close(flat_output[key], shaped_output[key], rtol=0, atol=0)
        torch.testing.assert_close(direct_flat_output[key], shaped_output[key], rtol=0, atol=0)


def test_flat_storage_rollout_matches_shaped_1200_point_clouds(world_model):
    model = world_model
    points = torch.randn(1, 1, 1, 1200, 6).expand(-1, 2, -1, -1, -1)
    actions = torch.randn(1, 2, 3, model.action_dim)
    with torch.no_grad():
        shaped_output = model.rollout({"pointcloud": points}, actions)
        flat_output = model.rollout({"pointcloud": points.flatten(-2)}, actions)
    torch.testing.assert_close(flat_output["predicted_emb"], shaped_output["predicted_emb"], rtol=0, atol=0)


def test_flat_storage_get_cost_matches_shaped_and_prefers_goal_pointcloud(world_model):
    model = world_model
    points = torch.randn(1, 1, 1, 1200, 6).expand(-1, 2, -1, -1, -1)
    goal_points = torch.randn(1, 1, 1, 1200, 6).expand(-1, 2, -1, -1, -1)
    rgb_goal = torch.randint(0, 256, (1, 2, 1, 3, 8, 8), dtype=torch.uint8)
    actions = torch.randn(1, 2, 3, model.action_dim)
    with torch.no_grad():
        shaped = model.get_cost({"pointcloud": points, "goal": goal_points}, actions)
        flat = model.get_cost(
            {"pointcloud": points.flatten(-2), "goal": goal_points.flatten(-2)}, actions
        )
        explicit_point_goal = model.get_cost(
            {"pointcloud": points, "goal_pointcloud": goal_points, "goal": rgb_goal}, actions
        )
        explicit_flat_goal = model.get_cost(
            {"pointcloud": points.flatten(-2), "goal_pointcloud": goal_points.flatten(-2), "goal": rgb_goal},
            actions,
        )
    for output in (flat, explicit_point_goal, explicit_flat_goal):
        torch.testing.assert_close(output, shaped, rtol=0, atol=0)
