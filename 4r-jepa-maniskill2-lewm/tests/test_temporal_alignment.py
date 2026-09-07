"""Consecutive WM data/actions and physically equivalent MPC planning units.

Only an explicitly synthetic timestamp fixture is written. These tests use
the real official HDF5 reader/writer and the real model, without collection,
simulation, rendering, or a GPU.
"""
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from ms2_lewm.model import WorldModel
from ms2_lewm.official import lewm_config, load_source
from stable_worldmodel.data import load_dataset
from stable_worldmodel.data.formats.hdf5 import HDF5Writer


ROOT = Path(__file__).resolve().parents[1]
train_driver = load_source("_ms2_local_train_alignment", ROOT / "scripts/train_wm.py")
eval_driver = load_source("_ms2_local_eval_alignment", ROOT / "scripts/evaluate_wm.py")


@pytest.fixture
def timestamp_dataset(tmp_path):
    path = tmp_path / "SYNTHETIC_CONSECUTIVE_TIMESTAMP_FIXTURE.h5"
    with HDF5Writer(path, mode="error") as writer:
        for episode in range(2):
            length = 8
            timestamp = episode * 100 + np.arange(length, dtype=np.float32)
            observation = np.broadcast_to(timestamp[:, None], (length, 30)).copy()
            writer.write_episode({
                "pixels": np.zeros((length, 8, 8, 3), dtype=np.uint8),
                "pointcloud": np.broadcast_to(timestamp[:, None, None], (length, 1200, 6)).copy(),
                "action": timestamp[:, None] + np.arange(7, dtype=np.float32)[None] / 10,
                "observation": observation,
                "proprio": observation.copy(),
                "episode_idx": np.full(length, episode, dtype=np.int64),
                "step_idx": np.arange(length, dtype=np.int64),
            })
    with h5py.File(path, "r+") as handle:
        handle.attrs["SYNTHETIC_UNIT_TEST_ONLY"] = True
        handle.attrs["provenance"] = "Synthetic timestamp alignment test only, no simulator or expert data."
    config = train_driver.make_config(path, "synthetic_timestamp_alignment")
    dataset_cfg = OmegaConf.to_container(config.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    dataset_cfg["keys_to_load"] += ["step_idx", "episode_idx"]
    dataset = load_dataset(name, transform=None, **dataset_cfg)
    yield dataset, config
    if dataset.h5_file is not None:
        dataset.h5_file.close()


def test_current_training_contract_is_four_consecutive_frames_and_single_actions(timestamp_dataset):
    dataset, config = timestamp_dataset
    assert config.history_size == lewm_config().history_size == 3
    assert config.data.dataset.frameskip == dataset.frameskip == config.model.history_stride == 1
    assert config.model.action_dim == dataset.get_dim("action") == 7
    assert dataset.num_steps == dataset.span == 4
    assert len(dataset) == 10  # Five valid windows in each eight-frame episode.
    for idx, (episode, start) in enumerate(dataset.clip_indices):
        row = dataset[idx]
        timestamp = episode * 100 + torch.arange(start, start + 4, dtype=torch.float32)
        torch.testing.assert_close(row["step_idx"], torch.arange(start, start + 4), rtol=0, atol=0)
        torch.testing.assert_close(row["episode_idx"], torch.full((4,), episode), rtol=0, atol=0)
        torch.testing.assert_close(row["pointcloud"][:, 0, 0], timestamp, rtol=0, atol=0)
        assert row["action"].shape == (4, 7)
        torch.testing.assert_close(row["action"][:, 0], timestamp, rtol=0, atol=0)


def test_real_model_predicts_next_frame_with_the_last_context_single_action(timestamp_dataset):
    dataset, config = timestamp_dataset
    # Windows s=2 and s=3 are loaded by the actual production dataset class.
    rows = [dataset[2], dataset[3]]
    batch = {key: torch.stack([row[key] for row in rows]) for key in ("pointcloud", "action")}
    model = WorldModel(action_dim=config.model.action_dim, history_stride=config.model.history_stride,
                       temporal_config=OmegaConf.to_container(config.model.temporal_config, resolve=True)).eval()
    captured = {}

    def capture_predictor(_module, arguments):
        captured["departure_action"] = arguments[1].detach().clone()

    def capture_head(_module, arguments):
        captured["head_input"] = arguments[0].detach().clone()

    handles = [model.predictor.register_forward_pre_hook(capture_predictor),
               model.predictor.head.register_forward_pre_hook(capture_head)]
    try:
        with torch.no_grad():
            result = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert model.action_dim == 7
    assert model.predictor.head.mlp[0].in_features == 512 + 7 == 519
    torch.testing.assert_close(captured["departure_action"], batch["action"][:, 2], rtol=0, atol=0)
    torch.testing.assert_close(captured["departure_action"][:, 0], torch.tensor([4., 5.]), rtol=0, atol=0)
    torch.testing.assert_close(captured["head_input"][:, -7:], batch["action"][:, 2], rtol=0, atol=0)
    assert captured["head_input"].shape == (2, 519)
    # o_s, o_(s+1), o_(s+2) plus a_(s+2) predict latent1(o_(s+3)).
    torch.testing.assert_close(batch["pointcloud"][:, 3, 0, 0], torch.tensor([5., 6.]), rtol=0, atol=0)
    torch.testing.assert_close(result["target"], result["latent1"][:, 3], rtol=0, atol=0)
    assert result["prediction"].shape == (2, 512)


def test_eval_single_action_units_preserve_official_physical_horizons_and_cem_parameters():
    cfg, solver, alignment = eval_driver.make_eval_configs(device="cpu")
    original = OmegaConf.load(ROOT / "vendor/le-wm/config/eval/cube.yaml")
    original_solver = OmegaConf.load(ROOT / "vendor/le-wm/config/eval/solver/cem.yaml")
    assert cfg.plan_config.action_block == 1
    assert cfg.plan_config.history_len == 3
    assert cfg.plan_config.horizon == cfg.plan_config.receding_horizon == 25
    assert alignment["official_plan"] == {"horizon": 5, "receding_horizon": 5, "action_block": 5}
    assert alignment["original_control_step_horizon"] == alignment["adapted_control_step_horizon"] == 25
    assert alignment["original_control_steps_between_replans"] == alignment["adapted_control_steps_between_replans"] == 25
    assert cfg.eval == original.eval
    assert cfg.eval.eval_budget == 50 and cfg.eval.goal_offset_steps == 25
    for key in ("_target_", "batch_size", "num_samples", "var_scale", "n_steps", "topk"):
        assert solver[key] == original_solver[key]
    assert solver.seed == cfg.seed == original.seed
    assert solver.device == "cpu"


def test_evaluation_rejects_old_or_unrecorded_temporal_contract_before_loading():
    good = {"_target_": "ms2_lewm.model.WorldModel", "history_stride": 1, "action_dim": 7,
            "temporal_config": OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs/temporal_encoder.yaml"), resolve=True)}
    eval_driver.validate_evaluation_model_config(good)
    for config in ({**good, "history_stride": 5}, {**good, "action_dim": 35},
                   {key: value for key, value in good.items() if key != "history_stride"}):
        with pytest.raises(ValueError, match="history_stride=1 and action_dim=7"):
            eval_driver.validate_evaluation_model_config(config)
    for temporal in (None, {"architecture": "lewm_predictor_transformer"}):
        config = dict(good)
        if temporal is None:
            config.pop("temporal_config")
        else:
            config["temporal_config"] = temporal
        with pytest.raises(ValueError, match="(?i)(temporal|architecture)"):
            eval_driver.validate_evaluation_model_config(config)


class RecordingCost(torch.nn.Module):
    """Record actual inputs from the official CEM boundary."""
    history_size = 3

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls = []

    def get_cost(self, info, actions):
        self.calls.append({"points": info["pointcloud"].clone(), "actions": actions.clone()})
        return actions[:, :, -1].square().sum(-1)


def test_eval_real_cem_observes_cached_execution_and_prepends_actual_normalized_actions():
    from types import SimpleNamespace
    import gymnasium as gym
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.solver import CEMSolver

    model = RecordingCost()
    solver = CEMSolver(model=eval_driver.adapt_evaluation_cost(model), batch_size=2,
                       num_samples=4, n_steps=1, topk=2, device="cpu", seed=34)
    scaler = StandardScaler().fit(np.stack([np.full(7, -0.2), np.full(7, 0.8)]))
    config = PlanConfig(horizon=3, receding_horizon=3, action_block=1, history_len=3)
    policy = eval_driver.make_consecutive_policy(solver=solver, config=config, process={"action": scaler})
    policy.set_env(SimpleNamespace(num_envs=2,
        action_space=gym.spaces.Box(-1, 1, (2, 7), np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (7,), np.float32)))

    def info(steps, previous, offsets=(0, 0), flush=None):
        timestamp = np.asarray(steps, dtype=np.float32) + offsets
        result = {
            "pointcloud": np.broadcast_to(timestamp[:, None, None, None], (2, 1, 24, 6)).copy(),
            "goal_pointcloud": np.zeros((2, 1, 24, 6), dtype=np.float32),
            "action": np.asarray(previous, dtype=np.float32)[:, None],
            "step_idx": np.asarray(steps, dtype=np.int64)[:, None],
            "terminated": np.zeros(2, dtype=bool),
        }
        if flush is not None:
            result["_needs_flush"] = np.asarray(flush, dtype=bool)
        return result

    issued = []
    previous = np.full((2, 7), np.nan, dtype=np.float32)
    for step in range(4):
        previous = policy.get_action(info([step, step], previous))
        issued.append(previous.copy())
        assert len(model.calls) == (1 if step < 3 else 2)
    assert model.calls[0]["points"].shape[2] == 1
    assert model.calls[0]["actions"].shape[2] == 3  # No invented startup action prefix.
    assert model.calls[1]["points"].shape[2] == 3
    torch.testing.assert_close(model.calls[1]["points"][0, 0, :, 0, 0], torch.tensor([1., 2., 3.]))
    assert model.calls[1]["actions"].shape == (2, 4, 5, 7)
    expected = scaler.transform(np.stack(issued[1:3], axis=1).reshape(-1, 7)).reshape(2, 2, 7)
    torch.testing.assert_close(model.calls[1]["actions"][:, 0, :2], torch.from_numpy(expected).float())

    # Reset env0 independently while env1 keeps its current cached plan.
    previous = policy.get_action(info([0, 4], previous, offsets=(100, 0), flush=[True, False]))
    assert len(model.calls) == 3
    assert model.calls[-1]["points"].shape[0:3] == (1, 4, 1)
    assert model.calls[-1]["points"][0, 0, 0, 0, 0] == 100
    assert model.calls[-1]["actions"].shape[2] == 3
    # World EnvPool reuses its dictionary; a persisting flush flag must not
    # clear the new history or cached plan again on the next real step.
    policy.get_action(info([1, 5], previous, offsets=(100, 0), flush=[True, False]))
    assert len(model.calls) == 3
    assert len(policy._point_history[0]) == 2
    assert len(policy._point_history[1]) == 3


def test_eval_cost_groups_different_episode_history_lengths_without_dummy_actions():
    model = RecordingCost()
    adapter = eval_driver.adapt_evaluation_cost(model)
    candidates = torch.randn(2, 4, 5, 7)
    past = torch.randn(2, 4, 2, 7)
    past[0].fill_(float("nan"))  # Initial episode has no executed context actions.
    info = {"pointcloud": torch.randn(2, 4, 3, 24, 6),
            "goal_pointcloud": torch.randn(2, 4, 1, 24, 6),
            "_wm_context_action": past,
            "_wm_history_length": torch.tensor([[1] * 4, [3] * 4])}
    costs = adapter.get_cost(info, candidates)
    assert len(model.calls) == 2
    assert [call["points"].shape[2] for call in model.calls] == [1, 3]
    torch.testing.assert_close(model.calls[0]["actions"], candidates[:1])
    torch.testing.assert_close(model.calls[1]["actions"], torch.cat([past[1:], candidates[1:]], dim=2))
    assert torch.isfinite(costs).all()
    torch.testing.assert_close(costs, candidates[:, :, -1].square().sum(-1))


def test_dataset_evaluation_initial_step_marker_does_not_reset_history_on_first_live_step():
    from types import SimpleNamespace
    import gymnasium as gym
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.solver import CEMSolver

    model = RecordingCost()
    solver = CEMSolver(model=eval_driver.adapt_evaluation_cost(model), num_samples=2,
                       n_steps=1, topk=2, device="cpu", seed=38)
    policy = eval_driver.make_consecutive_policy(solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=3, action_block=1, history_len=3),
        dataset_start_steps=[7])
    policy.set_env(SimpleNamespace(num_envs=1,
        action_space=gym.spaces.Box(-1, 1, (1, 7), np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (7,), np.float32)))
    previous = np.full((1, 1, 7), np.nan, dtype=np.float32)
    for local_step in range(3):
        info = {"pointcloud": np.full((1, 1, 24, 6), 7 + local_step, dtype=np.float32),
                "goal_pointcloud": np.zeros((1, 1, 24, 6), dtype=np.float32),
                "action": previous,
                "step_idx": np.asarray([[7 if local_step == 0 else local_step]], dtype=np.int64),
                "terminated": np.zeros(1, dtype=bool)}
        previous = policy.get_action(info)[:, None]
    assert len(model.calls) == 1  # No accidental flush/replan when 7 switches to 1.
    assert [float(frame[0, 0]) for frame in policy._point_history[0]] == [7., 8., 9.]
    assert len(policy._executed_actions[0]) == 2
