from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from planning.latent_cem import AuxiliaryRollout
from scripts import eval_liftcube_latent_cem as cem_eval
from scripts.eval_liftcube_latent_cem import (
    build_goal_cloud,
    extract_proprio,
    run_episode,
    summarize_height_trace,
)


class FakePose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float32)
        self.q = np.asarray(q, dtype=np.float32)


class FakeObject:
    def __init__(self):
        self.pose = FakePose([0.0, 0.0, 0.1])

    def set_pose(self, pose) -> None:
        self.pose = FakePose(pose.p, pose.q)


class FakeLiftCubeEnv:
    def __init__(
        self,
        success_step: int | None = None,
        terminated_step: int | None = None,
    ):
        self.unwrapped = self
        self.obj = FakeObject()
        self.goal_height = 0.2
        self.cube_half_size = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self._sim_steps_per_control = 10
        self.goal_pos = np.array([0.5, 0.0, 0.2], dtype=np.float32)
        self.action_space = SimpleNamespace(shape=(7,))
        self.success_step = success_step
        self.terminated_step = terminated_step
        self.step_count = 0
        self.actions: list[np.ndarray] = []

    def _set_step_pose(self) -> None:
        self.obj.set_pose(FakePose([self.step_count / 100.0, 0.0, 0.1]))

    def reset(self, seed: int):
        self.step_count = 0
        self.actions.clear()
        self._set_step_pose()
        return self.get_obs(), {"seed": seed}

    def get_state(self):
        return {"step": self.step_count, "pose": self.obj.pose.p.copy()}

    def set_state(self, state) -> None:
        self.step_count = int(state["step"])
        self.obj.set_pose(FakePose(state["pose"]))

    def get_obs(self):
        xyz = np.repeat(self.obj.pose.p[None], 1200, axis=0)
        xyzw = np.column_stack([xyz, np.ones(len(xyz), dtype=np.float32)])
        return {
            "agent": {
                "qpos": np.full(9, self.step_count, dtype=np.float32),
                "qvel": np.full(9, self.step_count + 100, dtype=np.float32),
            },
            "extra": {
                "tcp_pose": np.array(
                    [self.step_count, 0, 0, 1, 0, 0, 0], dtype=np.float32
                )
            },
            "pointcloud": {
                "xyzw": xyzw.astype(np.float32),
                "robot_seg": np.zeros((len(xyzw), 1), dtype=bool),
            }
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.step_count += 1
        self._set_step_pose()
        success = self.success_step is not None and self.step_count >= self.success_step
        terminated = (
            self.terminated_step is not None
            and self.step_count >= self.terminated_step
        )
        return self.get_obs(), float(self.step_count), terminated, False, {"success": success}


class FakeWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(T=3, K=1, D=2, skip_frame=10)

    def _aggregate(self, points):
        mean_xyz = points.mean(dim=2)
        return mean_xyz[..., :2].unsqueeze(2)

    def _predict_sequence(self, latent, actions, env):
        chunks = actions.reshape(actions.shape[0], 3, 10, 7)
        delta = chunks[..., 0].mean(dim=2)
        prediction = latent.clone()
        prediction[..., 1] += delta.unsqueeze(-1)
        return prediction, latent


class RawObservationEnv(FakeLiftCubeEnv):
    def get_obs(self):
        wrapped = super().get_obs()
        return {"image": wrapped["pointcloud"]}


class FakePointCloudWrapper:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped

    def observation(self, observation):
        return {"pointcloud": observation["image"]}


class FakeOuterWrapper:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped


class RecordingPlanner:
    def __init__(self):
        self.config = SimpleNamespace(
            horizon=2, chunk_size=10, action_dim=7, trajectory_weight=0.25
        )
        self.calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self.last_cost = 0.125

    def plan(self, model, latent_history, context_actions, goal_latent, *, env, seed):
        self.calls.append(
            (latent_history.detach().cpu().clone(), context_actions.detach().cpu().clone(), seed)
        )
        plan = torch.zeros(2, 10, 7, device=latent_history.device)
        plan[..., 0] = 0.25
        plan[..., -1] = -1.0
        return plan


class IdentityProprioEncoder:
    def normalize(self, value):
        return value


class FakeEnabledWorldModel(FakeWorldModel):
    def __init__(self):
        super().__init__()
        self.cfg.proprio = SimpleNamespace(enabled=True, dim=25)
        self.proprio_encoder = IdentityProprioEncoder()

    def predict_transition(
        self, latent_history, proprio_history_normalized, actions, env=None
    ):
        batch = latent_history.shape[0]
        object_pos = torch.zeros(batch, 3, device=latent_history.device)
        object_pos[:, 2] = 0.30
        return SimpleNamespace(
            scene_latent=latent_history[:, -1],
            next_proprio_normalized=proprio_history_normalized[:, -1],
            next_proprio=proprio_history_normalized[:, -1],
            object_pos=object_pos,
        )


class RecordingGoalPlanner:
    def __init__(self):
        self.config = SimpleNamespace(
            horizon=2, chunk_size=10, action_dim=7, trajectory_weight=0.25
        )
        self.calls = []
        self.last_cost = 0.025
        self.last_cost_components = {
            "height": 0.0,
            "hold": 0.0,
            "stability": 0.0,
            "latent": 0.0,
            "approach": 0.0,
            "early_close": 0.0,
        }
        self.last_rollout = None

    def plan_with_goal(
        self,
        model,
        latent_history,
        proprio_history_normalized,
        context_actions,
        goal_spec,
        *,
        env,
        seed,
    ):
        self.calls.append(
            (
                latent_history.detach().cpu().clone(),
                proprio_history_normalized.detach().cpu().clone(),
                context_actions.detach().cpu().clone(),
                goal_spec,
                seed,
            )
        )
        raw_proprio = torch.zeros(1, 2, 25, device=latent_history.device)
        object_pos = torch.zeros(1, 2, 3, device=latent_history.device)
        object_pos[..., 2] = 0.30
        self.last_rollout = AuxiliaryRollout(
            scene_latent=torch.zeros(1, 2, 1, 2, device=latent_history.device),
            proprio_normalized=torch.zeros(
                1, 2, 25, device=latent_history.device
            ),
            proprio=raw_proprio,
            object_pos=object_pos,
        )
        plan = torch.zeros(2, 10, 7, device=latent_history.device)
        plan[..., -1] = 1.0
        return plan


def test_extract_online_proprio_uses_manifest_layout() -> None:
    observation = {
        "agent": {
            "qpos": np.arange(9, dtype=np.float32),
            "qvel": np.arange(9, dtype=np.float32) + 10,
        },
        "extra": {"tcp_pose": np.arange(7, dtype=np.float32) + 20},
    }

    value = extract_proprio(observation)

    np.testing.assert_array_equal(value[:9], np.arange(9))
    np.testing.assert_array_equal(value[9:18], np.arange(9) + 10)
    np.testing.assert_array_equal(value[18:], np.arange(7) + 20)


def test_enabled_planner_receives_goal_latent_without_replacing_physical_success(monkeypatch) -> None:
    monkeypatch.setattr(
        cem_eval,
        "build_goal_cloud",
        lambda *args, **kwargs: np.array([[0.7, 0.2, 0.3]], dtype=np.float32),
    )
    env = FakeLiftCubeEnv()
    planner = RecordingGoalPlanner()

    result = run_episode(
        env,
        FakeEnabledWorldModel(),
        planner,
        env_seed=4100,
        cem_seed=3101,
        max_steps=30,
    )

    assert result["planner_predicted_success"] is True
    assert result["success"] is False
    assert result["success_by_peak_and_hold"] is False
    assert result["environment_success"] is False
    _, proprio_history, _, goal, _ = planner.calls[0]
    torch.testing.assert_close(
        proprio_history[0, :, 0], torch.tensor([0.0, 10.0, 20.0])
    )
    assert np.isclose(goal.initial_cube_height, 0.1)
    assert np.isclose(goal.lift_threshold, 0.22)
    assert goal.demonstrated_goal_latent is not None
    torch.testing.assert_close(
        goal.demonstrated_goal_latent,
        torch.tensor([[[0.7, 0.2]]]),
    )
    assert result["replans"][0]["planner_predicted_success"] is True
    np.testing.assert_allclose(
        result["replans"][0]["predicted_object_height"], [0.3, 0.3]
    )


def test_goal_capture_restores_complete_environment_state() -> None:
    env = FakeLiftCubeEnv()
    env.reset(seed=4100)
    for _ in range(7):
        env.step(np.zeros(7, dtype=np.float32))
    before = env.get_state()

    goal = build_goal_cloud(env, sample_key=9, point_count=1)

    np.testing.assert_allclose(goal[0], env.goal_pos)
    assert env.get_state()["step"] == before["step"]
    np.testing.assert_array_equal(env.get_state()["pose"], before["pose"])


def test_goal_capture_replays_observation_wrappers_on_raw_base_obs() -> None:
    base = RawObservationEnv()
    env = FakeOuterWrapper(FakePointCloudWrapper(base))
    base.reset(seed=4100)

    goal = build_goal_cloud(env, sample_key=9, point_count=1)

    np.testing.assert_allclose(goal[0], base.goal_pos)


def test_summarize_height_trace_distinguishes_peak_from_held_success() -> None:
    summary = summarize_height_trace(
        [0.02, 0.12, 0.06],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=2,
        height_tolerance=0.01,
    )

    assert summary["max_cube_height"] == 0.12
    assert summary["final_cube_height"] == 0.06
    assert summary["peak_reached"] is True
    assert summary["success"] is False


def test_summarize_height_trace_reports_peak_and_final_height() -> None:
    summary = summarize_height_trace(
        [0.02, 0.08, 0.12],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=1,
        height_tolerance=0.01,
    )

    assert summary["max_cube_height"] == 0.12
    assert summary["final_cube_height"] == 0.12
    assert summary["peak_reached"] is True


def test_summarize_height_trace_rejects_no_lift() -> None:
    summary = summarize_height_trace(
        [0.02, 0.021, 0.019],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=1,
        height_tolerance=0.01,
    )

    assert summary["peak_reached"] is False
    assert summary["success"] is False
    assert np.isclose(summary["max_height_delta"], 0.001)


def test_summarize_height_trace_checks_stable_peak_region() -> None:
    unstable = summarize_height_trace(
        [0.12, 0.11, 0.12],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=3,
        height_tolerance=0.005,
    )
    stable = summarize_height_trace(
        [0.12, 0.11, 0.12],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=3,
        height_tolerance=0.01,
    )

    assert unstable["peak_reached"] is True
    assert unstable["stable_peak"] is False
    assert unstable["success"] is False
    assert stable["stable_peak"] is True
    assert stable["success"] is True


def test_summarize_height_trace_centers_peak_window_on_one_based_step() -> None:
    summary = summarize_height_trace(
        [0.111, 0.112, 0.12, 0.113, 0.112, 0.05],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=5,
        height_tolerance=0.02,
    )

    assert summary["max_height_step"] == 3
    assert summary["stable_peak"] is True
    assert summary["success"] is True


def test_summarize_height_trace_handles_empty_and_nan_inputs() -> None:
    empty = summarize_height_trace(
        [],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=1,
        height_tolerance=0.01,
    )
    nan_trace = summarize_height_trace(
        [0.02, float("nan"), 0.12],
        cube_z0=0.02,
        lift_threshold=0.10,
        hold_steps=1,
        height_tolerance=0.01,
    )

    assert np.isnan(empty["final_cube_height"])
    assert empty["success"] is False
    assert nan_trace["has_nan"] is True
    assert nan_trace["stable_peak"] is False
    assert nan_trace["success"] is False


def test_run_episode_uses_0_10_20_context_and_shifts_action_history() -> None:
    env = FakeLiftCubeEnv()
    model = FakeWorldModel()
    planner = RecordingPlanner()

    result = run_episode(
        env,
        model,
        planner,
        env_seed=4100,
        cem_seed=3101,
        max_steps=40,
    )

    assert result["success"] is False
    assert result["steps"] == 40
    assert result["planning_calls"] == 2
    assert len(result["replans"]) == 2
    first_latents, first_actions, first_seed = planner.calls[0]
    second_latents, second_actions, second_seed = planner.calls[1]
    torch.testing.assert_close(first_latents[0, :, 0, 0], torch.tensor([0.0, 0.1, 0.2]))
    torch.testing.assert_close(second_latents[0, :, 0, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert first_seed == 3101 and second_seed == 3102
    expected_warmup = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    torch.testing.assert_close(first_actions[0], expected_warmup.expand(20, -1))
    torch.testing.assert_close(second_actions[0, :10], expected_warmup.expand(10, -1))
    torch.testing.assert_close(second_actions[0, 10:, 0], torch.full((10,), 0.25))
    torch.testing.assert_close(second_actions[0, 10:, -1], torch.full((10,), -1.0))
    assert {
        "predicted_cost",
        "zero_action_cost",
        "cube_height",
        "reward",
        "success",
        "arm_action_mean",
        "arm_action_std",
        "gripper",
    } <= result["replans"][0].keys()


def test_run_episode_keeps_running_after_info_success_for_hold_evaluation() -> None:
    env = FakeLiftCubeEnv(success_step=23)
    result = run_episode(
        env,
        FakeWorldModel(),
        RecordingPlanner(),
        env_seed=4100,
        cem_seed=3101,
        max_steps=60,
    )

    assert result["success"] is False
    assert result["environment_success"] is True
    assert result["steps"] == 60
    assert result["planning_calls"] == 4
    assert len(env.actions) == 60


def test_run_episode_records_height_trace_until_early_termination() -> None:
    env = FakeLiftCubeEnv(terminated_step=4)
    result = run_episode(
        env,
        FakeWorldModel(),
        RecordingPlanner(),
        env_seed=4100,
        cem_seed=3101,
        max_steps=60,
    )

    assert result["steps"] == 4
    assert np.isclose(result["final_cube_height"], 0.1)
    np.testing.assert_allclose(result["height_trace"], [0.1, 0.1, 0.1, 0.1])
    assert result["height_trace_control_steps"] == [1, 2, 3, 4]
    assert result["height_trace_sim_frames"] == [10, 20, 30, 40]
    assert result["control_steps"] == 4
    assert result["sim_frames"] == 40


def test_cli_records_model_batch_size_in_provenance(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output"

    class FakeEnv:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        cem_eval,
        "load_checkpoint_model",
        lambda path, device: (object(), {"step": 2500}),
    )
    monkeypatch.setattr(
        cem_eval,
        "action_stats",
        lambda data_root, split: (np.zeros(7), np.ones(7)),
    )
    monkeypatch.setattr(cem_eval, "sha256_file", lambda path: "abc123")
    monkeypatch.setattr(
        cem_eval,
        "score_validation_reference",
        lambda model, state, device, data_root: {},
    )
    monkeypatch.setattr(cem_eval, "make_env", FakeEnv)
    monkeypatch.setattr(
        cem_eval,
        "run_episode",
        lambda *args, **kwargs: {
            "success": False,
            "initial_cost": 0.0,
            "replans": [],
        },
    )

    cem_eval.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--data-root",
            str(tmp_path),
            "--seeds",
            "4100",
            "--model-batch-size",
            "7",
            "--device",
            "cpu",
        ]
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["cem"]["model_batch_size"] == 7


def test_validation_reference_uses_rgb_when_model_has_six_channels(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, root, **kwargs):
            captured.update(kwargs)

        def __getitem__(self, index):
            context = torch.zeros(3, 1, 6)
            future = torch.zeros(2, 1, 6)
            actions = torch.zeros(20, 7)
            return index, context, future, actions

        def context_actions_for_index(self, index):
            return torch.zeros(20, 7)

    class FakeModel:
        cfg = SimpleNamespace(T=3, K=1, D=2, C=6, skip_frame=10)

        def _aggregate(self, points):
            return torch.zeros(points.shape[0], points.shape[1], 1, 2)

        def _predict_sequence(self, latent, actions, env):
            return latent, latent

    monkeypatch.setattr(cem_eval, "LiftCubeRolloutDataset", FakeDataset)

    cem_eval.score_validation_reference(
        FakeModel(), {}, torch.device("cpu"), tmp_path
    )

    assert captured["include_rgb"] is True


def test_enabled_validation_reference_supplies_normalized_proprio(
    tmp_path, monkeypatch
) -> None:
    class Ref:
        episode_id = 7
        start = 0

    class FakeDataset:
        index = [Ref()]

        def __init__(self, root, **kwargs):
            self.T = kwargs["T"]
            self.skip_frame = kwargs["skip_frame"]

        def __getitem__(self, index):
            return (
                index,
                torch.zeros(3, 1, 6),
                torch.zeros(2, 1, 6),
                torch.zeros(20, 7),
            )

        def context_actions_for_index(self, index):
            return torch.zeros(20, 7)

        def _load_episode(self, episode_id):
            frames = 50
            qpos = np.zeros((frames, 9), dtype=np.float32)
            qpos[:, 0] = np.arange(frames)
            return {
                "qpos": qpos,
                "qvel": np.zeros((frames, 9), dtype=np.float32),
                "tcp_pose": np.zeros((frames, 7), dtype=np.float32),
            }

    class FakeEncoder:
        def normalize(self, value):
            return value + 1.0

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.cfg = SimpleNamespace(
                T=3,
                K=1,
                D=2,
                C=6,
                skip_frame=10,
                proprio=SimpleNamespace(enabled=True, dim=25),
            )
            self.proprio_encoder = FakeEncoder()
            self.seen_proprio = []

        def _aggregate(self, points):
            return torch.zeros(points.shape[0], points.shape[1], 1, 2)

        def predict_transition(
            self, latent_history, proprio_history_normalized, actions, env=None
        ):
            self.seen_proprio.append(proprio_history_normalized.detach().clone())
            batch = latent_history.shape[0]
            next_proprio = proprio_history_normalized[:, -1]
            return SimpleNamespace(
                scene_latent=latent_history[:, -1],
                next_proprio_normalized=next_proprio,
                next_proprio=next_proprio,
                object_pos=torch.zeros(batch, 3),
            )

        def _predict_sequence(self, *args, **kwargs):
            raise AssertionError("enabled reference used latent-only rollout")

    monkeypatch.setattr(cem_eval, "LiftCubeRolloutDataset", FakeDataset)
    model = FakeModel()

    result = cem_eval.score_validation_reference(
        model, {}, torch.device("cpu"), tmp_path
    )

    assert set(result) == {
        "demonstrated_action_cost",
        "zero_action_cost",
        "persistence_cost",
    }
    torch.testing.assert_close(
        model.seen_proprio[0][0, :, 0], torch.tensor([1.0, 11.0, 21.0])
    )
