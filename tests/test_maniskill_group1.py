import json
from pathlib import Path

import numpy as np
import pytest

from scripts.train_maniskill_group1 import (
    _forward,
    _load_initial_weights,
    _set_proprio_stats,
    _trainable_parameters,
    _unpack_batch,
    _validate_data_channels,
    _wrong_actions,
    load_group1_config,
    run_preflight,
    task_data_root,
    validate_group1_manifest,
)
from scripts.diagnose_maniskill_group1 import representation_diagnostics
from scripts.eval_maniskill_group1 import offline_metrics


def _write_manifest(
    root: Path, *, task="LiftCube-v0", total=100, train=None, val=None
):
    train_count = int(total * 0.9)
    train = list(range(train_count)) if train is None else train
    val = list(range(train_count, total)) if val is None else val
    manifest = {
        "task": task,
        "episode_ids": list(range(total)),
        "train_episode_ids": train,
        "val_episode_ids": val,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_group1_manifest_requires_liftcube_90_10_split(tmp_path):
    root = tmp_path / "LiftCube-v0"
    _write_manifest(root)
    result = validate_group1_manifest(root)
    assert result["task"] == "LiftCube-v0"
    assert len(result["train_episode_ids"]) == 90
    assert len(result["val_episode_ids"]) == 10

    _write_manifest(root, task="PickCube-v0")
    with pytest.raises(ValueError, match="LiftCube-v0"):
        validate_group1_manifest(root)


def test_group1_manifest_accepts_configured_pickcube_900_100_split(tmp_path):
    root = tmp_path / "PickCube-v0"
    _write_manifest(root, task="PickCube-v0", total=1000)

    result = validate_group1_manifest(
        root,
        expected_task="PickCube-v0",
        expected_episodes=1000,
    )

    assert result["task"] == "PickCube-v0"
    assert len(result["train_episode_ids"]) == 900
    assert len(result["val_episode_ids"]) == 100


def test_group1_config_and_data_root_accept_pickcube_override():
    repo_root = Path(__file__).resolve().parents[1]
    _, train_cfg = load_group1_config(
        repo_root / "configs" / "maniskill_group1_scratch.yaml",
        ["train.task=PickCube-v0", "train.expected_episodes=1000"],
    )

    assert train_cfg.task == "PickCube-v0"
    assert train_cfg.expected_episodes == 1000
    assert task_data_root(Path("/datasets/compact"), train_cfg.task) == Path(
        "/datasets/compact/PickCube-v0"
    )


def test_group1_config_has_scratch_training_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg, train_cfg = load_group1_config(repo_root / "configs" / "maniskill_group1_scratch.yaml", [])
    assert train_cfg.task == "LiftCube-v0"
    assert train_cfg.env == "maniskill2"
    assert train_cfg.steps > 0
    assert model_cfg.T == 3
    assert model_cfg.C == 3
    assert model_cfg.action.envs["maniskill2"].dim == 7
    assert model_cfg.aggregator.backend == "scratch_ptv3"
    assert train_cfg.include_rgb is False


def test_wrong_actions_preserve_chunks_but_break_batch_alignment():
    import torch

    actions = torch.arange(3 * 6, dtype=torch.float32).view(3, 6, 1)
    wrong = _wrong_actions(actions, chunk_size=2)
    assert torch.equal(wrong, actions.roll(1, dims=0))
    assert not torch.equal(wrong, actions)


def test_wrong_actions_roll_transition_chunks_for_batch_size_one():
    import torch

    actions = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    wrong = _wrong_actions(actions, chunk_size=2)
    expected = actions.view(1, 3, 2, 1).roll(1, dims=1).reshape_as(actions)
    assert torch.equal(wrong, expected)
    assert not torch.equal(wrong, actions)


def test_skip10_action_token_ranking_config():
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg, train_cfg = load_group1_config(
        repo_root / "configs" / "maniskill_group1_skip10_action_tokens_rank.yaml", []
    )
    assert model_cfg.skip_frame == 10
    assert model_cfg.action.output_scale == 1.0
    assert model_cfg.predictor.fusion == "cross_attention"
    assert model_cfg.loss.ranking_weight == 1.0
    assert model_cfg.loss.ranking_margin == 0.1
    assert train_cfg.steps == 2500

    runner = (repo_root / "scripts" / "run_skip10_action_tokens_rank_experiment.sh").read_text()
    assert "python=/nix/lewm/4d-jepa-worldmodel/bin/python" in runner


def test_skip10_action_sequence_batch8_config_disables_motion_weighting():
    repo_root = Path(__file__).resolve().parents[1]
    model_cfg, train_cfg = load_group1_config(
        repo_root / "configs" / "maniskill_group1_skip10_action_sequence_batch8.yaml", []
    )

    assert model_cfg.skip_frame == 10
    assert model_cfg.action.output_scale == 1.0
    assert model_cfg.action.token_encoder_layers == 1
    assert model_cfg.action.token_encoder_heads == 4
    assert model_cfg.action.token_mlp_ratio == 2.0
    assert model_cfg.action.token_position_encoding == "sinusoidal"
    assert model_cfg.action.token_position_scale == 1.0
    assert model_cfg.predictor.fusion == "cross_attention"
    assert model_cfg.loss.flow_weight == 0.0
    assert model_cfg.loss.ranking_weight == 1.0
    assert model_cfg.loss.ranking_motion_weighted is False
    assert train_cfg.batch_size == 8
    assert train_cfg.steps == 2500
    assert train_cfg.run_name == "group1_skip10_action_sequence_batch8_2500"

    runner = (
        repo_root / "scripts" / "run_skip10_action_sequence_batch8_experiment.sh"
    ).read_text()
    assert "python=/nix/lewm/4d-jepa-worldmodel/bin/python" in runner
    assert "maniskill_group1_skip10_action_sequence_batch8.yaml" in runner
    assert "group1_skip10_action_sequence_batch8_2500" in runner


def test_sonata_config_uses_rgb_frozen_backend_and_same_action_pipeline():
    repo_root = Path(__file__).resolve().parents[1]

    model_cfg, train_cfg = load_group1_config(
        repo_root / "configs" / "maniskill_group1_sonata_frozen.yaml", []
    )

    assert model_cfg.C == 6
    assert model_cfg.T == 3
    assert model_cfg.skip_frame == 10
    assert model_cfg.aggregator.backend == "sonata_ptv3"
    assert model_cfg.aggregator.sonata.freeze_backbone is True
    assert model_cfg.predictor.fusion == "cross_attention"
    assert model_cfg.loss.flow_weight == 0.0
    assert model_cfg.loss.ranking_weight == 1.0
    assert train_cfg.include_rgb is True
    assert train_cfg.batch_size == 8
    assert train_cfg.steps == 2500


def test_sonata_proprio_config_uses_new_data_and_aux_losses():
    repo_root = Path(__file__).resolve().parents[1]

    model_cfg, train_cfg = load_group1_config(
        repo_root / "configs" / "maniskill_group1_sonata_proprio_goal.yaml", []
    )

    assert train_cfg.data_root.endswith("compact_rgb_proprio")
    assert train_cfg.run_name == "group1_skip10_sonata_proprio_goal_batch8_2500"
    assert train_cfg.include_rgb is True
    assert train_cfg.include_proprio is True
    assert model_cfg.C == 6
    assert model_cfg.T == 3
    assert model_cfg.skip_frame == 10
    assert model_cfg.K == 512
    assert model_cfg.D == 64
    assert model_cfg.aggregator.backend == "sonata_ptv3"
    assert model_cfg.aggregator.sonata.freeze_backbone is True
    assert model_cfg.predictor.fusion == "cross_attention"
    assert model_cfg.proprio.enabled is True
    assert model_cfg.proprio.dim == 25
    assert model_cfg.loss.ranking_weight == 1.0
    assert model_cfg.loss.proprio_weight > 0
    assert model_cfg.loss.object_weight > 0
    assert train_cfg.batch_size == 8
    assert train_cfg.steps == 2500

    runner = (
        repo_root / "scripts" / "run_liftcube_sonata_proprio_goal_2500.sh"
    ).read_text()
    assert "LIFTCUBE_GPU" in runner
    assert "nvidia-smi" in runner
    assert "maniskill_group1_sonata_proprio_goal.yaml" in runner
    assert "group1_skip10_sonata_proprio_goal_batch8_2500" in runner
    assert "--resume" not in runner


def test_contact_finetune_config_and_runner_are_reproducible():
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "configs" / "maniskill_group1_sonata_contact_lzh24.yaml"
    model_cfg, train_cfg = load_group1_config(config, [])

    assert model_cfg.loss.contact_weight == 1.0
    assert model_cfg.loss.contact_positive_weight == pytest.approx(0.5886297376)
    assert model_cfg.loss.contact_negative_weight == pytest.approx(3.3207236842)
    assert train_cfg.contact_phase_repeat == 4
    assert train_cfg.contact_sidecar.endswith("contact_sidecar.npz")
    assert train_cfg.steps == 1000

    runner = (repo_root / "scripts" / "run_liftcube_sonata_contact_lzh24.sh").read_text()
    assert "--init-checkpoint" in runner
    assert "681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28" in runner
    assert "7c3868dcbeaf102e50461dc82f88c8cf605515c12692c0e47fc2199e5353af4f" in runner
    assert "liftcube-joint-cost-contact" in runner


def test_unpack_batch_preserves_legacy_and_enabled_shapes():
    import torch

    points = torch.zeros(2, 4, 8, 6)
    actions = torch.zeros(2, 30, 7)
    proprio = torch.zeros(2, 4, 25)
    object_pos = torch.zeros(2, 4, 3)

    legacy = _unpack_batch((points, actions))
    assert legacy[0] is points
    assert legacy[1] is actions
    assert legacy[2:] == (None, None, None)
    enabled = _unpack_batch(
        {
            "points": points,
            "actions": actions,
            "proprio": proprio,
            "object_pos": object_pos,
            "cube_robot_contact": torch.zeros(2, 4),
        }
    )
    assert enabled[0] is points
    assert enabled[1] is actions
    assert enabled[2] is proprio
    assert enabled[3] is object_pos
    assert enabled[4].shape == (2, 4)


def test_initial_checkpoint_allows_only_new_contact_head(tmp_path):
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self, *, contact: bool):
            super().__init__()
            self.existing = torch.nn.Linear(3, 2)
            self.contact_head = torch.nn.Linear(2, 1) if contact else None

    legacy = TinyModel(contact=False)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"step": 2500, "model": legacy.state_dict()}, checkpoint)

    initialized = TinyModel(contact=True)
    result = _load_initial_weights(initialized, checkpoint, torch.device("cpu"))

    assert result["source_step"] == 2500
    assert result["missing_keys"] == ["contact_head.bias", "contact_head.weight"]
    torch.testing.assert_close(
        initialized.existing.weight, legacy.existing.weight
    )

    broken = dict(legacy.state_dict())
    broken.pop("existing.weight")
    torch.save({"model": broken}, checkpoint)
    with pytest.raises(RuntimeError, match="existing.weight"):
        _load_initial_weights(initialized, checkpoint, torch.device("cpu"))


def test_set_proprio_stats_requires_enabled_exact_25d_values():
    import numpy as np

    from models import WorldModelConfig

    cfg = WorldModelConfig()
    with pytest.raises(ValueError, match="disabled"):
        _set_proprio_stats(
            cfg,
            np.zeros(25, dtype=np.float32),
            np.ones(25, dtype=np.float32),
        )

    cfg.proprio.enabled = True
    _set_proprio_stats(
        cfg,
        np.arange(25, dtype=np.float32),
        np.arange(1, 26, dtype=np.float32),
    )
    assert cfg.proprio.mean == list(range(25))
    assert cfg.proprio.std == list(range(1, 26))


def test_data_channel_validation_rejects_rgb_config_mismatch():
    from models import WorldModelConfig
    from scripts.train_maniskill_group1 import Group1TrainConfig

    with pytest.raises(ValueError, match="include_rgb"):
        _validate_data_channels(
            WorldModelConfig(C=3), Group1TrainConfig(include_rgb=True)
        )
    with pytest.raises(ValueError, match="include_rgb"):
        _validate_data_channels(
            WorldModelConfig(C=6), Group1TrainConfig(include_rgb=False)
        )


def test_optimizer_parameter_list_excludes_frozen_parameters():
    import torch

    trainable = torch.nn.Parameter(torch.ones(1))
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    module = torch.nn.ParameterList([trainable, frozen])

    result = _trainable_parameters(module)

    assert result == [trainable]


def test_group1_diagnostics_report_collapse_and_finiteness():
    import torch

    latents = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    result = representation_diagnostics(latents)
    assert result["samples"] == 3
    assert result["latent_std"] > 0
    assert result["effective_rank"] >= 1
    assert result["collapse_flag"] is False


def test_group1_offline_metrics_compare_prediction_and_persistence():
    import torch

    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 1.0], [2.0, 4.0]])
    persistence = torch.tensor([[0.0, 1.0], [3.0, 3.0]])
    result = offline_metrics(prediction, target, persistence)
    assert result["pred_mse"] == pytest.approx(0.5)
    assert result["persistence_mse"] == pytest.approx(0.75)
    assert result["prediction_gain"] == pytest.approx(1.5)


def test_preflight_passes_mapping_state_and_reports_auxiliary_metrics():
    import torch
    from types import SimpleNamespace

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.cfg = SimpleNamespace(
                skip_frame=2,
                loss=SimpleNamespace(ranking_weight=0.0),
                proprio=SimpleNamespace(enabled=True),
            )
            self.received = None

        def forward(
            self,
            points,
            actions,
            *,
            env,
            negative_actions,
            proprio,
            object_pos,
            cube_robot_contact,
        ):
            self.received = (proprio, object_pos, cube_robot_contact)
            signal = self.scale * (
                points.mean()
                + actions.mean()
                + proprio.mean()
                + object_pos.mean()
                + cube_robot_contact.mean()
            )
            loss = signal.square() + self.scale.square()
            names = (
                "pred_loss",
                "latent_loss",
                "flow_loss",
                "rank_loss",
                "reg_loss",
                "proprio_loss",
                "object_loss",
                "qpos_mse",
                "qvel_mse",
                "tcp_mse",
                "object_xyz_mse",
                "object_z_mse",
                "contact_loss",
                "contact_accuracy",
            )
            return {"loss": loss, **{name: loss for name in names}}

    points = torch.randn(2, 4, 8, 6)
    actions = torch.randn(2, 6, 7)
    proprio = torch.randn(2, 4, 25)
    object_pos = torch.randn(2, 4, 3)
    contact = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]], dtype=torch.float32)
    model = FakeModel()

    result = run_preflight(
        model,
        [
            {
                "points": points,
                "actions": actions,
                "proprio": proprio,
                "object_pos": object_pos,
                "cube_robot_contact": contact,
            }
        ],
        torch.device("cpu"),
        "maniskill2",
    )

    assert model.received[0].shape == (2, 4, 25)
    assert model.received[1].shape == (2, 4, 3)
    assert model.received[2].shape == (2, 4)
    assert result["proprio_shape"] == [2, 4, 25]
    assert result["object_pos_shape"] == [2, 4, 3]
    assert result["cube_robot_contact_shape"] == [2, 4]
    assert result["action_input_grad_norm"] > 0
    assert result["proprio_input_grad_norm"] > 0
    for name in (
        "proprio_loss",
        "object_loss",
        "qpos_mse",
        "qvel_mse",
        "tcp_mse",
        "object_xyz_mse",
        "object_z_mse",
        "contact_loss",
        "contact_accuracy",
    ):
        assert np.isfinite(result[name])


def test_forward_passes_optional_state_without_changing_negative_actions():
    import torch
    from types import SimpleNamespace

    class RecordingModel:
        cfg = SimpleNamespace(
            skip_frame=2, loss=SimpleNamespace(ranking_weight=1.0)
        )

        def __call__(self, points, actions, **kwargs):
            self.kwargs = kwargs
            return {"loss": torch.zeros(())}

    model = RecordingModel()
    points = torch.zeros(2, 4, 8, 6)
    actions = torch.arange(2 * 6 * 7, dtype=torch.float32).reshape(2, 6, 7)
    proprio = torch.zeros(2, 4, 25)
    object_pos = torch.zeros(2, 4, 3)
    contact = torch.zeros(2, 4)

    _forward(
        model,
        points,
        actions,
        "maniskill2",
        proprio=proprio,
        object_pos=object_pos,
        cube_robot_contact=contact,
    )

    assert model.kwargs["proprio"] is proprio
    assert model.kwargs["object_pos"] is object_pos
    assert model.kwargs["cube_robot_contact"] is contact
    torch.testing.assert_close(
        model.kwargs["negative_actions"], actions.roll(1, dims=0)
    )
