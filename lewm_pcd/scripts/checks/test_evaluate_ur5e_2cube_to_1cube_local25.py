"""Contract tests for 2cube-to-1cube local-goal transfer evaluation."""

from dataclasses import asdict
import hashlib
import json

import h5py
import numpy as np
import pytest

from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
    ensure_selection,
    sample_single_cube_specs,
)


@pytest.fixture
def single_cube_h5(tmp_path):
    path = tmp_path / "single.h5"
    lengths = np.array([31, 31], dtype=np.int64)
    offsets = np.array([0, 31], dtype=np.int64)
    total = int(lengths.sum())
    positions = np.zeros((total, 3), dtype=np.float64)
    positions[:31, 0] = np.arange(31) * 0.002
    quaternions = np.zeros((total, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("ep_len", data=lengths)
        h5.create_dataset("ep_offset", data=offsets)
        h5.create_dataset("qpos", data=np.zeros((total, 21)))
        h5.create_dataset("qvel", data=np.zeros((total, 20)))
        h5.create_dataset("pcd_xyz", data=np.zeros((total, 1200, 3), dtype=np.float32))
        h5.create_dataset("pcd_rgb", data=np.zeros((total, 1200, 3), dtype=np.uint8))
        h5.create_dataset("action", data=np.zeros((total, 5), dtype=np.float32))
        h5.create_dataset("privileged_block_0_pos", data=positions)
        h5.create_dataset("privileged_block_0_quat", data=quaternions)
    return path


def test_single_cube_sampler_is_seeded_unique_and_stays_inside_episode(single_cube_h5):
    first = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=False
    )
    again = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=False
    )
    other = sample_single_cube_specs(
        single_cube_h5, count=4, seed=43, goal_offset=25, nontrivial_only=False
    )

    assert first == again
    assert first != other
    assert all(spec.goal_step < spec.length for spec in first)
    assert len({(spec.episode_index, spec.start_step) for spec in first}) == 4


def test_single_cube_sampler_filters_strictly_nontrivial_windows(single_cube_h5):
    specs = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=True
    )

    assert all(spec.max_cube_displacement > 0.04 for spec in specs)
    assert all(spec.nontrivial and not spec.start_already_success for spec in specs)
    assert {spec.episode_index for spec in specs} == {0}


def test_single_cube_sampler_rejects_noncontiguous_offsets(single_cube_h5):
    with h5py.File(single_cube_h5, "r+") as h5:
        h5["ep_offset"][1] = 30

    with pytest.raises(ValueError, match="contiguous"):
        sample_single_cube_specs(single_cube_h5, count=1, seed=42, goal_offset=25)


def test_single_cube_sampler_rejects_nonfinite_goal(single_cube_h5):
    with h5py.File(single_cube_h5, "r+") as h5:
        h5["privileged_block_0_pos"][25, 0] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        sample_single_cube_specs(single_cube_h5, count=1, seed=42, goal_offset=25)


def test_selection_manifest_is_canonical_and_reusable_across_policies(single_cube_h5, tmp_path):
    specs = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=False
    )
    identity = {
        "target_dataset": str(single_cube_h5),
        "target_dataset_bytes": single_cube_h5.stat().st_size,
        "count": 4,
        "sample_seed": 42,
        "goal_offset": 25,
        "selection": "all",
    }
    path = tmp_path / "all" / "selection.json"

    manifest, digest = ensure_selection(path, identity, specs)
    reused, reused_digest = ensure_selection(path, identity, specs)
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert manifest == reused
    assert digest == reused_digest == hashlib.sha256(canonical).hexdigest()
    assert manifest["samples"] == [asdict(spec) for spec in specs]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("target_dataset_bytes", 1),
        ("sample_seed", 43),
        ("selection", "nontrivial"),
    ],
)
def test_selection_manifest_rejects_identity_changes(
    single_cube_h5, tmp_path, key, value
):
    specs = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=False
    )
    identity = {
        "target_dataset": str(single_cube_h5),
        "target_dataset_bytes": single_cube_h5.stat().st_size,
        "count": 4,
        "sample_seed": 42,
        "goal_offset": 25,
        "selection": "all",
    }
    path = tmp_path / "selection.json"
    ensure_selection(path, identity, specs)
    changed = {**identity, key: value}

    with pytest.raises(ValueError, match="selection manifest mismatch"):
        ensure_selection(path, changed, specs)


def test_selection_manifest_rejects_sample_changes(single_cube_h5, tmp_path):
    specs = sample_single_cube_specs(
        single_cube_h5, count=4, seed=42, goal_offset=25, nontrivial_only=False
    )
    other_specs = sample_single_cube_specs(
        single_cube_h5, count=4, seed=43, goal_offset=25, nontrivial_only=False
    )
    identity = {
        "target_dataset": str(single_cube_h5),
        "target_dataset_bytes": single_cube_h5.stat().st_size,
        "count": 4,
        "sample_seed": 42,
        "goal_offset": 25,
        "selection": "all",
    }
    path = tmp_path / "selection.json"
    ensure_selection(path, identity, specs)

    with pytest.raises(ValueError, match="selection manifest mismatch"):
        ensure_selection(path, identity, other_specs)


def test_single_cube_eval_keys_use_flat_real_dataset_fields():
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
        single_cube_eval_keys,
    )

    assert single_cube_eval_keys() == [
        "qpos",
        "qvel",
        "pcd_xyz",
        "pcd_rgb",
        "action",
        "privileged_block_0_pos",
        "privileged_block_0_quat",
    ]


def test_single_cube_goal_callables_restore_sampled_goal_pose():
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
        single_cube_goal_callables,
    )

    callables = single_cube_goal_callables()
    assert callables == [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]
    assert "goal_privileged_target_block_pos" not in json.dumps(callables)


def test_single_cube_world_uses_native_task_success_protocol():
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
        make_single_cube_world_kwargs,
    )

    kwargs = make_single_cube_world_kwargs()
    assert kwargs["env_type"] == "single"
    assert kwargs["mode"] == "task"
    assert kwargs["terminate_at_goal"] is True
    assert kwargs["env_name"] == "swm/OGBCubePCD-v0"


def test_source_artifacts_are_exact_2cube_task1_inputs(tmp_path):
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import source_artifacts

    checkpoint, train = source_artifacts(tmp_path)

    assert checkpoint == (
        tmp_path
        / "checkpoints"
        / "ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_10ep"
        / "weights_epoch_10.pt"
    )
    assert train == tmp_path / "full" / "train.h5"


def test_transfer_cem_protocol_keeps_ten_iterations():
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import (
        make_transfer_cem_kwargs,
    )

    assert make_transfer_cem_kwargs(42) == {
        "batch_size": 1,
        "num_samples": 300,
        "var_scale": 1.0,
        "n_steps": 10,
        "topk": 30,
        "device": "cuda",
        "seed": 42,
    }


def test_transfer_cli_defaults_match_approved_protocol():
    from scripts.eval.evaluate_ur5e_2cube_to_1cube_local25 import build_parser

    args = build_parser().parse_args(["--policy", "cem", "--selection", "all"])

    assert args.count == 50
    assert args.sample_seed == 42
    assert args.policy_seed == 42
    assert args.goal_offset == 25
    assert args.eval_budget == 50
