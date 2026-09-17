from __future__ import annotations

import subprocess
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.checks.check_franka_checkpoint import checkpoint_filename, sha256_file


ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "env_franka.sh"
PIPELINE = ROOT / "scripts" / "run_franka_pipeline.sh"
SMOKE = ROOT / "scripts" / "run_franka_smoke.sh"
COLLECTION = ROOT / "scripts" / "run_franka_collection.sh"
AUTORUN = ROOT / "scripts" / "run_franka_autotrain_cem.sh"
EVAL = ROOT / "eval.py"


def test_franka_runtime_uses_worktree_and_isolated_data_root() -> None:
    text = ENV.read_text(encoding="utf-8")
    assert 'dirname "${BASH_SOURCE[0]}"' in text
    assert '"${STABLEWM_HOME:?' in text
    assert 'UPSTREAM_SWM_ROOT:-$LEWM_PCD_ROOT/third_party/stable-worldmodel' in text
    assert "/root/" not in text


def test_pipeline_is_fail_stop_atomic_and_explicit() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'STAGE="${1:-verify}"' in text
    assert "mktemp" in text and 'mv "$tmp" "$STATUS"' in text
    assert "Refusing to overwrite" in text
    assert "collect-pcd" in text
    assert "collect-pix" in text
    assert "check-data" in text
    assert "train-smoke" in text
    assert "loader.num_workers=1" in text
    assert "loader.num_workers=0" not in text
    assert "10000" not in text
    assert "CEM" not in text


def test_shell_entrypoints_parse() -> None:
    for path in (ENV, PIPELINE, SMOKE, COLLECTION, AUTORUN):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_random_evaluation_uses_the_configured_seed_without_model_or_cem() -> None:
    text = EVAL.read_text(encoding="utf-8")
    random_branch = text.split("    else:\n        policy = swm.policy.RandomPolicy", 1)[1]
    assert random_branch.startswith("(seed=cfg.seed)")
    assert "load_pretrained" not in random_branch
    assert "CEMSolver" not in random_branch


def test_unique_episode_sampling_is_deterministic_distinct_and_goal_safe() -> None:
    from eval import sample_unique_episode_starts

    episode_idx = np.repeat(np.array([10, 20, 30, 40]), 6)
    step_idx = np.tile(np.arange(6), 4)
    first = sample_unique_episode_starts(
        episode_idx, step_idx, num_eval=3, goal_offset_steps=2, seed=42
    )
    second = sample_unique_episode_starts(
        episode_idx, step_idx, num_eval=3, goal_offset_steps=2, seed=42
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3,)
    assert len(np.unique(episode_idx[first])) == 3
    assert np.all(step_idx[first] <= 3)


def test_unique_episode_sampling_rejects_too_few_eligible_episodes() -> None:
    from eval import sample_unique_episode_starts

    episode_idx = np.repeat(np.array([10, 20]), 3)
    step_idx = np.tile(np.arange(3), 2)
    with pytest.raises(ValueError, match="eligible episodes"):
        sample_unique_episode_starts(
            episode_idx, step_idx, num_eval=3, goal_offset_steps=1, seed=42
        )


def test_collection_runner_is_staged_atomic_and_uses_distinct_shard_namespaces() -> None:
    text = COLLECTION.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'case "${1:-}"' in text
    assert "pilot" in text and "full" in text and "resume-full" in text
    assert "ogbench/franka_pilot/cube_franka_pcd_pilot_500.h5" in text
    assert "ogbench/franka_full/cube_franka_single_expert_pcd.h5" in text
    assert "collect_cube_pcd_sharded.py" in text
    assert 'collect_shards PILOT "$out" 500 10 5' in text
    assert 'collect_shards FULL "$out" 10000 20 8' in text
    assert '--max-parallel "$parallel"' in text
    assert "--no-merge" in text
    assert "summarize_episode_outcomes.py" in text
    assert "verify_franka_replay.py" in text and "--validate-only" in text
    assert "Refusing to overwrite existing output" in text
    assert "add_eval_columns.py" in text and "rechunk_h5.py" in text
    assert "all) pilot; full" in text
    assert "train.py" not in text and "eval.py" not in text and "CEM" not in text
    assert 'local partial="${rechunked}.partial"' in text
    assert 'rm -f -- "$partial"' in text
    assert 'rechunk_h5.py "$raw" "$partial" --rows 20' in text
    assert 'mv "$partial" "$rechunked"' in text
    assert "mktemp" in text and 'mv "$tmp" "$STATUS"' in text
    assert text.count("--config-name=ogb_franka_pcd") == 2


def test_checkpoint_hash_uses_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"franka-smoke")
    assert sha256_file(path) == hashlib.sha256(b"franka-smoke").hexdigest()


def test_checkpoint_filename_selects_requested_positive_epoch() -> None:
    assert checkpoint_filename(1) == "weights_epoch_1.pt"
    assert checkpoint_filename(10) == "weights_epoch_10.pt"
    with pytest.raises(ValueError, match="positive"):
        checkpoint_filename(0)


def test_autotrain_cem_runner_has_fixed_fail_stop_contract() -> None:
    text = AUTORUN.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "franka_collection.status" in text and "FULL_ACCEPTED" in text
    assert "cube_franka_single_expert_pcd.rechunk20" in text
    assert "--expected-episodes 10000" in text and "--validate-only" in text
    assert "franka_pcd_ms2_10ep_seed3072" in text
    assert "model=lewm_pcd" in text and "model/encoder=ms2" in text
    assert "trainer.max_epochs=10" in text and "loader.batch_size=128" in text
    assert "weights_epoch_10.pt" in text and "--epoch 10" in text
    assert "--config-name=cube_franka_pcd" in text
    assert "eval.num_eval=100" in text
    assert "summarize_cem_result.py" in text
    assert "franka_autotrain_cem.lock" in text and "flock -n" in text
    assert 'logs/eval/${RUN}_cem100.json' in text
    assert 'record["success_rate"] <= 100.0' in text
    assert "mktemp" in text and 'mv "$tmp" "$STATUS"' in text
    assert "train-smoke" not in text and "sonata" not in text.lower()


def test_autotrain_cem_uses_suffix_only_for_the_training_dataset() -> None:
    text = AUTORUN.read_text(encoding="utf-8")
    assert "DATA_NAME=ogbench/franka_full/cube_franka_single_expert_pcd.rechunk20" in text
    assert 'DATA_REL="${DATA_NAME}.h5"' in text
    assert 'data.dataset.name="$DATA_REL"' in text
    assert 'eval.dataset_name="$DATA_NAME"' in text
