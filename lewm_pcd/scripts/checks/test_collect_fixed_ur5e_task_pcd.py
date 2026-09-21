import subprocess
import sys
from pathlib import Path

import pytest

from scripts.data.collect_fixed_ur5e_task_pcd import CollectionSpec, CollectorState


def test_expert_spec_forces_fixed_task_contract():
    spec = CollectionSpec.create(
        family="triple",
        task_id=4,
        policy_kind="expert",
        episodes=100,
        seed=13076,
        num_envs=5,
        batch_size=20,
        max_attempts=None,
    )
    assert spec.task_name == "cycle"
    assert spec.desired_outcome == "success"
    assert spec.max_episode_steps == 420
    assert spec.max_attempts == 1000
    assert spec.world_kwargs["env_type"] == "triple"
    assert spec.world_kwargs["mode"] == "task"
    assert spec.world_kwargs["reward_task_id"] == 4
    assert spec.world_kwargs["permute_blocks"] is False
    assert spec.world_kwargs["terminate_at_goal"] is False


def test_random_spec_accepts_only_failures():
    spec = CollectionSpec.create(
        family="double",
        task_id=2,
        policy_kind="random",
        episodes=7,
        seed=23074,
        num_envs=1,
        batch_size=3,
        max_attempts=14,
    )
    assert spec.desired_outcome == "failure"
    assert spec.max_attempts == 14


def test_state_is_resumable_only_with_identical_spec():
    spec = CollectionSpec.create(
        family="double",
        task_id=4,
        policy_kind="expert",
        episodes=10,
        seed=3072,
        num_envs=2,
        batch_size=4,
        max_attempts=30,
    )
    state = CollectorState.new(spec)
    state.record_batch(attempted=4, accepted=3, shard="accepted_000.h5")
    assert state.attempts == 4
    assert state.accepted == 3
    assert state.remaining == 7
    state.assert_compatible(spec)

    changed = CollectionSpec.create(
        family="double",
        task_id=4,
        policy_kind="expert",
        episodes=11,
        seed=3072,
        num_envs=2,
        batch_size=4,
        max_attempts=30,
    )
    with pytest.raises(ValueError, match="resume arguments"):
        state.assert_compatible(changed)


def test_state_rejects_impossible_batch_accounting():
    spec = CollectionSpec.create(
        family="double", task_id=1, policy_kind="random",
        episodes=2, seed=1, num_envs=1, batch_size=1, max_attempts=4,
    )
    state = CollectorState.new(spec)
    with pytest.raises(ValueError, match="accepted"):
        state.record_batch(attempted=1, accepted=2, shard="bad.h5")


def test_collector_script_is_directly_executable_outside_repo(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "collect_fixed_ur5e_task_pcd.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--family" in result.stdout


def test_collection_spec_uses_explicit_gpu_friendly_world_defaults():
    spec = CollectionSpec.create(
        family="double", task_id=1, policy_kind="expert",
        episodes=1, seed=1, num_envs=1, batch_size=1, max_attempts=2,
    )
    assert spec.world_kwargs["env_type"] == "double"
    assert spec.world_kwargs["mode"] == "task"


def test_candidate_world_does_not_silently_use_single_cube(monkeypatch, tmp_path):
    import scripts.data.collect_fixed_ur5e_task_pcd as collector

    captured = {}

    class FakeWorld:
        def __init__(self, env_name, **kwargs):
            captured["env_name"] = env_name
            captured.update(kwargs)

        def set_policy(self, policy):
            captured["policy"] = policy

        def collect(self, **kwargs):
            captured["collect"] = kwargs

        class Envs:
            def close(self):
                pass

        envs = Envs()

    monkeypatch.setattr(collector.swm, "World", FakeWorld)
    spec = collector.CollectionSpec.create(
        family="triple", task_id=4, policy_kind="random",
        episodes=1, seed=1, num_envs=1, batch_size=1, max_attempts=2,
    )
    collector._collect_candidate(spec, tmp_path / "candidate.h5", 1, 1)
    assert captured["env_name"] == "swm/OGBCubePCD-v0"
    assert captured["env_type"] == "triple"
    assert captured["mode"] == "task"
