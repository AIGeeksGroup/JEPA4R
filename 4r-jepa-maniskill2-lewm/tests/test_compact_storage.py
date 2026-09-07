"""Real official readers and CPU model, with explicitly synthetic episodes."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from ms2_lewm.model import WorldModel
from ms2_lewm.official import load_source
from ms2_lewm.storage import CompactLanceWriter, StorageBudget, collection_writer, compact_episode
from stable_worldmodel.data import load_dataset
from stable_worldmodel.data.formats.lance import LanceWriter
from stable_worldmodel.world.world import _extract_init_goal


ROOT = Path(__file__).resolve().parents[1]
eval_driver = load_source("_compact_storage_eval", ROOT / "scripts/evaluate_wm.py")


def synthetic_episode(seed=4, length=8):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (length, 1200, 6)).astype(np.float32)
    points[..., 3:] = rng.uniform(0, 1, (length, 1200, 3))
    state = rng.normal(size=(length, 30)).astype(np.float32)
    return {
        "pointcloud": points, "rl_xyz": points[..., :3], "rl_rgb": points[..., 3:],
        "observation": state, "rl_state": state, "proprio": state,
        "action": rng.normal(size=(length, 7)).astype(np.float32),
        "pixels": rng.integers(0, 256, (length, 16, 16, 3), dtype=np.uint8),
        "sim_state": rng.normal(size=(length, 12)).astype(np.float32),
        "cube_pos": rng.normal(size=(length, 3)).astype(np.float32),
        "seed": np.full(length, seed, dtype=np.int64),
    }


@pytest.mark.parametrize("suffix", [".lance", ".h5"])
def test_compact_data_roundtrip_gradients_and_eval_boundary(tmp_path, suffix):
    path = tmp_path / f"SYNTHETIC_UNIT_TEST_ONLY{suffix}"
    episodes = [synthetic_episode(), synthetic_episode(5)]
    with collection_writer(path, expected_episodes=2, horizon=7, reserve_bytes=0) as writer:
        writer.write_episodes(episodes)
    dataset = load_dataset(str(path), num_steps=4, frameskip=1)
    assert set(dataset.column_names).isdisjoint({"rl_xyz", "rl_rgb", "rl_state", "proprio"})
    assert dataset.lengths.tolist() == [8, 8]
    row = dataset[0]
    assert row["pointcloud"].dtype == torch.float16
    restored = row["pointcloud"].reshape(4, 1200, 6).float().numpy()
    np.testing.assert_array_equal(restored, episodes[0]["pointcloud"][:4].astype(np.float16).astype(np.float32))
    assert float(np.abs(restored - episodes[0]["pointcloud"][:4]).max()) <= 2 ** -12
    np.testing.assert_array_equal(row["action"], episodes[0]["action"][:4])
    np.testing.assert_array_equal(row["observation"], episodes[0]["observation"][:4])
    assert row["pixels"].shape == (4, 3, 16, 16)
    assert not path.with_suffix(".h5" if suffix == ".lance" else ".lance").exists()
    if suffix == ".h5":
        with h5py.File(path) as handle:
            assert handle["pointcloud"].compression == "gzip"
            assert handle["pointcloud"].dtype == np.float16
    model = WorldModel().eval()
    batch = {name: torch.stack([dataset[0][name], dataset[5][name]]) for name in ("pointcloud", "action")}
    seen = []
    hook = model.frame_encoder.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].dtype))
    output = model(batch)
    output["loss"].backward()
    hook.remove()
    assert seen == [torch.float32]
    assert bool(torch.isfinite(output["loss"]))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.frame_encoder.parameters())
    # The untouched evaluation extractor retains all replay columns and the
    # same episode/step indices, using either official storage reader.
    eval_dataset = eval_driver.load_evaluation_dataset(path)
    initial, goals, videos = _extract_init_goal(eval_dataset, [0], [2], 3)
    assert initial["step_idx"].item() == 2
    assert goals["goal_step_idx"].item() == 5
    np.testing.assert_array_equal(initial["sim_state"][0], episodes[0]["sim_state"][2])
    assert goals["goal_pointcloud"].dtype == np.float16
    assert len(videos) == 1
    if getattr(dataset, "h5_file", None) is not None:
        dataset.h5_file.close()
    if getattr(eval_dataset, "h5_file", None) is not None:
        eval_dataset.h5_file.close()


def test_lance_images_match_official_jpeg_and_storage_half_dtype(tmp_path):
    episode = synthetic_episode()
    normal, compact = tmp_path / "official.lance", tmp_path / "compact.lance"
    with LanceWriter(normal, mode="error") as writer:
        writer.write_episode({name: list(values) for name, values in episode.items()})
    with CompactLanceWriter(compact, reserve_bytes=0) as writer:
        writer.write_episode(episode)
    left = load_dataset(str(normal), keys_to_load=["pixels"])
    right = load_dataset(str(compact), keys_to_load=["pixels"])
    torch.testing.assert_close(left[0]["pixels"], right[0]["pixels"], rtol=0, atol=0)


def test_dedup_does_not_discard_distinct_data_or_mutate_live_points():
    episode = synthetic_episode()
    original = episode["pointcloud"].copy()
    compact_episode(episode)
    np.testing.assert_array_equal(episode["pointcloud"], original)
    assert episode["pointcloud"].dtype == np.float32
    episode["rl_xyz"] = episode["rl_xyz"].copy() + 1
    with pytest.raises(ValueError, match="non-identical alias"):
        compact_episode(episode)


def test_storage_admission_uses_probe_then_measured_remaining_bytes(monkeypatch, tmp_path):
    import ms2_lewm.storage as storage
    free = [10_000_000]
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=free[0]))
    budget = StorageBudget(tmp_path / "large.lance", expected_episodes=10_000, horizon=200, reserve_bytes=1_000)
    report = budget.check(nominal=True)
    assert report["passed"]  # No fake 28.8GB compressed-data lower bound.
    assert report["estimated_remaining_bytes"] == 201 * 1200 * 6 * 2
    with pytest.raises(RuntimeError, match="Insufficient dataset disk space"):
        budget.check(completed_steps=201, episode_bytes=201_000, episode_steps=201)
    budget = StorageBudget(tmp_path / "small.lance", expected_episodes=2, horizon=9, reserve_bytes=1000)
    free[0] = 3_000
    report = budget.check(completed_steps=10, episode_bytes=1000, episode_steps=10)
    assert report["estimated_remaining_bytes"] == 1200  # Remaining 10, not all 20 steps.
    assert report["required_free_bytes"] == 2200
    free[0] = 999
    with pytest.raises(RuntimeError, match="Insufficient dataset disk space"):
        CompactLanceWriter(tmp_path / "forbidden.lance", reserve_bytes=1000)
    assert not (tmp_path / "forbidden.lance").exists()


def test_eval_accepts_flat_half_initial_frame_then_shaped_live_frames():
    import gymnasium as gym
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.solver import CEMSolver

    class Cost(torch.nn.Module):
        history_size = 3
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        def get_cost(self, info, actions):
            assert info["pointcloud"].dtype == info["goal_pointcloud"].dtype == torch.float32
            assert info["pointcloud"].shape[-2:] == (1200, 6)
            return actions.square().sum((-2, -1))

    solver = CEMSolver(model=eval_driver.adapt_evaluation_cost(Cost()), num_samples=2,
                       n_steps=1, topk=2, device="cpu", seed=31)
    policy = eval_driver.make_consecutive_policy(solver=solver,
        config=PlanConfig(horizon=3, receding_horizon=3, action_block=1, history_len=3),
        dataset_start_steps=[7])
    policy.set_env(SimpleNamespace(num_envs=1,
        action_space=gym.spaces.Box(-1, 1, (1, 7), np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (7,), np.float32)))
    previous = np.full((1, 1, 7), np.nan, dtype=np.float32)
    for step in range(4):
        current = np.full((1, 1, 1200, 6), step, dtype=np.float32)
        if step == 0:
            current = current.reshape(1, 1, 7200).astype(np.float16)
        info = {"pointcloud": current, "goal_pointcloud": np.zeros((1, 1, 7200), np.float16),
                "action": previous, "step_idx": np.asarray([[7 if step == 0 else step]]),
                "terminated": np.zeros(1, dtype=bool)}
        previous = policy.get_action(info)[:, None]
    assert [float(frame[0, 0]) for frame in policy._point_history[0]] == [1., 2., 3.]
