import numpy as np

from bench.cortexbench.adroit.smoke import (
    AdroitPairedWriter,
    EpisodePlan,
    draw_episode_plan,
    normalized_action,
)


def test_draw_episode_plan_is_seed_reproducible_and_bounded():
    first = draw_episode_plan(np.random.default_rng(19), episode_len=100)
    second = draw_episode_plan(np.random.default_rng(19), episode_len=100)
    assert first == second
    assert first.sample_type in (0, 1, 2)
    assert 0.0 <= first.noise_std <= 0.2
    if first.sample_type == 1:
        assert 15 <= first.branch_t <= 60
        assert first.branch_len == 100 or 10 <= first.branch_len <= 30
    else:
        assert first.branch_t == -1
        assert first.branch_len == 0


def test_normalized_action_is_finite_and_in_bounds():
    raw = np.linspace(-20.0, 20.0, 24)
    action = normalized_action(raw)
    assert action.shape == (24,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_paired_writer_split_keeps_visual_columns_separate():
    episode = {
        "pixels": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "pcd_xyz": np.zeros((2, 1200, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((2, 1200, 3), dtype=np.uint8),
        "action": np.zeros((2, 24), dtype=np.float32),
        "sample_type": np.zeros(2, dtype=np.int64),
    }
    pix, pcd = AdroitPairedWriter.split_episode(episode)
    assert "pixels" in pix and "pixels" not in pcd
    assert "pcd_xyz" not in pix and "pcd_xyz" in pcd
    assert "pcd_rgb" not in pix and "pcd_rgb" in pcd
    assert np.array_equal(pix["action"], pcd["action"])
