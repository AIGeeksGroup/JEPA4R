from pathlib import Path

from hydra import compose, initialize_config_dir


def test_pickcube_config_keeps_original_ms2_pointnet_baseline():
    config_dir = Path(__file__).resolve().parents[2] / "config" / "train"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="lewm",
            overrides=["data=maniskill_pickcube_pcd", "model=lewm_pcd"],
        )

    assert cfg.data.dataset.name == "maniskill/pickcube_pcd_train.h5"
    assert list(cfg.data.dataset.keys_to_load) == ["pcd_xyz", "pcd_rgb", "action"]
    assert cfg.model.encoder._target_ == "encoders.ManiSkillPointNet"
    assert cfg.model.obs_key == "pcd"
    assert list(cfg.data.preprocessor.center) == [-0.15, 0.0, 0.375]
    assert cfg.data.preprocessor.scale == 1.5


def test_training_seeds_rng_before_model_and_dataset_construction():
    source = (Path(__file__).resolve().parents[2] / "train.py").read_text()

    seed_pos = source.index("pl.seed_everything(cfg.seed, workers=True)")
    dataset_pos = source.index("dataset_cfg =")
    model_pos = source.index("world_model = hydra.utils.instantiate(cfg.model)")

    assert seed_pos < dataset_pos < model_pos
