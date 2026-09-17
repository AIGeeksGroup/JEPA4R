from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECT_ROOT = PROJECT_ROOT / "scripts" / "data" / "config"
TRAIN_ROOT = PROJECT_ROOT / "config" / "train"
EVAL_ROOT = PROJECT_ROOT / "config" / "eval"

EXPECTED_FILES = (
    COLLECT_ROOT / "ogb_franka_pix.yaml",
    COLLECT_ROOT / "ogb_franka_pcd.yaml",
    TRAIN_ROOT / "data" / "ogb_franka.yaml",
    TRAIN_ROOT / "data" / "ogb_franka_pcd.yaml",
    EVAL_ROOT / "cube_franka_pix.yaml",
    EVAL_ROOT / "cube_franka_pix_agg.yaml",
    EVAL_ROOT / "cube_franka_pcd.yaml",
    EVAL_ROOT / "cube_franka_pcd_agg.yaml",
)


@contextmanager
def _hydra_root(path: Path):
    with initialize_config_dir(version_base=None, config_dir=str(path)):
        yield


def _compose(path: Path, name: str, overrides: list[str] | None = None) -> DictConfig:
    with _hydra_root(path):
        return compose(config_name=name, overrides=overrides or [])


def test_all_parallel_franka_configs_exist() -> None:
    missing = [str(path) for path in EXPECTED_FILES if not path.is_file()]
    assert not missing, f"missing Franka configs: {missing}"


def test_collection_configs_keep_protocol_and_use_unique_franka_datasets() -> None:
    pix = _compose(COLLECT_ROOT, "ogb_franka_pix")
    pcd = _compose(COLLECT_ROOT, "ogb_franka_pcd")

    assert pix.env_id == "swm/OGBCubeFranka-v0"
    assert pcd.env_id == "swm/OGBCubeFrankaPCD-v0"
    assert pix.dataset_name == "ogbench/cube_franka_single_expert.h5"
    assert pcd.dataset_name == "ogbench/cube_franka_single_expert_pcd.h5"
    assert "rm75" not in pix.dataset_name and "rm75" not in pcd.dataset_name
    assert pix.world.max_episode_steps == pcd.world.max_episode_steps == 200
    assert pix.num_traj == pcd.num_traj == 10000
    assert pcd.pcd.num_points == 1200
    assert pcd.pcd.cube_fraction == 0.25


def test_training_configs_keep_five_step_action_blocking_and_pcd_transform() -> None:
    pix = _compose(
        TRAIN_ROOT,
        "lewm",
        ["data=ogb_franka", "model=lewm"],
    )
    pcd = _compose(
        TRAIN_ROOT,
        "lewm",
        ["data=ogb_franka_pcd", "model=lewm_pcd"],
    )

    assert pix.data.dataset.name == "ogbench/cube_franka_single_expert.h5"
    assert pcd.data.dataset.name == "ogbench/cube_franka_single_expert_pcd.h5"
    assert pix.data.dataset.frameskip == pcd.data.dataset.frameskip == 5
    assert list(pcd.data.preprocessor.center) == [0.425, 0.0, 0.2]
    assert pcd.data.preprocessor.scale == 4.0
    assert pcd.data.preprocessor.use_rgb is True


def test_eval_configs_match_collection_and_training_contracts() -> None:
    collect_pcd = _compose(COLLECT_ROOT, "ogb_franka_pcd")
    train_pcd = _compose(
        TRAIN_ROOT,
        "lewm",
        ["data=ogb_franka_pcd", "model=lewm_pcd"],
    )
    pix = _compose(EVAL_ROOT, "cube_franka_pix")
    pix_agg = _compose(EVAL_ROOT, "cube_franka_pix_agg")
    pcd = _compose(EVAL_ROOT, "cube_franka_pcd")
    pcd_agg = _compose(EVAL_ROOT, "cube_franka_pcd_agg")

    assert pix.world.env_name == "swm/OGBCubeFranka-v0"
    assert pcd.world.env_name == "swm/OGBCubeFrankaPCD-v0"
    assert pix.eval.dataset_name == "ogbench/cube_franka_single_expert"
    assert pcd.eval.dataset_name == "ogbench/cube_franka_single_expert_pcd"
    assert pix.plan_config.action_block == pcd.plan_config.action_block == 5
    assert pix.solver.n_steps == pcd.solver.n_steps == 10
    assert pix_agg.plan_config.history_len == pcd_agg.plan_config.history_len == 3
    assert pix_agg.eval.goal_window == pcd_agg.eval.goal_window == 3

    wrapper = pcd.world.pre_wrappers[0]
    assert wrapper.num_points == collect_pcd.pcd.num_points == 1200
    assert list(wrapper.cameras) == list(collect_pcd.pcd.cameras)
    assert list(map(list, wrapper.crop_box)) == list(
        map(list, collect_pcd.pcd.crop_box)
    )
    assert wrapper.foreground_only == collect_pcd.pcd.foreground_only
    assert wrapper.voxel_size == collect_pcd.pcd.voxel_size
    assert wrapper.cube_fraction == collect_pcd.pcd.cube_fraction
    assert list(pcd.pcd_process.center) == list(train_pcd.data.preprocessor.center)
    assert pcd.pcd_process.scale == train_pcd.data.preprocessor.scale
    assert pcd.pcd_process.use_rgb == train_pcd.data.preprocessor.use_rgb
