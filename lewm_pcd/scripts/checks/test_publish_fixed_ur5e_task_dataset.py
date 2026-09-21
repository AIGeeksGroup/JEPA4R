import h5py
import numpy as np
import pytest

from scripts.data.accepted_episode_h5 import copy_accepted_episodes
from scripts.data.publish_fixed_ur5e_task_dataset import publish_task_datasets


def write_source(path):
    lengths = np.array([3, 2, 4], dtype=np.int64)
    offsets = np.array([0, 3, 5], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f["ep_len"] = lengths
        f["ep_offset"] = offsets
        f["success"] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.uint8)
        f["action"] = np.zeros((9, 5), dtype=np.float32)
        f["qpos"] = np.zeros((9, 28), dtype=np.float32)
        f["qvel"] = np.zeros((9, 26), dtype=np.float32)
        f["control"] = np.zeros((9, 7), dtype=np.float32)
        f["observation"] = np.zeros((9, 37), dtype=np.float32)
        f["pcd_xyz"] = np.zeros((9, 1200, 3), dtype=np.float32)
        f["pcd_rgb"] = np.zeros((9, 1200, 3), dtype=np.uint8)


def test_publish_merges_expert_and_random_but_keeps_holdout_separate(tmp_path):
    source = tmp_path / "source.h5"
    expert = tmp_path / "expert.h5"
    random = tmp_path / "random.h5"
    holdout = tmp_path / "holdout.h5"
    train = tmp_path / "train.h5"
    write_source(source)
    copy_accepted_episodes(
        source, expert, outcome="success", limit=1,
        family="double", task_id=1, policy_kind="expert"
    )
    copy_accepted_episodes(
        source, random, outcome="failure", limit=2,
        family="double", task_id=1, policy_kind="random"
    )
    copy_accepted_episodes(
        source, holdout, outcome="success", limit=1,
        family="double", task_id=1, policy_kind="expert"
    )

    report = publish_task_datasets(
        expert_train=expert,
        random_failures=random,
        holdout=holdout,
        train_out=train,
        family="double",
        task_id=1,
        expert_train_episodes=1,
        random_failure_episodes=2,
        holdout_episodes=1,
    )
    assert report["train"]["episodes"] == 3
    assert report["train"]["policy_counts"] == {"expert": 1, "random": 2}
    assert report["holdout"]["episodes"] == 1
    assert train.exists()
    assert holdout.exists()


def test_publish_refuses_overwrite(tmp_path):
    train = tmp_path / "train.h5"
    train.touch()
    with pytest.raises(FileExistsError):
        publish_task_datasets(
            expert_train=tmp_path / "missing-expert.h5",
            random_failures=tmp_path / "missing-random.h5",
            holdout=tmp_path / "missing-holdout.h5",
            train_out=train,
            family="double",
            task_id=1,
            expert_train_episodes=1,
            random_failure_episodes=1,
            holdout_episodes=1,
        )
