import json
from pathlib import Path

import h5py
import numpy as np

from scripts.prepare_pickcube_compact_1000 import (
    build_final_manifest,
    remap_replay_episode_ids,
    select_source_batch,
)


def test_select_source_batch_preserves_original_episode_ids() -> None:
    metadata = {
        "env_info": {"env_id": "PickCube-v0"},
        "episodes": [
            {"episode_id": 10, "episode_seed": 100},
            {"episode_id": 11, "episode_seed": 101},
            {"episode_id": 12, "episode_seed": 102},
        ],
    }

    selected, source_ids_by_seed = select_source_batch(metadata, start=1, count=2)

    assert [episode["episode_id"] for episode in selected["episodes"]] == [11, 12]
    assert source_ids_by_seed == {101: 11, 102: 12}
    assert len(metadata["episodes"]) == 3


def test_remap_replay_episode_ids_uses_source_seed_mapping(tmp_path: Path) -> None:
    h5_path = tmp_path / "replay.h5"
    json_path = tmp_path / "replay.json"
    with h5py.File(h5_path, "w") as handle:
        handle.create_group("traj_0")
        handle.create_group("traj_1")
    json_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": 0, "episode_seed": 101},
                    {"episode_id": 1, "episode_seed": 102},
                ]
            }
        ),
        encoding="utf-8",
    )

    remap_replay_episode_ids(h5_path, json_path, {101: 11, 102: 12})

    with h5py.File(h5_path, "r") as handle:
        assert set(handle) == {"traj_11", "traj_12"}
    episodes = json.loads(json_path.read_text(encoding="utf-8"))["episodes"]
    assert [episode["episode_id"] for episode in episodes] == [11, 12]


def test_build_final_manifest_uses_global_train_only_statistics(tmp_path: Path) -> None:
    episode_ids = list(range(10))
    records = []
    for episode_id in episode_ids:
        frames = 2
        np.savez_compressed(
            tmp_path / f"episode_{episode_id:04d}.npz",
            qpos=np.full((frames, 9), episode_id, dtype=np.float32),
            qvel=np.zeros((frames, 9), dtype=np.float32),
            tcp_pose=np.zeros((frames, 7), dtype=np.float32),
        )
        records.append(
            {"episode_id": episode_id, "frames": frames, "actions": 1, "success": True}
        )
    source_h5 = tmp_path / "source.h5"
    source_json = tmp_path / "source.json"
    source_h5.write_bytes(b"official h5")
    source_json.write_text("{}", encoding="utf-8")
    template = {
        "task": "PickCube-v0",
        "episodes": records[:5],
        "episode_ids": episode_ids[:5],
        "train_episode_ids": episode_ids[:4],
        "val_episode_ids": episode_ids[4:5],
        "source_replay_h5": "temporary",
        "source_replay_h5_sha256": "temporary",
        "source_replay_json": "temporary",
        "source_replay_json_sha256": "temporary",
    }
    second = dict(template)
    second["episodes"] = records[5:]
    second["episode_ids"] = episode_ids[5:]

    manifest = build_final_manifest(
        tmp_path,
        [template, second],
        episode_ids,
        source_h5,
        source_json,
        seed=3072,
    )

    assert len(manifest["train_episode_ids"]) == 9
    assert len(manifest["val_episode_ids"]) == 1
    assert set(manifest["train_episode_ids"]).isdisjoint(manifest["val_episode_ids"])
    assert [record["episode_id"] for record in manifest["episodes"]] == episode_ids
    train_values = np.asarray(manifest["train_episode_ids"], dtype=np.float32)
    assert np.isclose(manifest["proprio_normalization"]["mean"][0], train_values.mean())
    assert manifest["proprio_normalization"]["sample_count"] == 18
    assert manifest["source_raw_h5_sha256"]
    assert manifest["source_raw_json_sha256"]
    assert "source_replay_h5" not in manifest


def test_lzh24_pickcube_runner_is_recoverable_and_hash_guarded() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_pickcube_compact_1000_lzh24.sh"
    assert runner.is_file()
    script = runner.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "--detach" in script
    assert "nvidia-smi --query-compute-apps" in script
    assert "a5920e22e4a9ae49e16d6c56aeb728a082d6ea9c32b62d979b6a95208f27dcd1" in script
    assert "aab69b6f89a2e6f2b6c168b7b85a97d3d0ed5ca78326b6a80484660f42d1665f" in script
    assert "compact_rgb_proprio_1000/PickCube-v0" in script
    assert "--batch-size 100" in script
    assert 'status_file="${output_root}.status"' in script
