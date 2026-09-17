from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data import collect_cube_pcd_sharded as sharded


def _args(**updates):
    values = {
        "shards": 3,
        "num_traj": 7,
        "seed": 3072,
        "script": "collect_cube_pcd.py",
        "threads": 2,
        "max_parallel": None,
        "no_merge": True,
        "overrides": ["env_type=single"],
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _write_shard(path: Path, lens=(2, 3), offsets=None, column_rows=None) -> None:
    lens = np.asarray(lens, dtype=np.int32)
    rows = int(lens.sum()) if column_rows is None else column_rows
    if offsets is None:
        offsets = np.concatenate(([0], np.cumsum(lens)[:-1]))
    with h5py.File(path, "w") as f:
        f["ep_len"] = lens
        f["ep_offset"] = np.asarray(offsets, dtype=np.int64)
        f.create_dataset("state", data=np.zeros((rows, 4), dtype=np.float32))
        f.create_dataset("nested/action", data=np.zeros((rows, 2), dtype=np.float32))


def test_manifest_creation_is_deterministic_and_persists_fixed_seeds(tmp_path: Path) -> None:
    args = _args()
    final = tmp_path / "datasets" / "cube.h5"

    first = sharded.make_manifest(args, final)
    second = sharded.make_manifest(args, final)

    assert first == second
    assert first["episode_counts"] == [3, 2, 2]
    assert len(first["world_seeds"]) == args.shards
    manifest_path = final.parent / "shards" / "collection_manifest.json"
    sharded.write_manifest_atomic(manifest_path, first)
    assert sharded.load_manifest(manifest_path) == first


def test_resume_rejects_requested_argument_mismatch(tmp_path: Path) -> None:
    final = tmp_path / "datasets" / "cube.h5"
    manifest = sharded.make_manifest(_args(), final)

    with pytest.raises(ValueError, match="resume arguments do not match"):
        sharded.assert_resume_matches(manifest, _args(seed=99), final)


@pytest.mark.parametrize("seed_values", [[100, 101, 102], ["bad", 200, 300]])
def test_resume_rejects_corrupt_or_overlapping_persisted_seeds(
    tmp_path: Path, seed_values
) -> None:
    final = tmp_path / "datasets" / "cube.h5"
    manifest = copy.deepcopy(sharded.make_manifest(_args(), final))
    manifest["world_seeds"] = seed_values

    with pytest.raises(ValueError, match="invalid collection manifest"):
        sharded.assert_resume_matches(manifest, _args(), final)


def test_resume_skips_valid_shards_promotes_valid_partials_and_cleans_bad_partials(
    tmp_path: Path,
) -> None:
    final = tmp_path / "datasets" / "cube.h5"
    shard_dir = final.parent / "shards"
    shard_dir.mkdir(parents=True)
    manifest = sharded.make_manifest(_args(), final)
    sharded.write_manifest_atomic(shard_dir / "collection_manifest.json", manifest)

    complete = sharded.shard_path(final, 0)
    valid_partial = sharded.partial_shard_path(final, 1)
    bad_partial = sharded.partial_shard_path(final, 2)
    _write_shard(complete, lens=(1, 1, 1))
    _write_shard(valid_partial, lens=(2, 3))
    _write_shard(bad_partial, lens=(2, 2), column_rows=3)

    completed = sharded.recover_shards(manifest, final, shard_dir)

    assert completed == {0, 1}
    assert complete.exists()
    assert sharded.shard_path(final, 1).exists()
    assert not valid_partial.exists()
    assert not bad_partial.exists()


def test_partial_cleanup_refuses_paths_outside_the_shard_directory(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    outside = tmp_path / "elsewhere.partial.h5"
    outside.write_bytes(b"keep")

    with pytest.raises(ValueError, match="outside shard directory"):
        sharded.remove_incomplete_partial(outside, shard_dir)

    assert outside.exists()


def test_h5_validation_requires_episode_counts_offsets_and_column_lengths(tmp_path: Path) -> None:
    valid = tmp_path / "valid.h5"
    _write_shard(valid)
    assert sharded.validate_shard(valid, expected_episodes=2) == (2, 5)

    wrong_count = tmp_path / "wrong_count.h5"
    _write_shard(wrong_count, lens=(5,))
    with pytest.raises(ValueError, match="expected 2 episodes"):
        sharded.validate_shard(wrong_count, expected_episodes=2)

    bad_offsets = tmp_path / "bad_offsets.h5"
    _write_shard(bad_offsets, offsets=(0, 4))
    with pytest.raises(ValueError, match="contiguous ep_offset"):
        sharded.validate_shard(bad_offsets, expected_episodes=2)

    bad_column = tmp_path / "bad_column.h5"
    _write_shard(bad_column, column_rows=4)
    with pytest.raises(ValueError, match="has 4 rows; expected 5"):
        sharded.validate_shard(bad_column, expected_episodes=2)


def test_h5_validation_rejects_scalar_columns_and_noninteger_episode_arrays(tmp_path: Path) -> None:
    scalar = tmp_path / "scalar.h5"
    _write_shard(scalar)
    with h5py.File(scalar, "a") as f:
        f["metadata"] = 1
    with pytest.raises(ValueError, match="scalar column"):
        sharded.validate_shard(scalar, expected_episodes=2)

    floating_lengths = tmp_path / "floating_lengths.h5"
    _write_shard(floating_lengths)
    with h5py.File(floating_lengths, "a") as f:
        del f["ep_len"]
        f["ep_len"] = np.array([2.0, 3.0])
    with pytest.raises(ValueError, match="ep_len must be an integer"):
        sharded.validate_shard(floating_lengths, expected_episodes=2)


def test_parser_rejects_nonpositive_parallel_collection_arguments() -> None:
    parser = sharded.build_parser()
    for flag in ("--shards", "--num-traj", "--threads", "--max-parallel"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--shards", "1", flag, "0"])


def test_exclusive_shard_lock_rejects_concurrent_launcher(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    first = sharded.acquire_shard_lock(shard_dir)
    try:
        with pytest.raises(RuntimeError, match="already holds the shard lock"):
            sharded.acquire_shard_lock(shard_dir)
    finally:
        sharded.release_shard_lock(first)


def test_merge_interruption_keeps_completed_shards_and_no_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "datasets" / "cube.h5"
    first = tmp_path / "cube.shard000.h5"
    second = tmp_path / "cube.shard001.h5"
    _write_shard(first, lens=(2,))
    _write_shard(second, lens=(3,))

    def interrupted(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "merge_h5_shards.py")

    monkeypatch.setattr(sharded.subprocess, "run", interrupted)
    with pytest.raises(subprocess.CalledProcessError):
        sharded.merge_shards_atomic(final, [first, second], expected_episodes=2, env={})

    assert first.exists()
    assert second.exists()
    assert not final.exists()
    assert sharded.merge_partial_path(final).exists()
