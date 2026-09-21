from argparse import Namespace
from pathlib import Path
import signal

import h5py
import numpy as np
import pytest

from envs.ur5e_cube_tasks import get_task
from scripts.data import collect_fixed_ur5e_task_pcd_sharded as sharded


def args_for(**updates):
    values = {
        "family": "double",
        "task_id": 1,
        "policy": "expert",
        "episodes": 10,
        "seed": 150000,
        "out": Path("unused.h5"),
        "shards": 3,
        "max_parallel": 2,
        "num_envs": 1,
        "batch_size": 20,
        "attempts_factor": 10,
        "threads": 2,
        "resume": False,
    }
    values.update(updates)
    return Namespace(**values)


def write_labeled(path, *, episodes, family="double", task_id=1, policy="expert"):
    task = get_task(family, task_id)
    lengths = np.full(episodes, 2, dtype=np.int64)
    offsets = np.arange(episodes, dtype=np.int64) * 2
    rows = episodes * 2
    success = np.zeros(rows, dtype=np.uint8)
    if policy == "expert":
        success[1::2] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        def add(name, data):
            data = np.asarray(data)
            chunks = (min(20, len(data)), *data.shape[1:])
            handle.create_dataset(
                name,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=chunks,
            )

        add("ep_len", lengths)
        add("ep_offset", offsets)
        add("success", success)
        add("qpos", np.zeros((rows, task.qpos_dim), dtype=np.float32))
        add("qvel", np.zeros((rows, task.qvel_dim), dtype=np.float32))
        add("control", np.zeros((rows, task.control_dim), dtype=np.float32))
        add("action", np.zeros((rows, task.action_dim), dtype=np.float32))
        add("observation", np.zeros((rows, task.observation_dim), dtype=np.float32))
        add("pcd_xyz", np.zeros((rows, 1200, 3), dtype=np.float32))
        add("pcd_rgb", np.zeros((rows, 1200, 3), dtype=np.uint8))
        add("source_policy", np.full(rows, policy == "expert", dtype=np.uint8))
        add("accepted_success", np.full(rows, policy == "expert", dtype=np.uint8))
        add("fixed_task_id", np.full(rows, task_id, dtype=np.uint8))
        add("cube_count", np.full(rows, task.num_cubes, dtype=np.uint8))


def test_manifest_partitions_counts_and_non_overlapping_seed_ranges(tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(args_for(), output)
    assert manifest["episode_counts"] == [4, 3, 3]
    assert manifest == sharded.make_manifest(args_for(), output)
    ranges = manifest["seed_ranges"]
    assert all(ranges[index][1] <= ranges[index + 1][0] for index in range(len(ranges) - 1))


def test_resume_rejects_changed_data_contract(tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(args_for(), output)
    with pytest.raises(ValueError, match="resume arguments"):
        sharded.assert_resume_matches(manifest, args_for(policy="random"), output)


def test_recovery_validates_complete_shards_and_preserves_work(tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(args_for(), output)
    complete = sharded.shard_path(output, 0)
    write_labeled(complete, episodes=4)
    work = sharded.shard_path(output, 1).with_name("shard_001.h5.work")
    work.mkdir(parents=True)
    assert sharded.recover_completed_shards(manifest, output) == {0}
    assert work.is_dir()


def test_recovery_rejects_invalid_complete_shard(tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(args_for(), output)
    write_labeled(sharded.shard_path(output, 0), episodes=3)
    with pytest.raises(ValueError, match="expected 4 episodes"):
        sharded.recover_completed_shards(manifest, output)


def test_stale_merge_is_removed_only_for_resume(tmp_path):
    output = tmp_path / "expert.h5"
    partial = sharded.merge_partial_path(output)
    partial.touch()
    sharded.remove_stale_merge(output, resume=False)
    assert partial.exists()
    sharded.remove_stale_merge(output, resume=True)
    assert not partial.exists()


def test_exclusive_lock_rejects_second_launcher(tmp_path):
    directory = tmp_path / "expert.h5.shards"
    directory.mkdir()
    first = sharded.acquire_lock(directory)
    try:
        with pytest.raises(RuntimeError, match="another launcher"):
            sharded.acquire_lock(directory)
    finally:
        sharded.release_lock(first)


def test_child_command_preserves_collection_contract(tmp_path):
    output = tmp_path / "random.h5"
    manifest = sharded.make_manifest(
        args_for(policy="random", episodes=4, shards=2), output
    )
    work = sharded.shard_path(output, 1).with_name("shard_001.h5.work")
    work.mkdir(parents=True)
    command = sharded.build_child_command(manifest, output, 1)
    assert command[-1] == "--resume"
    assert command[command.index("--policy") + 1] == "random"
    assert command[command.index("--num-envs") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "20"
    assert command[command.index("--episodes") + 1] == "2"


def test_scheduler_never_exceeds_max_parallel(monkeypatch, tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(
        args_for(episodes=4, shards=4, max_parallel=2), output
    )

    class Factory:
        def __init__(self):
            self.live = 0
            self.maximum_live = 0

        def __call__(self, *args, **kwargs):
            factory = self
            factory.live += 1
            factory.maximum_live = max(factory.maximum_live, factory.live)

            class Process:
                returncode = None
                polls = 0

                def poll(self):
                    self.polls += 1
                    if self.polls < 2:
                        return None
                    if self.returncode is None:
                        self.returncode = 0
                        factory.live -= 1
                    return self.returncode

                def terminate(self):
                    if self.returncode is None:
                        self.returncode = -15
                        factory.live -= 1

                def wait(self, timeout=None):
                    return self.poll()

                def kill(self):
                    self.terminate()

            return Process()

    factory = Factory()
    monkeypatch.setattr(sharded, "validate_shard", lambda *args: {})
    sharded.run_shards(
        manifest,
        output,
        completed=set(),
        popen=factory,
        sleep=lambda _: None,
    )
    assert factory.maximum_live == 2


def test_atomic_merge_validates_labeled_result(tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(
        args_for(episodes=4, shards=2), output
    )
    write_labeled(sharded.shard_path(output, 0), episodes=2)
    write_labeled(sharded.shard_path(output, 1), episodes=2)
    report = sharded.merge_shards_atomic(manifest, output)
    assert report["episodes"] == 4
    assert report["policy_counts"] == {"expert": 4, "random": 0}
    assert output.is_file()
    assert sharded.shard_path(output, 0).is_file()
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["ep_offset"][:], [0, 2, 4, 6])


def test_merge_failure_keeps_shards_and_never_publishes(monkeypatch, tmp_path):
    output = tmp_path / "expert.h5"
    manifest = sharded.make_manifest(
        args_for(episodes=4, shards=2), output
    )
    first = sharded.shard_path(output, 0)
    second = sharded.shard_path(output, 1)
    write_labeled(first, episodes=2)
    write_labeled(second, episodes=2)

    def fail_append(*args, **kwargs):
        raise RuntimeError("simulated append failure")

    monkeypatch.setattr(sharded, "append_shard", fail_append)
    with pytest.raises(RuntimeError, match="simulated append failure"):
        sharded.merge_shards_atomic(manifest, output)
    assert not output.exists()
    assert sharded.merge_partial_path(output).exists()
    assert first.exists() and second.exists()


def test_signal_handler_turns_termination_into_cleanup_path():
    with pytest.raises(KeyboardInterrupt):
        sharded.raise_interrupted(signal.SIGTERM, None)
