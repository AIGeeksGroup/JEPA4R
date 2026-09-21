import json

import h5py
import pytest

from scripts.checks.test_accepted_episode_h5 import write_toy
from scripts.data.select_fixed_ur5e_episodes import select_fixed_episodes


def test_selects_and_validates_expert_success(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "expert.h5"
    write_toy(source)

    report = select_fixed_episodes(
        source=source,
        output=output,
        family="double",
        task_id=1,
        outcome="success",
        policy="expert",
        episodes=1,
    )

    assert report["episodes"] == 1
    assert report["policy_counts"] == {"expert": 1, "random": 0}
    with h5py.File(output, "r") as handle:
        assert handle["ep_len"].shape == (1,)
        assert handle["source_policy"][0] == 1
    manifest = json.loads(output.with_suffix(".h5.manifest.json").read_text())
    assert manifest["report"] == report


def test_selects_and_validates_random_failures(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "random.h5"
    write_toy(source)

    report = select_fixed_episodes(
        source=source,
        output=output,
        family="double",
        task_id=1,
        outcome="failure",
        policy="random",
        episodes=2,
    )

    assert report["episodes"] == 2
    assert report["failures"] == 2
    assert report["policy_counts"] == {"expert": 0, "random": 2}


def test_short_selection_is_not_published(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "expert.h5"
    write_toy(source)

    with pytest.raises(RuntimeError, match="selected 1/2"):
        select_fixed_episodes(
            source=source,
            output=output,
            family="double",
            task_id=1,
            outcome="success",
            policy="expert",
            episodes=2,
        )

    assert not output.exists()
    assert not output.with_suffix(".h5.manifest.json").exists()
