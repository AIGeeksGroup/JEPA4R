"""Coverage survives the actual upstream collection path; no simulator/GPU."""

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from ms2_lewm.coverage import CoverageWriter, episode_coverage
from ms2_lewm.environment import ManiSkillPointCloudEnv


class _FakeRL:
    """Only physics is faked; the project adapter and official runner are real."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
        self.steps = 0
        self.seed = 0
        self.unwrapped = SimpleNamespace(
            obj=SimpleNamespace(pose=SimpleNamespace(p=np.array([0., 0., .02]))),
            tcp=SimpleNamespace(pose=SimpleNamespace(p=np.array([0., 0., .03]))),
            cube_half_size=np.array([.02, .02, .02]),
            goal_pos=np.array([0., 0., .12]),
            get_state=lambda: np.array([self.steps, self.seed], dtype=np.float32),
            evaluate=self.evaluate,
            agent=SimpleNamespace(check_grasp=lambda obj: np.bool_(self.steps == 1)),
        )

    def evaluate(self):
        return {"success": np.bool_(self.steps == 1 and self.seed % 2 == 0),
                "is_obj_placed": np.bool_(self.steps == 1),
                "is_robot_static": np.bool_(self.steps == 3)}

    def observation(self):
        return {"xyz": np.full((1200, 3), self.steps / 10, np.float32),
                "rgb": np.full((1200, 3), .5, np.float32),
                "state": np.full(30, self.steps, np.float32)}

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        if seed is not None:
            self.seed = seed
        self.unwrapped.obj.pose.p[2] = .02
        return self.observation()

    def step(self, action):
        self.steps += 1
        self.unwrapped.obj.pose.p[2] = .12 if self.steps == 1 else .02
        return self.observation(), 0., False, self.steps == 3, self.evaluate()

    def close(self):
        pass


def _adapter():
    env = ManiSkillPointCloudEnv.__new__(ManiSkillPointCloudEnv)
    env.rl_env = _FakeRL()
    env.action_space = env.rl_env.action_space
    env.observation_space = gym.spaces.Dict(
        pointcloud=gym.spaces.Box(-np.inf, np.inf, (1200, 6), np.float32))
    env._episode_seed = 0
    env._episode_callback = None
    env._last_obs = None
    return env


def test_boolean_diagnostics_survive_official_pool_world_and_lance(tmp_path):
    import lance
    from stable_worldmodel.world.env_pool import EnvPool
    from stable_worldmodel.world.world import World
    from stable_worldmodel.wrapper import EverythingToInfoWrapper
    from ms2_lewm.storage import collection_writer

    world = World.__new__(World)
    world.envs = EnvPool([lambda: EverythingToInfoWrapper(_adapter())])
    world.policy = SimpleNamespace(get_action=lambda info: np.zeros((1, 7), np.float32))
    world.reset(seed=400)
    for key in ("success", "is_obj_placed", "is_robot_static", "object_grasped"):
        assert isinstance(world.infos[key], np.ndarray)
        assert world.infos[key].dtype == np.bool_
        assert world.infos[key].shape == (1, 1)
    path = tmp_path / "coverage.lance"
    report_path = tmp_path / "coverage.json"
    writer = CoverageWriter(collection_writer(path, expected_episodes=2,
                            horizon=3, reserve_bytes=0), report_path)
    try:
        world.collect(writer=writer, episodes=2, seed=400)
    finally:
        world.close()
    table = lance.dataset(str(path)).to_table()
    for key, expected in {
        "success": [True, False, False, False, False, False],
        "object_grasped": [True, False, False, True, False, False],
        "is_obj_placed": [True, False, False, True, False, False],
        "is_robot_static": [False, False, True, False, False, True],
    }.items():
        np.testing.assert_array_equal(np.asarray(table[key].to_pylist()).reshape(-1), expected)
    assert table.num_rows == 6
    assert {"pointcloud", "action", "sim_state", "observation"} <= set(table.column_names)
    report = json.loads(report_path.read_text())
    assert report["episodes_with"]["native_success"] == 1
    assert report["episodes_with"]["no_native_success"] == 1
    assert report["episodes_available"]["native_success"] == 2
    rows = [json.loads(line) for line in writer.episodes_path.read_text().splitlines()]
    assert rows[0]["episode_has"]["native_success"] is True
    assert rows[0]["final_transition_success"] is False


def _episode(**columns):
    return {"pointcloud": np.zeros((3, 1200, 6), np.float32),
            "seed": np.full(3, 2), **columns}


def test_missing_native_columns_are_unknown_but_geometry_remains_available():
    row = episode_coverage(_episode(
        object_height_above_table=[0., .1, 0.],
        object_displacement=[0., .2, 0.],
        object_goal_distance=[.2, .01, .2]))
    for key in ("grasped", "lifted", "carried", "native_success"):
        assert row["available"][key] is False
        assert row["episode_has"][key] is None
        assert row["stage_frames"][key] is None
    for key in ("no_grasp", "no_native_success", "lost_grasp_after_grasp", "final_transition_success"):
        assert row[key] is None
    for key in ("object_elevated", "object_displaced", "at_goal"):
        assert row["available"][key] is True
        assert row["episode_has"][key] is True


@pytest.mark.parametrize("values", [[0., np.nan, 0.], [False], [None, None, None]])
def test_incomplete_success_column_is_unknown(values):
    assert episode_coverage(_episode(success=values))["episode_has"]["native_success"] is None


def test_coverage_forwards_every_episode_unmodified_and_tracks_unknowns(tmp_path):
    received = []
    writer = SimpleNamespace(write_episodes=lambda episodes: received.extend(episodes))
    coverage = CoverageWriter(writer, tmp_path / "coverage.json")
    episodes = [_episode(), _episode(success=[0, 1, 0]), _episode(success=[0, 0, 0])]
    coverage.write_episodes(episodes)
    assert len(received) == len(episodes)
    assert all(actual is original for actual, original in zip(received, episodes))
    report = json.loads(coverage.report_path.read_text())
    assert report["episodes_with"]["native_success"] == 1
    assert report["episodes_with"]["no_native_success"] == 1
    assert report["episodes_available"]["native_success"] == 2
    assert report["episodes_unknown"]["native_success"] == 1
    assert report["available"]["grasped"] is False
    assert report["episodes_with"]["grasped"] is None
    assert report["episodes_with"]["no_grasp"] is None
