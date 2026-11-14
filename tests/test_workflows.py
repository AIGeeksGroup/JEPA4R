"""The Cube 1-4 and Relocate workflows without simulators or GPUs: their stage plans and the paper budgets in them,
seeds, input widths, and the data conversions they run on small synthetic inputs."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def relocate_args(output, model="jepa4r", env="adroit-relocate"):
    return SimpleNamespace(env=env, model=model, seed=42, output=output, wm_python="wm-python", sim_python="sim-python")


def relocate_assets():
    return dict(
        rh56_expert_root="/bundle",
        eai_vc_root="/eai-vc",
        rh56_official_task_init="/official/rh56_official_task_init.json",
        adroit_expert="/expert.pickle",
        adroit_official_demos="/official/demos",
    )


def value_after(command, flag):
    return command[command.index(flag) + 1]


class CubeWorkflowTests(unittest.TestCase):
    def test_encoder_widths_and_policy_inputs(self):
        from jepa4r.policy.gciql.workflow_downstream import group_contract

        widths = {"lewm": 192, "jepa4r": 192, "pldm": 512, "lejepa": 512, "visreg": 768, "vjepa": 1024}
        for model, width in widths.items():
            for cube in (1, 2, 3, 4):
                contract = group_contract(model, cube)["input_contract"]
                self.assertEqual(contract["latent_dim"], width)
                self.assertEqual(contract["current_dim"], width + 19)
                self.assertEqual(contract["goal_dim"], width + 3 * cube)

    def test_gciql_protocol_is_the_paper_budget(self):
        from jepa4r.policy.gciql.workflow_downstream import PROTOCOL

        self.assertEqual(PROTOCOL["batch_size"], 1024)
        self.assertEqual(PROTOCOL["total_updates"], 1000000)
        self.assertEqual(PROTOCOL["eval_tasks"], (1, 2, 3, 4, 5))
        self.assertEqual(PROTOCOL["eval_episodes_per_task"], 50)

    def test_collection_plan_counts_horizons_and_seeds(self):
        from jepa4r.data.cube.collection import build_plan

        with tempfile.TemporaryDirectory() as tmp:
            for cube, horizon in enumerate((200, 500, 1000, 600), 1):
                plan = build_plan(cube, Path(tmp) / str(cube))
                self.assertEqual(plan["raw_horizon"], horizon)
                self.assertEqual(plan["split_counts"], {"train": 10000, "validation": 1000})
                self.assertEqual(len(plan["parts"]), 125)
                self.assertTrue(all(part["episodes"] <= 100 for part in plan["parts"]))
                first, second = plan["parts"][:2]
                self.assertEqual(first["world_seed"], 1000000 * cube)
                self.assertEqual(first["policy_seed"], 100000000 + 1000000 * cube)
                self.assertEqual(second["policy_seed"] - first["policy_seed"], 17)
                self.assertEqual(len({part["world_seed"] for part in plan["parts"]}), 125)
                self.assertFalse((Path(tmp) / str(cube)).exists())

    def test_controller_stages_by_model(self):
        from jepa4r.policy.gciql import workflow

        def stages(model):
            config = dict(
                cube_count=1,
                model=model,
                output_root="/run",
                python=dict(world_model="/wm/python", downstream="/rl/python"),
                official_root="/official",
                collection_root="/collected",
                data_root="/views",
                assets_root="/checkpoints",
                workers=2,
                prefetch=2,
                collection_seed=0,
            )
            return [row["stage"] for row in workflow.build_plan(config, "all")["stages"]]

        self.assertNotIn("prepare-encoder", stages("jepa4r"))
        self.assertIn("train", stages("jepa4r"))
        self.assertNotIn("train", stages("visreg"))
        self.assertIn("prepare-encoder", stages("visreg"))
        self.assertEqual(stages("lewm")[-3:], ["encode", "gciql", "evaluate"])

    def test_play_data_download_resumes_a_partial_file(self):
        from unittest import mock

        from jepa4r.data.cube import official_data

        payload = bytes(range(256)) * 4
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "play.npz.part").write_bytes(payload[:100])
            requests = []

            class Response:
                status = 206

                def __init__(self, data):
                    self.data = data

                def read(self, size):
                    block, self.data = self.data[:size], self.data[size:]
                    return block

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            def urlopen(request, timeout):
                requests.append(request.headers.get("Range"))
                return Response(payload[100:])

            item = dict(name="play.npz", url="https://example.invalid/play.npz", bytes=len(payload), episodes=1)
            with mock.patch.object(official_data.urllib.request, "urlopen", urlopen):
                record = official_data.download(item, tmp)
            self.assertEqual(requests, ["bytes=100-"])
            self.assertEqual(Path(record["path"]).read_bytes(), payload)

    def test_collection_parts_join_into_one_virtual_dataset(self):
        from jepa4r.data.cube.collection import merge_raw

        with tempfile.TemporaryDirectory() as tmp:
            parts = []
            for index, lengths in enumerate(([2, 3], [4])):
                path = Path(tmp) / f"part_{index}.h5"
                with h5py.File(path, "w") as f:
                    f["obs/state"] = np.full((sum(lengths), 2), index, np.float32)
                    f["ep_len"] = np.array(lengths)
                    f["ep_offset"] = np.r_[0, np.cumsum(lengths)[:-1]]
                parts.append(path)
            merge_raw(parts, Path(tmp) / "joined.h5")
            with h5py.File(Path(tmp) / "joined.h5", "r") as f:
                np.testing.assert_array_equal(f["obs/state"][:, 0], [0] * 5 + [1] * 4)
                np.testing.assert_array_equal(f["ep_len"][:], [2, 3, 4])
                np.testing.assert_array_equal(f["ep_offset"][:], [0, 2, 5])


class RelocateWorkflowTests(unittest.TestCase):
    def test_adroit_plan_uses_the_paper_budgets(self):
        from jepa4r.policy.bc.relocate import adroit_workflow

        stages = {s["name"]: s["command"] for s in adroit_workflow.plan(relocate_args(Path("/run")), relocate_assets())}
        self.assertEqual(
            list(stages), ["collect", "wm-train", "official-demos", "bc-seed100", "bc-seed200", "bc-seed300"]
        )
        self.assertEqual(value_after(stages["collect"], "--train-episodes"), "10000")
        self.assertEqual(value_after(stages["collect"], "--test-episodes"), "150")
        self.assertEqual(value_after(stages["official-demos"], "--count"), "100")
        bc = stages["bc-seed200"]
        self.assertEqual(value_after(bc, "--epoch"), "10")
        for override in ("seed=200", "num_demos=100", "epochs=100", "eval_num_traj=25", "bc_kwargs.batch_size=256"):
            self.assertIn(override, bc)

    def test_adroit_episode_plan_is_the_env_feedback_mixture(self):
        import ast

        from jepa4r.data.relocate import adroit_collect

        tree = ast.parse((REPO / "jepa4r/envs/adroit/env.py").read_text(encoding="utf-8"))
        env = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AdroitEnv")
        draw = next(n for n in env.body if isinstance(n, ast.FunctionDef) and n.name == "_draw_plan")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[draw], type_ignores=[]), "env.py", "exec"), namespace)
        settings = SimpleNamespace(
            force_type=None,
            mix=np.array([0.5, 0.3, 0.2]),
            noise_max=0.2,
            episode_len=200,
            branch_tail_prob=0.5,
            burst_len=(10, 30),
        )
        left, right = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(200):
            self.assertEqual(adroit_collect.episode_plan(left), namespace["_draw_plan"](settings, right))

    def test_adroit_expert_noise_precedes_the_single_clip(self):
        from jepa4r.data.relocate import adroit_collect

        policy = SimpleNamespace(get_action=lambda obs: (None, {"evaluation": np.full(30, 1.2, np.float32)}))
        env = SimpleNamespace(get_obs=lambda: np.zeros(39, np.float32))
        rng = SimpleNamespace(normal=lambda mean, std, size: np.full(size, -0.4))
        action = adroit_collect.mixed_action(policy, env, rng, noise=0.2, random_action=False)
        np.testing.assert_allclose(action, np.full(30, 0.8, np.float32), rtol=0, atol=1e-7)

    def test_released_encoders_skip_world_model_training(self):
        from jepa4r.policy.bc.relocate import adroit_workflow

        names = [s["name"] for s in adroit_workflow.plan(relocate_args(Path("/run"), "visreg"), relocate_assets())]
        self.assertNotIn("wm-train", names)

    def test_rh56_plan_batches_seeds_and_budgets(self):
        from jepa4r.policy.bc.relocate import rh56_workflow

        batches = rh56_workflow._episode_batches()
        episodes = [row for batch in batches for row in batch]
        self.assertEqual(len(batches), 40)
        self.assertEqual(sum(row["split"] == "train" for row in episodes), 9000)
        self.assertEqual(episodes[0]["seed"], 61000000)
        self.assertEqual([row["seed"] for row in episodes if row["split"] == "val"][0], 62000000)
        args = relocate_args(Path("/run"), env="rh56-relocate")
        stages = {s["name"]: s["command"] for s in rh56_workflow.plan(args, relocate_assets())}
        self.assertEqual(sum(name.startswith("rh56_collect_") for name in stages), 40)
        self.assertEqual(value_after(stages["rh56_wm"], "--seed"), "3072")
        self.assertEqual(value_after(stages["rh56_official_demo"], "--count"), "100")
        self.assertEqual(value_after(stages["rh56_official_demo"], "--seed-base"), "64500000")
        self.assertEqual([n for n in stages if n.startswith("rh56_bc_")], ["rh56_bc_100", "rh56_bc_200", "rh56_bc_300"])
        self.assertEqual(value_after(stages["rh56_bc_300"], "--epoch"), "10")

    def test_rh56_bc_budget_and_evaluation_seeds(self):
        from jepa4r.envs.twins.rh56 import bc_bridge

        self.assertEqual(
            bc_bridge.RUN_SPEC,
            dict(epochs=100, num_demos=100, eval_num_traj=25, eval_frequency=5, save_frequency=10, batch_size=256),
        )
        mapping = bc_bridge.EvaluationSeedMap(200, bc_bridge.evaluation_seed_base(200), 25)
        self.assertEqual(mapping(200), 65200000)
        self.assertEqual(mapping(224), 65200024)
        with self.assertRaises(ValueError):
            mapping(225)

    def test_rh56_entry_dispatches_stages_to_their_modules(self):
        import contextlib
        import io

        from jepa4r.policy.bc.relocate import run_rh56

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_rh56.main(["wm", "--route", "pcd", "--dry-run", "--manifest", "m.json"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["module"], "jepa4r.envs.twins.rh56.wm_bridge")
        self.assertEqual(plan["argv"][:4], ["--manifest", "m.json", "--route", "pcd"])
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                run_rh56.main(["bc", "--dry-run"])

    def test_rh56_demo_layout(self):
        from jepa4r.envs.twins.rh56.bc_bridge import prepare_demo

        rng = np.random.default_rng(0)
        arrays = dict(
            observations=rng.normal(size=(201, 56)),
            proprio=rng.normal(size=(201, 13)),
            actions=rng.uniform(-1, 1, size=(200, 12)),
            rewards=rng.normal(size=200),
            initial_state_json=np.array(json.dumps({"seed": 64500000})),
            pixels=rng.integers(0, 255, size=(201, 4, 4, 3), dtype=np.uint8),
            xyz=rng.normal(size=(201, 5, 3)).astype(np.float32),
            rgb=rng.integers(0, 255, size=(201, 5, 3)).astype(np.float32),
        )
        rgb, pcd = prepare_demo(arrays, "rgb"), prepare_demo(arrays, "pcd")
        np.testing.assert_array_equal(rgb["images"], arrays["pixels"][:-1])
        self.assertEqual(pcd["images"].shape, (200, 5, 6))
        np.testing.assert_array_equal(rgb["observations"], arrays["observations"][:-1])
        np.testing.assert_array_equal(rgb["env_infos"]["proprio"], arrays["proprio"][:-1])
        self.assertEqual(rgb["init_state_dict"], {"seed": 64500000})

    def test_rh56_episode_split_and_train_rows(self):
        from jepa4r.envs.twins.rh56.wm_bridge import EpisodePartition, TrainColumnView

        dataset = SimpleNamespace(
            lengths=[3, 2, 4],
            offsets=[0, 3, 5],
            clip_indices=[(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)],
            get_col_data=lambda col: np.arange(9 * 2).reshape(9, 2),
        )
        manifest = dict(
            episodes=[
                dict(episode_index=0, split="train"),
                dict(episode_index=1, split="val"),
                dict(episode_index=2, split="train"),
            ]
        )
        partition = EpisodePartition(dataset, manifest)
        self.assertEqual(partition.clip_indices, {"train": [0, 1, 3, 4, 5], "val": [2]})
        rows = TrainColumnView(dataset, partition).get_col_data("action")
        np.testing.assert_array_equal(rows, np.r_[np.arange(0, 6), np.arange(10, 18)].reshape(7, 2))

    def test_rh56_publication_is_a_virtual_dataset(self):
        from jepa4r.envs.twins.rh56.data_tools import publish_dataset

        with tempfile.TemporaryDirectory() as tmp:
            batches = []
            for index, length in enumerate((3, 2)):
                batch = Path(tmp) / f"batch_{index}"
                batch.mkdir()
                for route in ("rgb", "pcd"):
                    with h5py.File(batch / f"paired_{route}.h5", "w") as f:
                        f["action"] = np.full((length, 2), index, np.float32)
                        f["ep_len"] = np.array([length], np.int32)
                        f["ep_offset"] = np.array([0], np.int64)
                record = dict(seed=index, split="train", episode_index=0, npz=f"train_{index}.npz", rows=length)
                (batch / "manifest.json").write_text(json.dumps(dict(episodes=[record])))
                batches.append(batch)
            manifest = publish_dataset(batches, Path(tmp) / "data")
            self.assertEqual([row["episode_index"] for row in manifest["episodes"]], [0, 1])
            with h5py.File(Path(tmp) / "data/paired_rgb.h5", "r") as f:
                np.testing.assert_array_equal(f["action"][:, 0], [0, 0, 0, 1, 1])
                np.testing.assert_array_equal(f["ep_offset"][:], [0, 3])

    def test_official_replays_keep_every_other_field(self):
        from jepa4r.data.relocate.adroit_data import rgb_demo
        from jepa4r.data.relocate.convert_adroit_demos_3d import convert_demo

        class Env:
            def __init__(self):
                self.calls = []

            def set_env_state(self, state):
                self.calls.append(("set", state["qpos"]))

            def step(self, action):
                self.calls.append(("step", float(action[0])))

        demo = dict(
            actions=np.arange(3 * 30, dtype=np.float64).reshape(3, 30), init_state_dict=dict(qpos=7), extra="kept"
        )
        env = Env()
        rgb = rgb_demo(demo, env, lambda: np.zeros((2, 2, 3), np.uint8))
        self.assertEqual(rgb["images"].shape, (3, 2, 2, 3))
        self.assertEqual(env.calls, [("set", 7), ("step", 0.0), ("step", 30.0), ("step", 60.0)])
        cloud = convert_demo(demo, Env(), lambda: (np.ones((1200, 3), np.float32), np.zeros((1200, 3), np.uint8)))
        self.assertEqual(cloud["images"].shape, (3, 1200, 6))
        self.assertEqual((rgb["extra"], cloud["extra"]), ("kept", "kept"))

    def test_official_assets_drop_only_the_images(self):
        from jepa4r.data.demos.prepare_official_assets import prepare_assets

        demos = [
            dict(
                images=np.zeros(2),
                actions=np.full((2, 30), k),
                init_state_dict=dict(obj_pos=[k, 0, 0], target_pos=[0, k, 0]),
            )
            for k in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "relocate-v0.pickle"
            source.write_bytes(pickle.dumps(demos))
            manifest = prepare_assets(source, Path(tmp) / "official")
            self.assertEqual(manifest["demo_count"], 3)
            copy = pickle.loads((Path(tmp) / "official" / manifest["demo_files"][2]["relative_path"]).read_bytes())
            self.assertNotIn("images", copy)
            np.testing.assert_array_equal(copy["actions"], demos[2]["actions"])
            records = json.loads((Path(tmp) / "official/rh56_official_task_init.json").read_text())["records"]
            self.assertEqual(records[1]["obj_pos"], [1.0, 0.0, 0.0])

    def test_paired_writer_keeps_terminal_action_and_nonvisual_alignment(self):
        from jepa4r.data.relocate.paired_writer import AdroitPairedWriter

        episode = {
            "pixels": np.zeros((3, 3, 4, 4), dtype=np.uint8),
            "pcd_xyz": np.ones((3, 7, 3), dtype=np.float32),
            "pcd_rgb": np.zeros((3, 7, 3), dtype=np.uint8),
            "action": np.array([[0.5] * 12, [-0.5] * 12, [np.nan] * 12], dtype=np.float32),
            "observation": np.arange(3 * 56, dtype=np.float32).reshape(3, 56),
        }
        with tempfile.TemporaryDirectory() as tmp:
            rgb, pcd = Path(tmp) / "rgb.h5", Path(tmp) / "pcd.h5"
            with AdroitPairedWriter(rgb, pcd) as writer:
                writer.write_episode(episode)
                writer.write_episode(episode)
            with h5py.File(rgb) as a, h5py.File(pcd) as b:
                self.assertNotIn("pcd_xyz", a)
                self.assertNotIn("pixels", b)
                np.testing.assert_array_equal(a["ep_offset"][:], [0, 3])
                for key in ("action", "observation", "ep_len", "ep_offset"):
                    np.testing.assert_array_equal(a[key][:], b[key][:])
                self.assertTrue(np.isnan(a["action"][2]).all())

    def test_metric_depth_projection_keeps_world_axes_and_units(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "surface_geometry", REPO / "jepa4r/envs/adroit/surface_geometry.py"
        )
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        depth = np.full((2, 2), 2.0, dtype=np.float64)
        points = api.depth_to_world(depth, np.array([1.0, 2.0, 3.0]), np.eye(3), 90.0)
        expected = np.array([[[0.0, 3.0, 1.0], [2.0, 3.0, 1.0]], [[0.0, 1.0, 1.0], [2.0, 1.0, 1.0]]])
        np.testing.assert_allclose(points, expected, rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
