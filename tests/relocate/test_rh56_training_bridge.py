"""Lightweight RH56 boundary tests; no Torch, simulator or training required."""

import ast
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "_rh56_training_test_" + name, ROOT / "jepa4r/envs/twins/rh56" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [str(ROOT), *sys.path]):
        spec.loader.exec_module(module)
    return module


class TrainingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wm = load("wm_bridge")
        cls.bc = load("bc_bridge")

    def test_bc_bindings_restore_after_success_and_failure(self):
        official_names = (
            "load_pretrained_model",
            "env_constructor",
            "setup_wandb",
            "BatchNormMLP",
            "sample_paths",
            "rollout_from_init_states",
        )
        wrapper_names = ("load_pretrained_model", "get_proprioception")
        for fail in (False, True):
            with self.subTest(fail=fail):
                official = SimpleNamespace(**{name: object() for name in official_names})
                wrapper = SimpleNamespace(**{name: object() for name in wrapper_names})
                before = (vars(official).copy(), vars(wrapper).copy())
                replacements = {
                    name: object()
                    for name in ("load_embedding", "env_constructor", "make_policy", "sample_paths", "train_rollout")
                }
                logger = object()
                try:
                    with self.bc.install_bc_runtime(official, wrapper, logger=logger, **replacements):
                        self.assertIs(official.load_pretrained_model, replacements["load_embedding"])
                        self.assertIs(wrapper.load_pretrained_model, replacements["load_embedding"])
                        self.assertIs(official.env_constructor, replacements["env_constructor"])
                        self.assertIs(official.setup_wandb({}), logger)
                        self.assertIs(official.BatchNormMLP, replacements["make_policy"])
                        self.assertIs(official.sample_paths, replacements["sample_paths"])
                        self.assertIs(official.rollout_from_init_states, replacements["train_rollout"])
                        env = SimpleNamespace(unwrapped=SimpleNamespace(get_proprio=lambda: "proprio13"))
                        self.assertEqual(wrapper.get_proprioception(env, "adroit"), "proprio13")
                        if fail:
                            raise RuntimeError("original loop failed")
                except RuntimeError as error:
                    self.assertTrue(fail)
                    self.assertEqual(str(error), "original loop failed")
                self.assertEqual(vars(official), before[0])
                self.assertEqual(vars(wrapper), before[1])

    def test_bc_training_call_is_inside_restoring_scope(self):
        tree = ast.parse((ROOT / "jepa4r/envs/twins/rh56/bc_bridge.py").read_text())
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == "install_bc_runtime"
                for item in node.items
            )
        ]
        self.assertEqual(len(scopes), 1)
        self.assertIn(
            "official.bc_pvr_train_loop",
            {ast.unparse(node.func) for node in ast.walk(scopes[0]) if isinstance(node, ast.Call)},
        )

    def test_temporary_bindings_restore_nested_and_partial_installation(self):
        runtime = load("_runtime")
        target = SimpleNamespace(value="original")
        with runtime.temporary_bindings((target, "value", "outer")):
            with runtime.temporary_bindings((target, "value", "inner")):
                self.assertEqual(target.value, "inner")
            self.assertEqual(target.value, "outer")
        self.assertEqual(target.value, "original")
        with self.assertRaises(AttributeError):
            with runtime.temporary_bindings((target, "value", "changed"), (target, "missing", "invalid")):
                self.fail("partial installation must fail before running the loop")
        self.assertEqual(vars(target), {"value": "original"})

    def test_shared_file_hash_matches_original_full_file_formula(self):
        import hashlib

        runtime = load("_runtime")
        functions = [
            self.wm.sha,
            self.bc._sha,
            *(load(name).sha for name in ("collect", "data_tools", "official_init")),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "payload"
            for data in (b"", b"RH56", b"x" * ((8 << 20) + 17)):
                path.write_bytes(data)
                self.assertEqual(runtime.file_sha256(path), hashlib.sha256(data).hexdigest())
                for function in functions:
                    self.assertEqual(function(path), runtime.file_sha256(path))

    def test_shared_json_preserves_encoding_and_exclusive_receipts(self):
        runtime = load("_runtime")
        payload = {"z": [1, True], "a": "RH56-\u6d4b\u8bd5"}
        writers = [self.wm.atomic_json, self.bc.atomic_json, load("collect").atomic_json, load("data_tools").save]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            for writer in writers:
                writer(path, payload)
                expected = json.dumps(payload, indent=2).replace("\n", os.linesep).encode("utf-8")
                self.assertEqual(path.read_bytes(), expected)
                self.assertFalse(path.with_suffix(".tmp.json").exists())
            path = Path(temp) / "immutable.json"
            original = load("official_init")
            original.write_json(path, payload, new=True)
            expected = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
            expected = expected.replace("\n", os.linesep).encode("utf-8")
            self.assertEqual(path.read_bytes(), expected)
            with self.assertRaises(FileExistsError):
                runtime.atomic_json(path, {"changed": True}, new=True)
            with self.assertRaises(ValueError):
                original.write_json(path, {"bad": float("nan")})
            self.assertEqual(path.read_bytes(), expected)

    def dataset(self, path):
        seeds = np.repeat([61000000, 62000000, 63000000], 6)
        actions = np.arange(36, dtype=np.float32).reshape(18, 2)
        dataset = SimpleNamespace(
            h5_path=str(path),
            lengths=[6, 6, 6],
            offsets=[0, 6, 12],
            clip_indices=[(episode, frame) for episode in range(3) for frame in (0, 1)],
            get_col_data=lambda key: seeds if key == "episode_seed" else actions,
        )
        manifest = {
            "episodes": [
                dict(episode_index=i, seed=int(seeds[i * 6]), initial_state_sha256=chr(97 + i) * 64, split=split)
                for i, split in enumerate(("train", "val", "test"))
            ]
        }
        return dataset, manifest, actions

    def test_episode_split_and_train_only_normalization_restore_hooks(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset, manifest, actions = self.dataset(Path(temp) / "paired.h5")

            def original_normalizer(data, source, target):
                return data.get_col_data(source).copy()

            def original_split(data, lengths, generator=None):
                return ["original"]

            train = SimpleNamespace(
                get_column_normalizer=original_normalizer,
                spt=SimpleNamespace(data=SimpleNamespace(random_split=original_split)),
            )
            with self.assertRaisesRegex(RuntimeError, "expected boundary interruption"):
                # This test isolates hook restoration using an in-memory reader.
                # The checks on real data files have their own tests.
                with (
                    mock.patch(
                        "jepa4r.envs.twins.rh56.input_identity.validate_episode_producers",
                        return_value={"explicit_synthetic_hook_fixture": True},
                    ),
                    self.wm.install_episode_split(
                        train, dataset.h5_path, manifest, subset_factory=lambda data, indices: (data, indices)
                    ) as audit,
                ):
                    normalized_input = train.get_column_normalizer(dataset, "action", "action")
                    np.testing.assert_array_equal(normalized_input, actions[:6])
                    subsets = train.spt.data.random_split(dataset, [0.9, 0.1])
                    self.assertEqual([indices for _, indices in subsets], [[0, 1], [2, 3]])
                    self.assertEqual(audit.partition.episode_ids["test"], [2])
                    self.assertEqual(audit.normalizers["action"]["rows"], 6)
                    raise RuntimeError("expected boundary interruption")
            self.assertIs(train.get_column_normalizer, original_normalizer)
            self.assertIs(train.spt.data.random_split, original_split)

    def test_hooks_do_not_intercept_other_dataset_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset, manifest, _ = self.dataset(Path(temp) / "other.h5")
            normalizer = mock.Mock(return_value="untouched")
            splitter = mock.Mock(return_value=["untouched"])
            train = SimpleNamespace(
                get_column_normalizer=normalizer, spt=SimpleNamespace(data=SimpleNamespace(random_split=splitter))
            )
            with self.wm.install_episode_split(train, Path(temp) / "selected.h5", manifest):
                self.assertEqual(train.get_column_normalizer(dataset, "action", "action"), "untouched")
                self.assertEqual(train.spt.data.random_split(dataset, [0.9, 0.1]), ["untouched"])
            normalizer.assert_called_once_with(dataset, "action", "action")
            splitter.assert_called_once_with(dataset, [0.9, 0.1], generator=None)

    def test_uint8_switch_rejects_non_rgb_or_non_cuda_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "must-not-exist"
            for route, device in [("pcd", "cuda"), ("rgb", "cpu")]:
                with self.subTest(route=route, device=device):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        self.wm.main(
                            [
                                "--manifest",
                                "unused",
                                "--dataset",
                                "unused",
                                "--route",
                                route,
                                "--device",
                                device,
                                "--output",
                                str(output),
                                "--uint8-gpu-preprocess",
                            ]
                        )
                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(output.exists())

    def test_loader_budget_changes_do_not_change_formal_update_budget(self):
        baseline = self.wm.resolve_budget("formal")
        for workers, prefetch in [(0, 1), (24, 2), (28, 3)]:
            budget = self.wm.resolve_budget("formal", workers=workers, prefetch_factor=prefetch)
            self.assertEqual(
                self.wm.expected_optimizer_steps(budget, 1606197), self.wm.expected_optimizer_steps(baseline, 1606197)
            )
            self.assertEqual(budget["completed_epochs"], 10)
            self.assertIsNone(budget["train_limit"])
        for value in (0, 4):
            with self.assertRaisesRegex(ValueError, "prefetch"):
                self.wm.resolve_budget("formal", prefetch_factor=value)

    def test_baseline_fingerprint_ignores_run_outputs_but_detects_learning_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "jepa4r"  # the package; its configs are in ../configs
            (root / "models/encoders").mkdir(parents=True)
            (root / "models/world_model.py").write_text("original training source\n")
            encoder = root / "models/encoders" / "encoder.py"
            encoder.write_text("original encoder\n")
            (root.parent / "configs/method/relocate/train").mkdir(parents=True)
            (root.parent / "configs/method/relocate/train/lewm.yaml").write_text("history_size: 3\n")
            before = self.wm.tree_sha(root)
            (root / "outputs").mkdir()
            (root / "outputs/state.json").write_text('{"phase":"training"}')
            (root / "weights.pt").write_bytes(b"cached weights")
            self.assertEqual(self.wm.tree_sha(root), before)
            encoder.write_text("changed encoder\n")
            self.assertNotEqual(self.wm.tree_sha(root), before)

    def test_source_directories_reject_output_and_fresh_output_never_overwrites(self):
        for name in ("models", "data", "envs", "policy", "baselines", "decoder", "eval", "metrics", "stages", "utils"):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.wm.validate_output_location(ROOT / "jepa4r" / name / "run", ROOT / "jepa4r")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            sentinel = output / "existing.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                self.wm.main(["--manifest", "unused", "--dataset", "unused", "--route", "rgb", "--output", str(output)])
            with self.assertRaises(FileExistsError):
                self.bc.run_bc(SimpleNamespace(out=str(output), baseline=str(ROOT / "jepa4r")))
            self.assertEqual(sentinel.read_text(), "keep")
            with self.assertRaisesRegex(ValueError, "protected"):
                self.bc.run_bc(
                    SimpleNamespace(
                        out=str(output / "fresh"),
                        baseline=str(ROOT / "jepa4r"),
                        demo_cache_root=str(ROOT / "jepa4r/models/relocate"),
                    )
                )
            self.assertFalse((output / "fresh").exists())

    def test_bc_default_paths_and_aliases_are_checkout_relative(self):
        args = [
            "--route",
            "pcd",
            "--checkpoint",
            "external-checkpoint",
            "--demos-manifest",
            "external-demos.json",
            "--out",
            "fresh-output",
        ]
        with mock.patch.dict(os.environ, {"CORTEXBENCH_EAI_VC_ROOT": "external-eai-vc"}):
            with mock.patch.object(self.bc, "run_bc", side_effect=lambda options: options):
                options = self.bc.main(args)
        self.assertEqual(Path(options.baseline), ROOT / "jepa4r")
        self.assertEqual(options.eai_root, "external-eai-vc")
        with mock.patch.object(self.bc, "run_bc", side_effect=lambda options: options):
            options = self.bc.main(args + ["--baseline-root", "other-root", "--eai-vc-root", "other-eai"])
        self.assertEqual(options.baseline, "other-root")
        self.assertEqual(options.eai_root, "other-eai")

    def test_bc_native_t_plus_one_layout_and_point_color_units_are_preserved(self):
        arrays = dict(
            observations=np.zeros((201, 56), np.float32),
            proprio=np.zeros((201, 13), np.float32),
            actions=np.zeros((200, 12), np.float32),
            rewards=np.zeros(200, np.float32),
            initial_state_json=np.array(json.dumps({"seed": 64500000})),
            xyz=np.zeros((201, 1200, 3), np.float32),
            rgb=np.full((201, 1200, 3), 255, np.uint8),
        )
        result = self.bc.prepare_demo(arrays, "pcd")
        self.assertEqual(result["images"].shape, (200, 1200, 6))
        self.assertEqual(result["actions"].shape, (200, 12))
        self.assertEqual(result["env_infos"]["proprio"].shape, (200, 13))
        np.testing.assert_array_equal(result["images"][..., 3:], 255)
        self.assertEqual(result["init_state_dict"]["seed"], 64500000)

    def test_bc_formal_protocol_requires_complete_official_evaluation_schedule(self):
        spec = self.bc.build_run_spec("formal", 10, 100, None)
        self.assertEqual((spec["num_demos"], spec["epochs"], spec["eval_num_traj"]), (100, 100, 25))
        rows = [dict(epoch_loss=0.1, _step=i) for i in range(1, 101)]
        rows += [dict(**{"eval/epoch": e, "eval/success": 4.0}) for e in list(range(5, 100, 5)) + [99]]
        result = self.bc.validate_metric_rows(rows, 100, 5)
        self.assertEqual((result["loss_epochs"], result["evaluation_rounds"]), (100, 20))
        with self.assertRaisesRegex(ValueError, "evaluation rounds"):
            self.bc.validate_metric_rows(rows[:-1], 100, 5)

    def test_learning_is_delegated_to_existing_entrypoints(self):
        wm = ast.parse((ROOT / "jepa4r/envs/twins/rh56/wm_bridge.py").read_text())
        bc = ast.parse((ROOT / "jepa4r/envs/twins/rh56/bc_bridge.py").read_text())

        def calls(tree):
            return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}

        self.assertIn("train.run_random_split", calls(wm))
        self.assertIn("official.bc_pvr_train_loop", calls(bc))
        self.assertNotIn("torch.optim.AdamW", calls(wm))


if __name__ == "__main__":
    unittest.main()
