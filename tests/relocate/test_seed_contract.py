"""CPU seed forwarding only; mock construction/Manager, no training."""

import ast
import importlib.util
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    s = importlib.util.spec_from_file_location(name, ROOT / relative)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class SeedContractTests(unittest.TestCase):
    def call_shared(self, explicit):
        # LeWM train.py run() -> world_model.run_random_split
        train = load("jepa4r/models/world_model.py", "_guide_seed_train")
        # imported inside run_random_split
        import hydra
        import lightning as pl
        import stable_pretraining as spt
        import stable_worldmodel as swm

        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 20

            def __getitem__(self, i):
                return {"action": torch.zeros(2)}

            def get_dim(self, key):
                return 2

        cfg = OmegaConf.create(
            {
                "data": {
                    "dataset": {"name": "unused", "frameskip": 5, "keys_to_load": ["action"]},
                    "preprocessor": {"_target_": "PREPROCESS"},
                },
                "model": {"_target_": "MODEL", "action_encoder": {"input_dim": 0}},
                "seed": 17,
                "train_split": 0.9,
                "loader": {"batch_size": 2, "num_workers": 0},
                "optimizer": {"type": "AdamW", "lr": 0.001},
                "loss": {"sigreg": {"kwargs": {"knots": 17, "num_proj": 4}}},
                "trainer": {},
                "wandb": {"enabled": False},
                "subdir": "seed_probe",
                "output_model_name": "seed_probe",
            }
        )
        events = []
        captured = {}

        def instantiate(value):
            if value.get("_target_") == "MODEL":
                events.append(("model", torch.initial_seed(), random.random(), np.random.uniform()))
                return torch.nn.Linear(1, 1)
            return lambda value: value

        class Trainer:
            def __init__(self, **kwargs):
                events.append(("trainer", torch.initial_seed()))

        class Manager:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __call__(self):
                return None

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "9123"}),
            mock.patch.object(swm.data, "load_dataset", return_value=Dataset()),
            mock.patch.object(train, "get_column_normalizer", return_value=lambda value: value),
            mock.patch.object(hydra.utils, "instantiate", side_effect=instantiate),
            mock.patch.object(swm.data.utils, "get_cache_dir", return_value=tmp),
            mock.patch.object(spt, "Module", side_effect=lambda **kw: SimpleNamespace(**kw)),
            mock.patch.object(spt.data, "DataModule", side_effect=lambda **kw: SimpleNamespace(**kw)),
            mock.patch.object(pl, "Trainer", Trainer),
            mock.patch.object(spt, "Manager", Manager),
        ):
            torch.manual_seed(555)
            if explicit:
                train.run_random_split(cfg, None, manager_seed=3072, original_layout=True)
            else:
                train.run_random_split(cfg, None, original_layout=True)
        return events, captured

    def test_shared_explicit_seed_precedes_model_and_manager(self):
        events, captured = self.call_shared(True)
        self.assertEqual(events[0][0:2], ("model", 3072))
        self.assertEqual(captured["seed"], 3072)
        self.assertEqual(events[0][2], random.Random(3072).random())
        self.assertEqual(events[0][3], np.random.RandomState(3072).uniform())

    def test_shared_legacy_omission_keeps_existing_behavior(self):
        events, captured = self.call_shared(False)
        self.assertEqual(events[0][0:2], ("model", 555))
        self.assertNotIn("seed", captured)

    def test_native_wrapper_explicitly_forwards_cfg_seed(self):
        tree = ast.parse((ROOT / "jepa4r/policy/bc/relocate/adroit_learning.py").read_text())
        calls = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "native.run_random_split"
        ]
        self.assertTrue(
            any(
                any(k.arg == "manager_seed" and ast.unparse(k.value) == "int(cfg.seed)" for k in n.keywords)
                for n in calls
            )
        )

    def test_rh56_smoke_and_full_wm_use3072_without_changing_collection_or_bc(self):
        w = load("jepa4r/policy/bc/relocate/rh56_workflow.py", "_seed_rh_plan")
        for mode in ("smoke", "full"):
            args = SimpleNamespace(
                env="rh56-relocate",
                model="jepa4r",
                mode=mode,
                output=Path("/external/new"),
                sim_python="sim",
                wm_python="wm",
                seed=42,
            )
            stages = w.plan(
                args,
                {
                    "rh56_expert_root": "/external/bundle",
                    "eai_vc_root": "/external/eai",
                    "rh56_official_task_init": "/external/init",
                },
            )
            command = next(x["command"] for x in stages if x["name"] == "rh56_wm")
            self.assertEqual(command[command.index("--seed") + 1], "3072")
            self.assertEqual(
                [x["name"] for x in stages if x["name"].startswith("rh56_bc_")],
                ["rh56_bc_100"] if mode == "smoke" else ["rh56_bc_100", "rh56_bc_200", "rh56_bc_300"],
            )

    def test_rh56_original_manager_shim_passes_explicit_seed(self):
        tree = ast.parse((ROOT / "jepa4r/envs/twins/rh56/wm_bridge.py").read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "manager")
        captured = {}
        env = {
            "args": SimpleNamespace(seed=3072),
            "resume_checkpoint": None,
            "original_manager": lambda *a, **kw: captured.update(kw),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<actual-rh-manager>", "exec"),
            env,
        )
        env["manager"]()
        self.assertEqual(captured["seed"], 3072)

    def test_pldm_seeded_manager_context_is_explicit_and_restores_binding(self):
        m = load("jepa4r/baselines/relocate/train_pldm.py", "_seed_pldm")
        captured = {}

        def original(*a, **kw):
            return captured.update(kw)

        native = SimpleNamespace(spt=SimpleNamespace(Manager=original))
        with mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "9123"}):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with m.explicit_training_seed(native, 3072):
                    self.assertEqual(torch.initial_seed(), 3072)
                    native.spt.Manager()
                    raise RuntimeError("probe")
        self.assertIs(native.spt.Manager, original)
        self.assertEqual(captured["seed"], 3072)

    def test_lejepa_explicit_seed_and_legacy_torch0_are_distinct(self):
        tree = ast.parse((ROOT / "jepa4r/baselines/relocate/train_lejepa_relocate.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_training_seed")
        env = {"torch": torch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<actual-lejepa-seed>", "exec"),
            env,
        )
        env["initialize_training_seed"](3072)
        self.assertEqual(torch.initial_seed(), 3072)
        before_py = random.getstate()
        before_np = np.random.get_state()
        env["initialize_training_seed"](None)
        self.assertEqual(torch.initial_seed(), 0)
        self.assertEqual(random.getstate(), before_py)
        np.testing.assert_array_equal(np.random.get_state()[1], before_np[1])

    def test_lejepa_adapter_passes3072_for_both_budgets(self):
        m = load("jepa4r/baselines/relocate/train_lejepa.py", "_seed_lejepa")
        for mode in ("smoke", "formal"):
            parser = m.make_parser()
            args = parser.parse_args(["--dataset", "/external/input.h5", "--output", "/external/new", "--mode", mode])
            plan = m.build_plan(args, parser)
            argv = plan["native_argv"]
            self.assertEqual(argv[argv.index("--seed") + 1], "3072")


if __name__ == "__main__":
    unittest.main()
