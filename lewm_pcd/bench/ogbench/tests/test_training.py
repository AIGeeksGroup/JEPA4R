"""Training switches are inspectable without GPU runtime imports."""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "lewm_pcd"))


class TrainingTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("bench.ogbench.training")
        except ModuleNotFoundError:
            self.fail("common training switch has not been implemented")

    def test_shared_entry_and_tiny_native_defaults(self):
        module = self.module()
        for profile in ("single200", "double500", "triple_front600", "quadruple600"):
            command = module.build_command(profile, "pldm", "/sealed", "/fresh", python="PY")
            self.assertEqual(command[:3], ["PY", "-m", "bench.ogbench.pldm.train"])
            self.assertIn(profile, command)
        command = module.build_command("single200", "lejepa", "/sealed", "/fresh", python="PY")
        self.assertEqual(command[:3], ["PY", "-m", "bench.ogbench.tiny"])
        self.assertFalse(set(("--bs", "--V", "--lr", "--lamb", "--proj-dim")) & set(command))
        with self.assertRaises(ValueError):
            module.build_command("single200", "lejepa_vitb", "/sealed", "/fresh")

    def test_resource_overrides_and_environment(self):
        module = self.module()
        command = module.build_command("double500", "pldm", "/sealed", "/fresh", workers=3, prefetch=1)
        self.assertEqual(command[command.index("--workers") + 1], "3")
        self.assertEqual(command[command.index("--prefetch") + 1], "1")
        env = module.build_environment("/sealed", "/fresh", environ={"PYTHONPATH": "prior"})
        self.assertEqual(Path(env["STABLEWM_HOME"]), Path("/fresh").resolve())
        self.assertIn(str(ROOT / "lewm_pcd"), env["PYTHONPATH"].split(os.pathsep))
        self.assertIn("prior", env["PYTHONPATH"].split(os.pathsep))
        with self.assertRaises(ValueError):
            module.build_command("single200", "pldm", "/sealed", "/fresh", prefetch=0)
        with self.assertRaises(ValueError):
            module.build_command("single200", "lejepa", "/sealed", "/fresh", workers=0)

    def test_profile_budget_uses_same_sixteen_row_method(self):
        self.module()
        shared = importlib.import_module("bench.ogbench.pldm.shared_data")
        expected = {"single200": 185, "double500": 485,
                    "triple_front600": 585, "quadruple600": 585}
        for profile, per in expected.items():
            value = shared.contract(profile)
            self.assertEqual(value["per_episode"], per)
            self.assertEqual(value["frames"], 16)
            self.assertEqual(value["actions"], 15)
            self.assertEqual(value["total_updates"], 10 * value["train_batches"])
            self.assertEqual(shared.location(-1, profile, 2), (1, per - 1))
            with self.assertRaises(IndexError):
                shared.location(2 * per, profile, 2)

    def test_pldm_method_is_exact_frozen_copy(self):
        self.module()
        path = ROOT / "lewm_pcd/bench/ogbench/pldm/method.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "0400e518a220d7843f62b70427dd645300a46f3ddb951ce58e15ab52625b7712")

    def test_accept_rejects_cross_profile_even_when_update_budgets_match(self):
        self.module()
        train = importlib.import_module("bench.ogbench.pldm.train")
        shared = importlib.import_module("bench.ogbench.pldm.shared_data")
        self.assertEqual(shared.contract("triple_front600"), shared.contract("quadruple600"))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=ROOT / "lewm_pcd/bench/ogbench/pldm",
                method="pldm", profile="triple_front600", phase="formal",
                workers=20, prefetch=2, training_output=Path(directory))
            old_args = SimpleNamespace(**{**vars(args), "profile": "quadruple600"})
            protocol = dict(binding=train.binding(old_args), model_config=train.factory_config(old_args),
                            loader_counts=shared.contract(old_args.profile))
            train.write(args.training_output / "result/actual_protocol.json", protocol)
            (args.training_output / "actual_exit_code").write_text("0\n")
            (args.training_output / "stdout.log").write_text("")
            with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                 self.assertRaisesRegex(ValueError, "current.*binding|profile.*binding"):
                train.accept(args)

    def test_accept_protocol_matches_current_method_contract_and_sources(self):
        self.module()
        train = importlib.import_module("bench.ogbench.pldm.train")
        shared = importlib.import_module("bench.ogbench.pldm.shared_data")
        args = SimpleNamespace(root=ROOT / "lewm_pcd/bench/ogbench/pldm",
            method="pldm", profile="quadruple600", phase="formal", workers=20, prefetch=2)
        protocol = dict(binding=train.binding(args), model_config=train.factory_config(args),
                        loader_counts=shared.contract(args.profile))
        self.assertTrue(hasattr(train, "validate_protocol"), "current-protocol guard is missing")
        train.validate_protocol(args, protocol)
        import copy
        for key, bad in (("method", "lejepa"), ("contract", {}),
                         ("manifest_sha256", "0" * 64), ("source_freeze_sha256", "0" * 64)):
            changed = copy.deepcopy(protocol)
            changed["binding"][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "current.*binding|profile.*binding"):
                train.validate_protocol(args, changed)


if __name__ == "__main__":
    unittest.main()
