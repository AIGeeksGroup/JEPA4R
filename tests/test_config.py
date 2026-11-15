"""Configuration composition; no simulators, model weights or GPUs required."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from jepa4r.config import available, compose, load_group, run_dir

GROUPS = ("env", "method", "policy", "decoder", "encoder")


def impl_modules(cfg):
    stack, found = [OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)], []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("impl"), str):
                found.append(node["impl"])
            stack.extend(node.values())
    return found


class GroupTests(unittest.TestCase):
    def test_every_config_loads_and_is_named_after_its_file(self):
        for group in GROUPS:
            self.assertTrue(available(group), group)
            for name in available(group):
                with self.subTest(group=group, name=name):
                    self.assertEqual(load_group(group, name).name, name)

    def test_every_impl_module_imports(self):
        modules = {m for group in GROUPS for name in available(group) for m in impl_modules(load_group(group, name))}
        self.assertTrue(modules)
        for module in sorted(modules):
            with self.subTest(module=module):
                importlib.import_module(module)


class ComposeTests(unittest.TestCase):
    def test_policy_defaults_to_the_task(self):
        for env, policy in (("cube_single", "gciql"), ("puzzle_4x5", "dqc"), ("scene", "bc"), ("mw_hammer", "bc")):
            with self.subTest(env=env):
                self.assertEqual(compose([f"env={env}", "method=jepa4r"]).policy.name, policy)
        self.assertEqual(compose(["env=adroit_relocate", "method=jepa4r"]).env.policy_data.demos_per_task, 100)

    def test_per_task_overrides_and_encoder_references(self):
        cases = (
            ("cube_double", "pldm", "impala"),
            ("scene", "pldm", "vit_tiny14"),
            ("cube_single", "vjepa21", "vit_l16"),
            ("mw_hammer", "vjepa21", "vit_b16"),
            ("scene", "jepa4r_dgcnn", "dgcnn"),
            ("scene", "jepa4r", "pointnet"),
        )
        for env, method, encoder in cases:
            with self.subTest(env=env, method=method):
                self.assertEqual(compose([f"env={env}", f"method={method}"]).method.encoder.name, encoder)
        cfg = compose(["env=scene", "method=jepa4r"])
        self.assertEqual(list(cfg.method.encoder.hidden), [1080, 4320])
        self.assertAlmostEqual(cfg.method.train.lr, 5e-5)

    def test_command_line_overrides(self):
        cfg = compose(["env=scene", "method=jepa4r", "method.train.lr=1e-4", "method.encoder=pointnet2", "tag=x"])
        self.assertAlmostEqual(cfg.method.train.lr, 1e-4)
        self.assertEqual(cfg.method.encoder.arch, "pointnet2")
        self.assertEqual(cfg.tag, "x")

    def test_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"JEPA4R_OUTPUTS": str(root / "outputs"), "JEPA4R_DATA": str(root / "data")}
            with mock.patch.dict(os.environ, env):
                cfg = compose(["env=scene", "method=jepa4r", "tag=ctx2"])
                self.assertEqual(run_dir(cfg, "collect_wm_data"), root / "data" / "wm" / "scene")
                self.assertEqual(run_dir(cfg, "train_wm"), root / "outputs" / "train_wm" / "scene" / "jepa4r" / "ctx2")
                cfg = compose(["env=cube_single", "decoder=page4d"])
                self.assertEqual(
                    run_dir(cfg, "eval_recon"), root / "outputs" / "eval_recon" / "cube_single" / "page4d" / "default"
                )
                self.assertEqual(
                    run_dir(compose(["method=jepa4r"]), "count_params"),
                    root / "outputs" / "count_params" / "jepa4r" / "default",
                )


if __name__ == "__main__":
    unittest.main()
