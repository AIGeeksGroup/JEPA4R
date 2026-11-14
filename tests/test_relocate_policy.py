"""train_policy on Adroit Relocate and the RH56 twin: the run mode of the configs selects the aggregation plan -- full
(the paper budgets): 3 seeds x 20 evaluations of 25 episodes; smoke: one evaluation of one episode. The harness itself
is replaced by recorded evaluation lines; no simulator runs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jepa4r.config import compose
from jepa4r.eval.aggregation import BC_EPOCHS
from jepa4r.policy import bc

ENVS = ("adroit_relocate", "rm75b_rh56_relocate")


def run_dir(root, cfg, seed):
    return root / (f"bc-seed{seed}/run" if cfg.env.suite == "adroit" else f"rh56/bc_seed{seed}")


class RelocatePolicyTests(unittest.TestCase):
    def relocate_policy(self, cfg, runs):
        """runs: {seed: [(epoch, success %)]} -> (metrics, bc_summary.json) of bc.relocate_policy."""
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "harness", Path(tmp) / "out"
            out.mkdir()
            for seed, rows in runs.items():
                run = run_dir(root, cfg, seed)
                run.mkdir(parents=True)
                lines = [json.dumps({"eval/epoch": epoch, "eval/success": success}) for epoch, success in rows]
                (run / "metrics.jsonl").write_text("\n".join(lines) + "\n")
            with (
                mock.patch.object(bc, "relocate_stage", return_value={"status": "complete"}),
                mock.patch.object(bc, "relocate_run_root", return_value=root),
            ):
                metrics = bc.relocate_policy(cfg, out)
            return metrics, json.loads((out / "bc_summary.json").read_text())

    def test_configs_run_the_paper_budgets_by_default(self):
        for env in ENVS:
            with self.subTest(env=env):
                cfg = compose([f"env={env}", "method=jepa4r"])
                self.assertEqual(cfg.env.harness.mode, "full")
                self.assertEqual(list(cfg.policy.eval.seeds), [100, 200, 300])

    def test_full_mode_aggregates_three_seeds_of_twenty_evaluations(self):
        for env in ENVS:
            with self.subTest(env=env):
                cfg = compose([f"env={env}", "method=jepa4r"])
                runs = {
                    seed: [(epoch, 4.0 * ((i + k) % 26)) for i, epoch in enumerate(BC_EPOCHS)]
                    for k, seed in enumerate((100, 200, 300))
                }
                metrics, summary = self.relocate_policy(cfg, runs)
                best = [max(s for _, s in rows) for rows in runs.values()]
                self.assertAlmostEqual(metrics["success"], sum(best) / 3)
                self.assertAlmostEqual(summary["best_mean_percent"], sum(best) / 3)
                self.assertEqual([s["seed"] for s in summary["seeds"]], [100, 200, 300])
                self.assertEqual([len(s["rounds"]) for s in summary["seeds"]], [20, 20, 20])

    def test_smoke_mode_aggregates_one_evaluation_of_one_episode(self):
        for env in ENVS:
            with self.subTest(env=env):
                cfg = compose([f"env={env}", "method=jepa4r", "env.harness.mode=smoke"])
                metrics, summary = self.relocate_policy(cfg, {100: [(0, 100.0)]})
                self.assertEqual(metrics["success"], 100.0)
                self.assertEqual(summary["seeds"][0]["rounds"][0]["episodes"], 1)

    def test_unknown_mode_is_rejected(self):
        cfg = compose(["env=adroit_relocate", "method=jepa4r", "env.harness.mode=formal"])
        with self.assertRaises(ValueError):
            self.relocate_policy(cfg, {100: [(0, 100.0)]})


if __name__ == "__main__":
    unittest.main()
