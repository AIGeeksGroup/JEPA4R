"""train_policy on Adroit Relocate and the RH56 twin: the paper's rule over the workflow's BC records -- 3 seeds x 20
evaluations of 25 episodes, per seed the best evaluation, the mean over seeds. The workflow itself is replaced by
recorded evaluation lines; no simulator runs."""

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
            root, out = Path(tmp) / "workflow", Path(tmp) / "out"
            out.mkdir()
            for seed, rows in runs.items():
                run = run_dir(root, cfg, seed)
                run.mkdir(parents=True)
                lines = [json.dumps({"eval/epoch": epoch, "eval/success": success}) for epoch, success in rows]
                (run / "metrics.jsonl").write_text("\n".join(lines) + "\n")
            with (
                mock.patch.object(bc, "relocate_stage", return_value=str(root)),
                mock.patch.object(bc, "relocate_run_root", return_value=root),
            ):
                metrics = bc.relocate_policy(cfg, out)
            return metrics, json.loads((out / "bc_summary.json").read_text())

    def test_configs_evaluate_three_seeds(self):
        for env in ENVS:
            with self.subTest(env=env):
                cfg = compose([f"env={env}", "method=jepa4r"])
                self.assertEqual(list(cfg.policy.eval.seeds), [100, 200, 300])

    def test_aggregates_three_seeds_of_twenty_evaluations(self):
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
                self.assertEqual(summary["seeds"][0]["rounds"][0]["episodes"], 25)

    def test_a_missing_evaluation_round_is_an_error(self):
        cfg = compose(["env=adroit_relocate", "method=jepa4r"])
        runs = {seed: [(epoch, 50.0) for epoch in BC_EPOCHS[:-1]] for seed in (100, 200, 300)}
        with self.assertRaises(ValueError):
            self.relocate_policy(cfg, runs)


if __name__ == "__main__":
    unittest.main()
