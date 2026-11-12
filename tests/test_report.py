"""Paper-vs-reproduction report on synthetic runs; no simulators, model weights or GPUs required."""
import json
import tempfile
import unittest
from pathlib import Path

from jepa4r.report import build_table, load_records, render_markdown, report

GRID = {
    "name": "grid", "stage": "train_policy", "rows": "method", "cols": "env", "metric": "success",
    "derived": {"mean": {"mean_of": ["a", "b"]}},
    "values": {"m": {"a": 10.0, "b": 20.0, "mean": 15.0}},
    "published": {"p": {"label": "P", "values": {"a": 1.0}}},
}


def write_run(root, stage, metrics, **key):
    record = {"stage": stage, "env": None, "method": None, "decoder": None, "tag": "default", **key,
              "metrics": metrics, "timestamp": "2026-01-01T00:00:00"}
    path = Path(root).joinpath(stage, *map(str, key.values()), "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_grid_with_derived_and_published_rows(self):
        write_run(self.root, "train_policy", {"success": 12.0}, env="a", method="m")
        write_run(self.root, "train_policy", {"success": 22.0}, env="b", method="m")
        write_run(self.root, "train_policy", {"success": 99.0}, env="a", method="m", tag="other")
        rows, cols, cells = build_table(GRID, load_records(self.root))
        self.assertEqual((rows, cols), (["m", "p"], ["a", "b", "mean"]))
        self.assertEqual((cells[("m", "a")].ours, cells[("m", "a")].delta), (12.0, 2.0))
        self.assertEqual(cells[("m", "mean")].ours, 17.0)
        self.assertEqual(cells[("p", "a")].op, "published")
        self.assertIn("12.0 / 10.0", render_markdown(GRID, rows, cols, cells))

    def test_flat_and_formatted_tables(self):
        write_run(self.root, "eval_recon", {"nn_mean": 0.8, "imag_nn@15": 0.7, "imag_nn_max": 1.0},
                  env="cube_single", decoder="page4d")
        records = load_records(self.root)
        fixed = {"env": "cube_single", "decoder": "page4d"}
        flat = {"name": "flat", "stage": "eval_recon", "fixed": fixed,
                "values": {"nn_mean": 0.79, "imag_nn_max": {"value": 1.1, "op": "<="}}}
        _, _, cells = build_table(flat, records)
        self.assertEqual(cells[("nn_mean", None)].ours, 0.8)
        self.assertTrue(cells[("imag_nn_max", None)].ok())
        formatted = {"name": "fmt", "stage": "eval_recon", "fixed": fixed, "rows": "series", "cols": "env_step",
                     "metric_format": "{row}@{col}", "values": {"imag_nn": {15: 0.75, 100: 0.85}}}
        _, _, cells = build_table(formatted, records)
        self.assertEqual(cells[("imag_nn", 15)].ours, 0.7)
        self.assertIsNone(cells[("imag_nn", 100)].ours)

    def test_report_on_the_paper_tables_without_runs(self):
        text = report(outputs=self.root / "outputs", out=self.root / "report")
        self.assertIn("Tab. 1", text)
        self.assertIn("Tab. 4d", text)
        self.assertTrue((self.root / "report" / "report.md").is_file())
        self.assertTrue((self.root / "report" / "report.csv").is_file())


if __name__ == "__main__":
    unittest.main()
