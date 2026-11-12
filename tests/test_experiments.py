"""Experiment specifications against the paper's numbers; no simulators, model weights or GPUs required."""
import unittest

from jepa4r.config import REPO_ROOT, compose, run_dir
from jepa4r.orchestrate import expand, load_spec
from jepa4r.report import EXPECTED_DIR, load_expected, required_runs

SPECS = sorted((REPO_ROOT / "experiments").glob("*.yaml"))


class SpecTests(unittest.TestCase):
    def test_every_job_composes(self):
        for path in SPECS:
            jobs = expand(load_spec(path), with_optional=True)
            self.assertTrue(jobs, path.name)
            for job in jobs:
                with self.subTest(spec=path.stem, job=job.command()):
                    run_dir(compose(job.args()), job.stage)

    def test_every_paper_number_has_a_job(self):
        for path in SPECS:
            spec = load_spec(path)
            jobs = expand(spec, with_optional=True)
            self.assertTrue(spec.get("expected"), path.name)
            for rel in spec["expected"]:
                for table in load_expected(REPO_ROOT / rel):
                    for stage, want in required_runs(table):
                        with self.subTest(table=table["name"], run=dict(want)):
                            self.assertTrue(any(job.stage == stage and set(want) <= set(job.keys.items())
                                                for job in jobs))

    def test_every_expected_file_belongs_to_a_spec(self):
        referenced = {rel for path in SPECS for rel in load_spec(path).get("expected", [])}
        for path in EXPECTED_DIR.glob("*.yaml"):
            self.assertIn(f"paper/expected/{path.name}", referenced)

    def test_optional_jobs_are_skipped_by_default(self):
        spec = load_spec(REPO_ROOT / "experiments" / "tab2_cortexbench.yaml")
        self.assertFalse(any(job.keys.get("method") == "random_vitb" for job in expand(spec)))
        self.assertTrue(any(job.keys.get("method") == "random_vitb" for job in expand(spec, with_optional=True)))

    def test_ablation_variants_and_filters(self):
        spec = load_spec(REPO_ROOT / "experiments" / "tab4_ablations.yaml")
        jobs = [job for job in expand(spec, stages=["train_wm"], where={"env": "scene"})
                if job.keys["method"] == "jepa4r"]
        configs = {job.keys["tag"]: compose(job.args()) for job in jobs}
        self.assertEqual(set(configs), {"default", "ctx2", "ctx4", "lr2.5e-5", "lr1e-4", "bs64", "bs256"})
        self.assertEqual((configs["ctx2"].method.history, configs["ctx2"].method.num_frames), (2, 3))
        self.assertAlmostEqual(configs["lr2.5e-5"].method.train.lr, 2.5e-5)
        self.assertEqual(configs["bs256"].method.train.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
