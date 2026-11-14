"""Native adapter + actual EAI/MJRL loop, using explicit CPU synthetic I/O only."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
from contextlib import nullcontext


def worker(spec_file):
    import numpy as np
    import torch
    from jepa4r.policy.bc.relocate import adroit_learning as adapter
    from test_bc_recovery import actual_module

    torch.set_num_threads(1)
    spec = json.loads(Path(spec_file).read_text())
    root = Path(spec["root"])
    out = root / spec["phase"]
    policies = []
    logs = []
    official = actual_module(Path(os.environ["JEPA4R_EAI_LOOP"]), [606, 30], [], policies, logs)
    encoder_module = ModuleType("lewm_cortex")
    other_module = ModuleType("jepa_baselines")
    encoder_module.__file__ = str(Path(__file__))
    other_module.__file__ = str(Path(__file__))

    def embedding(*a, **kw):
        return (torch.nn.Linear(1, 1), 192, None, {})

    for name in ("load_lewm", "load_lewm_pcd"):
        setattr(encoder_module, name, embedding)
    for name in ("load_pldm", "load_lejepa", "load_visreg", "load_vjepa21"):
        setattr(other_module, name, embedding)
    original_env = official.env_constructor

    def environment(**kwargs):
        result = original_env(**kwargs)
        result.env.embedding = encoder_module.load_lewm_pcd()[0]
        return result

    official.env_constructor = environment
    original_paths = official.sample_paths

    def paths(*args, **kwargs):
        result = original_paths(*args, **kwargs)
        for item in result:
            item.update(actions=np.zeros((200, 30), np.float32), observations=np.zeros((200, 606), np.float32))
        return result

    official.sample_paths = paths
    run = out / "run"

    class Logger:
        def __init__(self):
            self.stream = (run / "metrics.jsonl").open("a")

        def log(self, data, step=None, **kwargs):
            row = {key: float(value) for key, value in data.items()}
            if step is not None:
                row["_step"] = int(step)
            self.stream.write(json.dumps(row) + "\n")
            self.stream.flush()

    official.setup_wandb = lambda config: Logger()
    tail = [
        "seed=100",
        "epochs=1",
        "num_demos=1",
        "eval_num_traj=1",
        "num_cpu=1",
        "bc_kwargs.batch_size=128",
        "hydra.run.dir=" + str(run),
        "hydra.job.chdir=true",
        "job_name=policy",
        "data_dir=" + str(root / "converted/rgb"),
    ]
    args = SimpleNamespace(
        model="jepa4r",
        checkpoint=root / "encoder",
        output=out,
        epoch=1,
        entry_args=tail,
        bc_recovery=True,
        bc_resume_checkpoint=Path(spec["resume"]) if spec.get("resume") else None,
        bc_pause_after_updates=spec.get("pause"),
        policy_evidence=False,
    )

    def original_entry(*unused, **kwargs):
        run.mkdir(parents=True, exist_ok=False)
        previous = Path.cwd()
        os.chdir(run)
        config = dict(
            seed=100,
            data_dir=str(root / "converted/rgb"),
            num_demos=1,
            device="cpu",
            epochs=1,
            job_name="policy",
            env_kwargs=dict(
                render_gpu_id=0,
                env_name="relocate-v0",
                suite="adroit",
                embedding_name="fixture",
                history_window=3,
                proprio_key="proprio",
            ),
            bc_kwargs=dict(batch_size=128, lr=1e-3, hidden_sizes="(256,256,256)", nonlinearity="relu", dropout=0.0),
            eval_frequency=5,
            save_frequency=10,
            eval_num_traj=1,
            num_cpu=1,
        )
        try:
            official.bc_pvr_train_loop(config)
        finally:
            os.chdir(previous)

    modules = {"lewm_cortex": encoder_module, "jepa_baselines": other_module, "train_loop": official}
    # The original EAI set_seed covers NumPy/Torch. Independent unit forks
    # also share an explicit Python RNG fixture, as the recovery helper tests do.
    __import__("random").seed(100)
    with (
        patch.dict(sys.modules, modules),
        patch.object(adapter, "verify_frozen_checkpoint", return_value={"synthetic_unit_only": True}),
        patch.object(adapter.runpy, "run_path", original_entry),
    ):
        adapter.bc(args)


class AdapterWiringTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("JEPA4R_EAI_LOOP") and os.environ.get("JEPA4R_MJRL_ROOT"),
        "requires restored pinned source; no installation",
    )
    def test_real_native_loop_continuous_and_independent_resume(self):
        import numpy as np
        import torch
        from jepa4r.policy.gciql.gates.arrays import compare_nested

        persistent = os.environ.get("JEPA4R_TEST_ARTIFACT_ROOT")
        if persistent:
            Path(persistent).mkdir(parents=True, exist_ok=False)
        with nullcontext(persistent) if persistent else tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "converted/rgb"
            data.mkdir(parents=True)
            (root / "converted/report.json").write_text('{"synthetic_unit_only":true}')
            (root / "encoder").mkdir()
            (root / "encoder/weights_epoch_1.pt").write_bytes(b"explicit synthetic encoder fixture")
            (root / "encoder/config.json").write_text("{}")
            rng = np.random.default_rng(710)
            demo = dict(
                features=rng.normal(size=(200, 606)).astype(np.float32),
                actions=rng.normal(size=(200, 30)).astype(np.float32),
                embeddings=np.zeros((200, 1), np.float32),
                rewards=np.zeros(200),
                init_state_dict={"fixture": True},
            )
            with (data / "relocate-v0.pickle").open("wb") as f:
                pickle.dump([demo], f, protocol=4)
            for phase, pause, resume, expected in [
                ("continuous", None, None, 0),
                ("pause", 1, None, 75),
                ("resume", None, str(root / "pause/recovery/update_000001.pt"), 0),
            ]:
                spec = root / (phase + ".json")
                spec.write_text(json.dumps(dict(root=str(root), phase=phase, pause=pause, resume=resume)))
                process = subprocess.run(
                    [sys.executable, "-B", __file__, "--worker", str(spec)],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
                )
                (root / (phase + ".stdout.log")).write_text(process.stdout)
                (root / (phase + ".stderr.log")).write_text(process.stderr)
                (root / (phase + ".exit.json")).write_text(
                    json.dumps(
                        dict(
                            actual_exit_code=process.returncode,
                            expected_exit_code=expected,
                            command=[sys.executable, "-B", __file__, "--worker", str(spec)],
                        ),
                        indent=2,
                    )
                )
                self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
            a = torch.load(root / "continuous/recovery/update_000002.pt", weights_only=False)
            b = torch.load(root / "resume/recovery/update_000002.pt", weights_only=False)

            def ordinary_arrays(value):
                if torch.is_tensor(value):
                    return value.detach().cpu().numpy()
                if isinstance(value, dict):
                    return {k: ordinary_arrays(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [ordinary_arrays(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(ordinary_arrays(v) for v in value)
                return value

            for key in ("model", "optimizers", "schedulers", "scaler", "rng", "cursor"):
                self.assertTrue(compare_nested(ordinary_arrays(a[key]), ordinary_arrays(b[key]))["passed"], key)
            ca = json.loads((root / "continuous/recovery/update_000002.pt.caller.json").read_text())
            cb = json.loads((root / "resume/recovery/update_000002.pt.caller.json").read_text())
            self.assertEqual(ca["state"], cb["state"])
            report = json.loads((root / "resume/report.json").read_text())
            self.assertEqual(report["optimizer_updates"], 2)
            self.assertEqual(report["optimizer_evidence"][0]["hook_count"], 2)
            self.assertEqual(json.loads((root / "pause/pause.json").read_text())["actual_exit_code"], 75)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
