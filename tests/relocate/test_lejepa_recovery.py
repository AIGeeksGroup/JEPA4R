"""CPU mechanics for the native manual-loop recovery hook; no GPU/model training case."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import random
import os
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from jepa4r.models.relocate.recovery import stateful_loader_class, RecoveryPause
from jepa4r.baselines.relocate.lejepa_recovery import LejepaRecovery


class Rows(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        return torch.tensor([i / 12.0, np.random.uniform(), random.random()]) + torch.rand(3) * 0.01


def seed():
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)


def same(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
        return
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
        return
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            same(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            same(test, x, y)
    else:
        test.assertEqual(a, b)


class LejepaRecoveryTests(unittest.TestCase):
    @contextmanager
    def artifacts(self):
        if os.environ.get("JEPA4R_TEST_ARTIFACT_ROOT"):
            root = Path(os.environ["JEPA4R_TEST_ARTIFACT_ROOT"]) / self._testMethodName
            root.mkdir(parents=True, exist_ok=False)
            yield str(root)
        else:
            with tempfile.TemporaryDirectory() as directory:
                yield directory

    def run_branch(self, directory, pause=False, resume=None):
        seed()
        loader = stateful_loader_class(DataLoader)(Rows(), batch_size=2, shuffle=True, drop_last=True)
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.1), torch.nn.Linear(4, 2))
        sigreg = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(net.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, 0.8)
        args = SimpleNamespace(
            seed=3072, workers=0, epochs=10, max_steps=2, dataset="toy-real-on-fetch", bs=2, V=1, name="toy"
        )
        identity = {
            "model": "toy-manual",
            "data": "toy-rng",
            "config": "same-two-step",
            "split": "train",
            "action_dim": 30,
        }
        hook = LejepaRecovery(
            directory, resume_from=resume, pause_after_updates=1 if pause else None, identity=identity
        )
        start = hook.attach(net, sigreg, opt, scheduler, None, loader, args)
        try:
            for epoch in range(start, 10):
                net.train()
                for batch_index, vs in enumerate(loader, start=hook.next_batch_index(epoch)):
                    hook.before_input(vs)
                    loss = net(vs).square().mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    hook.after_step(epoch, batch_index, loss=loss)
                    if hook.global_step >= args.max_steps:
                        return hook.last_checkpoint
        except RecoveryPause:
            return hook.last_checkpoint

    def test_original_manual_state_and_rng_restore(self):
        with self.artifacts() as d:
            root = Path(d)
            a = self.run_branch(root / "continuous")
            p = self.run_branch(root / "pause", pause=True)
            b = self.run_branch(root / "resume", resume=p)
            x = torch.load(a, map_location="cpu", weights_only=False)
            y = torch.load(b, map_location="cpu", weights_only=False)
            for key in ("model", "optimizers", "schedulers", "scaler", "rng", "cursor"):
                same(self, x[key], y[key])
            self.assertEqual(x["progress"]["global_step"], 2)
            self.assertEqual(x["progress"]["batch_records"], y["progress"]["batch_records"])
            initial = torch.load(root / "continuous/starting.pt", map_location="cpu", weights_only=False)
            fork = torch.load(root / "pause/starting.pt", map_location="cpu", weights_only=False)
            for key in ("model", "optimizers", "schedulers", "scaler", "rng", "cursor"):
                same(self, initial[key], fork[key])
            self.assertEqual(initial["progress"]["phase"], "unstarted_reference_not_resumable")

    def test_completed_checkpoint_and_changed_sidecar_rejected(self):
        import json

        with self.artifacts() as d:
            root = Path(d)
            complete = self.run_branch(root / "complete")
            with self.assertRaisesRegex(ValueError, "completed"):
                self.run_branch(root / "bad_completed", resume=complete)
            partial = self.run_branch(root / "pause", pause=True)
            m = partial.with_suffix(partial.suffix + ".json")
            j = json.loads(m.read_text())
            j["progress"]["global_step"] = 0
            m.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError, "progress"):
                self.run_branch(root / "bad_sidecar", resume=partial)


if __name__ == "__main__":
    unittest.main()
