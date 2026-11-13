# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_training_state.py (verbatim except: imports / paths adapted to the package)
"""Fixed real-batch cursor and full training-state recovery contracts."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from jepa4r.utils import training_state as state


class CursorTests(unittest.TestCase):
    def test_resume_preserves_next_id_and_rejects_other_stream(self):
        a = state.FixedBatchCursor(['train:41', 'train:42', 'train:43'])
        self.assertEqual(a.take(), 'train:41')
        saved = a.state_dict()
        b = state.FixedBatchCursor(['train:41', 'train:42', 'train:43'])
        b.load_state_dict(saved)
        self.assertEqual(b.take(), 'train:42')
        self.assertEqual(b.state_dict()['next_batch_id'], 'train:43')
        with self.assertRaises(ValueError):
            state.FixedBatchCursor(['validation:41']).load_state_dict(saved)

    def test_identity_mismatch_is_rejected(self):
        identity = dict(model='tiny', data='sha256-data', config='sha256-cfg', split='train', action_dim=5)
        state.validate_identity(identity, dict(identity))
        for key, wrong in [('split', 'validation'), ('action_dim', 30), ('data', 'another-data')]:
            with self.assertRaises(ValueError):
                state.validate_identity(identity, dict(identity, **{key: wrong}))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires the existing PyTorch environment')
class TorchStateTests(unittest.TestCase):
    def test_full_optimizer_scheduler_rng_and_next_batch_resume(self):
        import random
        import numpy as np
        import torch
        torch.manual_seed(17)
        identity = dict(model='small-test', data='fixed-real-batch-contract', config='cfg', split='train', action_dim=5)
        a = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(a.parameters(), lr=.01)
        schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        cursor = state.FixedBatchCursor(['a', 'b', 'c'])
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        def step(model, optimizer, scheduler):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
            scheduler.step()

        step(a, opt, schedule)
        cursor.take()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            state.save_torch_state(path, model=a, optimizers={'main': opt},
                                   schedulers={'main': schedule}, scaler=None,
                                   identity=identity, cursor=cursor, progress={'step': 1, 'epoch': 0})
            expected_draws = (random.random(), np.random.random(), torch.rand(3))
            step(a, opt, schedule)
            expected = {k: v.clone() for k, v in a.state_dict().items()}
            b = torch.nn.Linear(3, 2)
            other_opt = torch.optim.AdamW(b.parameters(), lr=.01)
            other_schedule = torch.optim.lr_scheduler.StepLR(other_opt, step_size=1, gamma=.9)
            other_cursor = state.FixedBatchCursor(['a', 'b', 'c'])
            progress = state.restore_torch_state(path, model=b, optimizers={'main': other_opt},
                                                schedulers={'main': other_schedule}, scaler=None,
                                                identity=identity, cursor=other_cursor)
            self.assertEqual(progress['step'], 1)
            actual_draws = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(expected_draws[:2], actual_draws[:2])
            self.assertTrue(torch.equal(expected_draws[2], actual_draws[2]))
            self.assertEqual(other_cursor.take(), 'b')
            step(b, other_opt, other_schedule)
            self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in expected.items()))
            self.assertEqual(other_opt.state_dict()['param_groups'], opt.state_dict()['param_groups'])
            for key, expected_state in opt.state_dict()['state'].items():
                for name, value in expected_state.items():
                    actual = other_opt.state_dict()['state'][key][name]
                    self.assertTrue(torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual)
            self.assertEqual(other_schedule.state_dict(), schedule.state_dict())
            with self.assertRaises(ValueError):
                state.restore_torch_state(path, model=b, optimizers={'main': other_opt},
                                          schedulers={'main': other_schedule}, scaler=None,
                                          identity=dict(identity, split='validation'), cursor=other_cursor)


if __name__ == '__main__':
    unittest.main()
