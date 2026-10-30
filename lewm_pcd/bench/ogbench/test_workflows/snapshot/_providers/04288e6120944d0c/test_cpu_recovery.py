"""CPU tensors only: never constructs the training model or touches its dataset."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

CASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resume_helpers_cpu_test', CASE / 'resume_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def payload():
    return {'backbone': {'w': torch.arange(4, dtype=torch.float32)},
            'head': {'w': torch.tensor([-0.0, 1.0])},
            'proj': {'counter': torch.tensor(3, dtype=torch.int64)}}


class CPURecovery(unittest.TestCase):
    def test_verified_save_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, shared = root / 'stage', root / 'shared'
            stage.mkdir(); shared.mkdir()
            receipt = helpers.verified_export(payload(), stage, shared, 'weights_epoch_8.pt')
            self.assertEqual(receipt['stage']['sha256'], receipt['primary']['sha256'])
            self.assertGreater(receipt['primary']['bytes'], 0)
            self.assertTrue((shared / 'weights_epoch_8.pt.receipt.json').is_file())
            with self.assertRaises(FileExistsError):
                helpers.verified_export(payload(), stage, shared, 'weights_epoch_8.pt')

    def test_empty_write_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, shared = root / 'stage', root / 'shared'
            stage.mkdir(); shared.mkdir()
            with patch.object(torch, 'save', side_effect=lambda value, stream: None):
                with self.assertRaises((ValueError, OSError)):
                    helpers.verified_export(payload(), stage, shared, 'weights_epoch_8.pt')
            self.assertFalse((shared / 'weights_epoch_8.pt').exists())
            self.assertTrue(list(stage.glob('*.partial')))

    def test_shared_mismatch_retains_valid_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, shared = root / 'stage', root / 'shared'
            stage.mkdir(); shared.mkdir()
            with patch.object(helpers.shutil, 'copyfileobj', side_effect=lambda source, target, **kwargs: target.write(b'wrong')):
                with self.assertRaises(ValueError):
                    helpers.verified_export(payload(), stage, shared, 'weights_epoch_9.pt')
            helpers.audit_checkpoint(stage / 'weights_epoch_9.pt')
            self.assertFalse((shared / 'weights_epoch_9.pt').exists())

    def test_finite_and_byte_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            broken = payload()
            broken['head']['w'][1] = float('nan')
            torch.save(broken, path)
            with self.assertRaises(ValueError):
                helpers.audit_checkpoint(path)
        self.assertFalse(helpers.tensor_bytes_equal(torch.tensor([-0.0]), torch.tensor([0.0])))

    def test_original_scheduler_position_without_optimizer_steps(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.AdamW([{'params': [parameter], 'lr': .002, 'weight_decay': .05}])
        s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=.01, total_iters=19531)
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=195310-19531, eta_min=.001)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[19531])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(resume_model=Path('parent.pt'), start_epoch=7, epochs=10)
            binding = {'batches_per_epoch': 19531, 'start_epoch': 7, 'total_epochs': 10}
            tracker = helpers.prepare_resume(scheduler, opt, None, args, 19531, Path(tmp), {'verified': True}, binding)
            self.assertEqual(scheduler.last_epoch, 136717)
            self.assertAlmostEqual(scheduler.get_last_lr()[0], .00125, places=12)
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(list(range(args.start_epoch, args.epochs)), [7, 8, 9])
            self.assertEqual([f'weights_epoch_{e+1}.pt' for e in range(7, 10)],
                             ['weights_epoch_8.pt', 'weights_epoch_9.pt', 'weights_epoch_10.pt'])
            self.assertEqual(tracker['logical_steps_before'], 136717)

    def test_default_resume_interface_is_off(self):
        args = SimpleNamespace(resume_model=None, start_epoch=0, epochs=10, verified_export_root=None)
        helpers.validate_resume_args(args)
        self.assertIsNone(helpers.load_resume_model(None, args, None, {}))
        self.assertIsNone(helpers.prepare_resume(None, None, None, args, 19531, None, None, {}))
        helpers.record_first_update(None, None, None)
        with self.assertRaises(ValueError):
            helpers.validate_resume_args(SimpleNamespace(resume_model=None, start_epoch=7, epochs=10, verified_export_root=None))

    def test_scalar_adam_hook_proves_actual_step_one(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        net = SimpleNamespace(named_parameters=lambda: [('scalar_cpu_test_parameter', parameter)])
        opt = torch.optim.AdamW([{'params': [parameter], 'lr': .002, 'weight_decay': .05}])
        s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=.01, total_iters=19531)
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=195310-19531, eta_min=.001)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[19531])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(resume_model=Path('parent.pt'), start_epoch=7, epochs=10)
            binding = {'batches_per_epoch': 19531, 'start_epoch': 7, 'total_epochs': 10}
            tracker = helpers.prepare_resume(scheduler, opt, net, args, 19531, Path(tmp), {'verified': True}, binding)
            parameter.grad = torch.ones_like(parameter)  # Scalar CPU test; no NN forward/backward.
            opt.step()
            scheduler.step()
            helpers.record_first_update(tracker, scheduler, Path(tmp))
            receipt = json.loads((Path(tmp) / 'first_actual_update_receipt.json').read_text())
            self.assertEqual(receipt['adam_step_min'], 1)
            self.assertEqual(receipt['adam_step_max'], 1)
            self.assertEqual(receipt['scheduler_last_epoch'], 136718)
            self.assertTrue(receipt['weight_bytes_changed'])
            self.assertTrue(receipt['gradient_finite'])

    def test_parent_strict_load_all_bytes(self):
        net = SimpleNamespace(backbone=torch.nn.Linear(2, 2), head=torch.nn.Linear(2, 1), proj=torch.nn.Linear(1, 1))
        expected = {name: getattr(net, name).state_dict() for name in ['backbone', 'head', 'proj']}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / 'parent.pt'
            torch.save(expected, parent)
            metadata, _ = helpers.audit_checkpoint(parent)
            for name in ['backbone', 'head', 'proj']:
                for parameter in getattr(net, name).parameters():
                    parameter.data.zero_()
            # torch state dictionaries are references; reload the real saved payload.
            args = SimpleNamespace(resume_model=parent)
            binding = {'parent_model': str(parent), 'parent_sha256': metadata['sha256'], 'parent_bytes': metadata['bytes']}
            receipt = helpers.load_resume_model(net, args, root, binding)
            self.assertTrue(receipt['all_loaded_tensor_bytes_equal'])
            self.assertFalse(receipt['exact_resume'])


if __name__ == '__main__':
    unittest.main()
