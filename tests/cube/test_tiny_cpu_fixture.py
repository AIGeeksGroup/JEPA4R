# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_tiny_cpu_fixture.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns)
"""Explicit numerical-only CPU routing preserves original factory arithmetic."""
import importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql import tiny_cpu_fixture as fixture


class ConfigTests(unittest.TestCase):
    def config(self):
        return dict(
            mode='smoke',
            model='lejepa',
            precision='fp32',
            stop_after=1,
            seed=0,
            numerical_fixture=dict(
                schema='cube_tiny_cpu_fp32_M3_v1', role='M3', configured_batch_size=256, actual_fixed_batch_size=2
            ),
        )
    def test_config_requires_explicit_M3_B2_and_retains_configured_256(self):
        self.assertEqual(fixture.validate_config(self.config()),2)
        for field,value in [('role','T4'),('configured_batch_size',2),('actual_fixed_batch_size',256)]:
            cfg=self.config();cfg['numerical_fixture'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):fixture.validate_config(cfg)
    def test_unmarked_fp32_or_two_updates_cannot_select_device_fixture(self):
        cfg=self.config();del cfg['numerical_fixture']
        with self.assertRaises(ValueError):fixture.validate_config(cfg)
        cfg=self.config();cfg['stop_after']=2
        with self.assertRaises(ValueError):fixture.validate_config(cfg)
    def test_T4_requires_immutable_original_recipe_and_first_checkpoint_provenance(self):
        cfg=self.config();cfg['stop_after']=2
        cfg['numerical_fixture'].update(schema='cube_tiny_cpu_fp32_fixed_batch_v2',role='T4')
        with self.assertRaisesRegex(ValueError,'provenance'):fixture.validate_config(cfg)


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires existing Torch CPU')
class DeviceTests(ConfigTests):
    def test_factory_and_randn_are_original_but_explicit_device_is_CPU(self):
        import torch
        class Encoder(torch.nn.Linear):
            def __init__(self):super().__init__(3,2)
        class SIGReg(torch.nn.Module):
            def forward(self,value):return module.torch.randn(2,2,device='cuda')+value
        module=SimpleNamespace(Encoder=Encoder,SIGReg=SIGReg,torch=torch,autocast=torch.amp.autocast,
                               GradScaler=torch.amp.GradScaler)
        originals=dict(vars(module))
        original_tensor_to=torch.Tensor.to
        with fixture.cpu_fp32(module,self.config())as metadata:
            net=module.Encoder().to('cuda');self.assertEqual(net.weight.device.type,'cpu')
            self.assertEqual(torch.ones(2).to('cuda',non_blocking=True).device.type,'cpu')
            value=module.SIGReg().to('cuda')(torch.zeros(2,2));self.assertEqual(value.device.type,'cpu')
            self.assertFalse(module.GradScaler(enabled=True).is_enabled())
            self.assertEqual(metadata['configured_batch_size'],256)
            self.assertEqual(metadata['actual_fixed_batch_size'],2)
        for name,value in originals.items():self.assertIs(getattr(module,name),value)
        self.assertIs(torch.Tensor.to,original_tensor_to)


if __name__=='__main__':unittest.main()
