# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/downstream/native.py (verbatim except: imports / paths adapted
# to the package; lines wrapped at 120 columns)
import importlib.util
import json
from pathlib import Path
import sys
from ._source import VENDOR, IMPLS, verify_sources, load_source
from .runtime import sha, atomic_json

def load_adapter(binding):
    record = binding['adapter']
    path = Path(record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('Encoder adapter source SHA mismatch')
    spec = importlib.util.spec_from_file_location('_cube_bound_encoder_adapter', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if (getattr(module, 'FAMILY', None), getattr(module, 'DIM', None)) != (binding['family'], binding['latent_dim']):
        raise ValueError('Encoder adapter family/output width mismatch')
    return module

class OnlineEncoder:

    def __init__(self, config):
        import torch
        self.config, self.binding = (config, config['encoder_binding'])
        binding = self.binding
        checkpoint = Path(binding['checkpoint'])
        if sha(checkpoint) != binding['checkpoint_sha256']:
            raise ValueError('Online encoder checkpoint SHA mismatch')
        self.torch, self.calls = (torch, 0)
        self.device = binding.get('device', 'cuda')
        if self.device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Preserved Native visual protocol requires a CUDA frozen encoder')
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        family = binding['family']
        if family == 'lejepa':
            from ...bc import jepa_baselines
            if sha(jepa_baselines.__file__) != binding['preprocessing']['source_sha256']:
                raise ValueError('Tiny loader/preprocessing source SHA mismatch')
            epoch = binding['epoch']
            if checkpoint != checkpoint.parent / f'weights_epoch_{epoch}.pt':
                raise ValueError('Tiny canonical checkpoint filename/epoch mismatch')
            self.model, dim, self.transform, _ = jepa_baselines.load_lejepa(str(checkpoint.parent), epoch)
            if dim != binding['latent_dim']:
                raise ValueError('Tiny head output width differs from bound recipe')
            self.model = self.model.to(self.device).eval().requires_grad_(False)
        else:
            self.adapter = load_adapter(binding)
            if family == 'pldm' and binding['preprocessing']['source_sha256'] != binding['adapter']['sha256']:
                raise ValueError('Frozen adapter must own the bound preprocessing')
            self.model = self.adapter.load_frozen(checkpoint, device=self.device)
            if family == 'lewm':
                path = Path(binding['preprocessing']['source_path'])
                if sha(path) != binding['preprocessing']['source_sha256']:
                    raise ValueError('LeWM Native preprocessing source SHA mismatch')
                spec = importlib.util.spec_from_file_location('_cube_native_lewm_preprocess', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.img_preprocess = module.get_img_preprocessor('pixels', 'pixels', 224)
                pre = binding['preprocessing']
                self.pcd_preprocess = module.get_pcd_preprocessor(
                    center=pre['center'] if config['modality'] == 'pcd' else (0.425, 0.0, 0.2),
                    scale=pre['scale'] if config['modality'] == 'pcd' else 4.0,
                    use_rgb=True,
                )
        if self.model.training or any((p.requires_grad for p in self.model.parameters())):
            raise ValueError('Online world-model encoder must remain frozen in eval mode')

    def encode(self, value):
        import numpy as np
        binding, torch = (self.binding, self.torch)
        family = binding['family']
        if self.config['modality'] == 'rgb':
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError('Native RGB must be uint8 HWC224')
            pixels = torch.from_numpy(np.ascontiguousarray(value[None]))
            if family == 'pldm':
                features = self.adapter.encode_uint8(
                    self.model, pixels, preprocess_device=binding['preprocessing']['device']
                )
            elif family == 'lejepa':
                from PIL import Image
                prepared = self.transform(Image.fromarray(value)).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    features = self.model(prepared)
            else:
                batch = self.img_preprocess({'pixels': pixels.permute(0, 3, 1, 2)})
                features = self.adapter.encode(self.model, batch['pixels'].to(self.device))
        else:
            if value.shape != (1200, 6) or value.dtype != np.float32 or (not np.isfinite(value).all()):
                raise ValueError('Native PCD must be finite FP32 XYZRGB[1200,6]')
            xyz = torch.from_numpy(value[None, :, :3].copy())
            color = np.clip(np.rint(value[None, :, 3:]), 0, 255).astype(np.uint8)
            batch = self.pcd_preprocess({'pcd_xyz': xyz, 'pcd_rgb': torch.from_numpy(color)})
            features = self.adapter.encode(self.model, batch['pcd'].to(self.device))
        latent = features[0].detach().cpu().numpy()
        if latent.shape != (binding['latent_dim'],) or latent.dtype != np.float32 or (not np.isfinite(latent).all()):
            raise ValueError('Online encoder did not return its actual bound FP32 output width')
        self.calls += 1
        return latent

def install_native(config, trainer, boundary, config_path):
    from . import native_core as core
    if config['modality'] == 'state':
        raise ValueError(
            'This preserved visual Native loop requires a visual method; use the original official state Native entry'
        )
    trainer.torch.set_num_threads(1)
    physics = load_source('_cube_original_native_physics', 'physics.py')
    physics_binding = physics.configure_numpy_physics_runtime()
    atomic_json(Path(config_path).parent / 'numpy_physics_runtime.json', physics_binding)
    native, encoder = (config['native_protocol'], config['encoder_binding'])
    if native.get('mode', 'full') not in ('diagnostic', 'full'):
        raise ValueError('Original Native mode is diagnostic or full')
    policy = native['policy']
    for key in ('training', 'acceptance', 'gate'):
        if sha(policy[key]) != policy[key + '_sha256']:
            raise ValueError('Native policy artifact SHA mismatch: ' + key)
    training = json.loads(Path(policy['training']).read_text())
    if training['input_contract'] != config['input_contract']:
        raise ValueError('Native and trained policy input dimensions differ')
    if training['encoder_binding'] != encoder:
        raise ValueError('Native and cached policy encoder/preprocessing bindings differ')
    for split in ('train', 'val'):
        if (
            training['feature_loader_arguments'][split]['manifest_sha256']
            != config['feature_loader_arguments'][split]['manifest_sha256']
        ):
            raise ValueError('Native recipe identifies a different training cache')
    group = 'cube' + str(config['cube_count']) + '_' + config['method']
    core.ROOT = VENDOR
    core.CASE = VENDOR
    core.LATENT_DIM = config['latent_dim']
    camera = dict(native['camera'])
    cfg = dict(
        cube_count=config['cube_count'],
        state_dim=config['state_dim'],
        current_dim=config['input_contract']['current_dim'],
        goal_dim=config['input_contract']['goal_dim'],
        max_episode_steps=native['max_episode_steps'],
        mode=config['modality'],
        camera=camera,
        wm=Path(encoder['checkpoint']),
        wm_sha256=encoder['checkpoint_sha256'],
        manifest_sha256=policy['manifest_sha256'],
    )
    for key in ('training', 'acceptance', 'gate', 'final'):
        cfg[key] = Path(policy[key])
    for key in ('training_sha256', 'acceptance_sha256', 'gate_sha256'):
        cfg[key] = policy[key]
    if config['modality'] == 'pcd':
        cfg.update(pcd_crop_box=camera['pointcloud_crop_box'], record_capture_receipts=True)
    core.GROUPS = {group: cfg}
    core.OnlineEncoder = lambda _group: OnlineEncoder(config)
    existing = sys.modules.get('gciql_boundary')
    if existing is not None and existing is not boundary:
        raise RuntimeError('Conflicting Native GCIQL boundary import')
    sys.modules['gciql_boundary'] = boundary
    reference = Path(config_path).parent / 'evaluator_reference.json'
    manifest = verify_sources()
    evaluator = IMPLS / 'utils/evaluation.py'
    atomic_json(
        reference, dict(source_pins={str(evaluator): manifest['files']['ogbench/impls/utils/evaluation.py']['sha256']})
    )
    core.EVALUATOR_REFERENCE = reference
    return (core, group)
