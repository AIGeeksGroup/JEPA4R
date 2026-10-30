"""Reuse the repaired M5 Native protocol with a PLDM family and true512 inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from pldm_inputs import CHECKPOINT_SHA, DIM, FAMILY

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
GROUP = 'P2_cube2_pldm_reaudit_rgb'
NATIVE_SHA = 'e40a2ff52bd0915b7953d9011c65e02c75f9e0b2d0e454b613d18c275f6c13f0'
SHIM_SHA = '003d54b64914102b6ffe7d801ae7c06b829910016a695c6bf0fb88c050964923'
HELPER_SHA = 'a83bcf181b00d57fe5b41ef4e61c42dded0561b1ec9d4dc2c7a1c64dfdb8bc3a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_latent(value, label):
    value = np.asarray(value)
    if value.shape != (DIM,) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f'{label}: PLDM requires finite FP32[512]')
    return value


class OnlinePLDMEncoder:
    def __init__(self, group_config):
        binding = group_config['encoder']
        assert binding['family'] == FAMILY and binding['latent_dim'] == DIM
        assert binding['checkpoint_sha256'] == CHECKPOINT_SHA
        path = Path(binding['adapter']['path'])
        assert sha(path) == binding['adapter']['sha256']
        spec = importlib.util.spec_from_file_location('_P3_cache_encoder', path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        assert adapter.FAMILY == FAMILY and adapter.DIM == DIM
        import torch
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Native PLDM encoder requires CUDA')
        self.torch, self.adapter = torch, adapter
        self.device = binding['selected_preprocess_device']
        assert self.device in ('cpu', 'cuda')
        selection = binding['preprocessing_selection']
        assert sha(selection['path']) == selection['sha256']
        assert json.loads(Path(selection['path']).read_text())['selected_device'] == self.device
        self.model = adapter.load_frozen(group_config['wm'], device='cuda')
        self.calls = 0

    def encode(self, value):
        if value.shape != (224, 224, 3) or value.dtype != np.uint8:
            raise ValueError('Native RGB must be uint8 HWC224')
        pixels = self.torch.from_numpy(np.ascontiguousarray(value[None]))
        features = self.adapter.encode_uint8(self.model, pixels, preprocess_device=self.device)
        self.calls += 1
        return validate_latent(features[0].detach().cpu().numpy(), 'online')


def install_core(path, group_config):
    path = Path(path)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == NATIVE_SHA
    core = types.ModuleType('_P3_repaired_native')
    core.__file__ = str(path)
    sys.modules[core.__name__] = core
    exec(compile(raw, str(path), 'exec'), core.__dict__)
    shim_path = Path(__file__).with_name('cube2_successor_v1')/'visual_sr_cube2.py'
    assert sha(shim_path) == SHIM_SHA
    shim = types.ModuleType('_P3_original_Cube3_registration')
    shim.__file__ = str(shim_path)
    exec(compile(shim_path.read_bytes(), str(shim_path), 'exec'), shim.__dict__)
    registered = copy.deepcopy(shim.GROUPS['G05'])
    registered.update(copy.deepcopy(group_config))
    group_config = registered
    assert (group_config['cube_count'], group_config['state_dim'], group_config['mode'], group_config['max_episode_steps']) == (2,37,'rgb',500)
    for key in ('wm', 'training', 'acceptance', 'gate', 'final'):
        if key in group_config:
            group_config[key] = Path(group_config[key])
    core.GROUPS = {GROUP: group_config}
    core.validate_latent = validate_latent
    core.OnlineEncoder = lambda _group: OnlinePLDMEncoder(group_config)
    return core


def configure_physics():
    path = ROOT/'assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_runtime_helpers.py'
    assert sha(path) == HELPER_SHA
    spec = importlib.util.spec_from_file_location('_P3_original_M5_physics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_numpy_physics_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--mode', choices=('tiny', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads((args.case/'functional_v1/NATIVE_BINDING_v1.json').read_text())
    assert binding['actual_exit_code'] == 0 and binding['group_id'] == GROUP
    for path, digest in binding['source_pins'].items():
        assert sha(path) == digest, path
    physics = configure_physics()
    core = install_core(ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py', binding['group'])
    if args.mode == 'tiny':
        core.GROUPS[GROUP]['max_episode_steps'] = 4
    core.main(GROUP, 'diagnostic' if args.mode == 'tiny' else 'full', args.output)
    import torch
    assert torch.get_num_threads() == 1
    core.atomic_json(args.output/'native_source_binding.json', dict(actual_exit_code=0,
        binding_sha256=sha(args.case/'functional_v1/NATIVE_BINDING_v1.json'), encoder_family=FAMILY,
        latent_dim=DIM, selected_preprocess_device=binding['group']['encoder']['selected_preprocess_device'],
        source_sha256=sha(__file__), original_Native_sha256=NATIVE_SHA,
        encoding_protocol=core.ENCODING_PROTOCOL, numpy_physics=physics,
        torch_num_threads=torch.get_num_threads(), torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
