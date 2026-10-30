"""Restore the original M5 CPU thread policy and invoke the existing SR entry."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
import visual_sr as core
from cube3_successor_v1 import visual_sr_cube3
visual_sr_cube3.install(core)
HELPER = ROOT / 'assets/imports/single_cube_rl3_101632_20261018_v1/source_layout/coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_runtime_helpers.py'


def configure_physics():
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == 'a83bcf181b00d57fe5b41ef4e61c42dded0561b1ec9d4dc2c7a1c64dfdb8bc3a'
    spec = importlib.util.spec_from_file_location('_original_M5_runtime', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_numpy_physics_runtime()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('G05', 'G06', 'G08', 'G09'), required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    physics = configure_physics()
    if args.mode == 'smoke':
        core.GROUPS[args.group] = dict(core.GROUPS[args.group], max_episode_steps=4)
    core.main(args.group, 'diagnostic' if args.mode == 'smoke' else 'full', args.output)
    import torch
    assert torch.get_num_threads() == 1
    core.atomic_json(args.output / 'runtime_threads.json', dict(
        numpy_physics=physics, torch_num_threads=torch.get_num_threads(),
        torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        environment={k: os.environ[k] for k in
                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JAX_PLATFORMS')}))
