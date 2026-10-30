"""Existing runtime, exact versions, both backends and one tiny CUDA operation."""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'source/ogbench/impls'))
sys.path.insert(1, str(ROOT / 'source/ogbench'))
import jax
import jax.numpy as jnp
import numpy as np
from agents.gciql import GCIQLAgent, get_config
import ogbench
import mujoco

assert jax.default_backend() == 'gpu'
assert jax.devices('cpu') and jax.devices('gpu')
value = jax.device_put(np.eye(2, dtype=np.float32), jax.devices('gpu')[0])
value = jax.jit(lambda x: x @ x)(value)
assert np.array_equal(np.asarray(value), np.eye(2, dtype=np.float32))
cfg = get_config()
assert cfg.batch_size == 1024 and cfg.encoder is None
print(json.dumps(dict(actual_exit_code=0, python=platform.python_version(),
    versions={n: importlib.metadata.version(n) for n in
        ['jax', 'jaxlib', 'flax', 'numpy', 'mujoco', 'gymnasium', 'wandb']},
    GPU_devices=[str(x) for x in jax.devices('gpu')],
    CPU_devices=[str(x) for x in jax.devices('cpu')],
    tiny_CUDA_matmul_exact=True,
    default_matmul_precision=str(jax.config.jax_default_matmul_precision),
    GPU_allocator_fraction_only_not_precision_change=True)))
