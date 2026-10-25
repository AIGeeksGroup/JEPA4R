"""Inspect training commands without importing torch or starting a child.

Pass ``build_environment(...)`` to subprocess together with ``build_command``.
Both child entries require a new output directory and keep their own log/status.
"""
import os
from pathlib import Path
import sys

from .contracts import get_profile


def build_command(profile_name, method, data_root, output_root, workers=None,
                  prefetch=None, python=None):
    profile = get_profile(profile_name)
    if profile['source'] != 'self_collected':
        raise ValueError('training requires a sealed self-collected Cube profile')
    if method not in ('pldm', 'lejepa'):
        raise ValueError('method must be pldm or lejepa (Tiny)')
    if workers is not None and (workers < 0 or (method == 'lejepa' and workers == 0)):
        raise ValueError('PLDM workers must be nonnegative; native Tiny requires positive workers')
    if prefetch is not None and prefetch < 1:
        raise ValueError('prefetch must be positive')
    entry = 'bench.ogbench.pldm.train' if method == 'pldm' else 'bench.ogbench.tiny'
    command = [python or sys.executable, '-m', entry, '--profile', profile_name,
               '--data-root', str(Path(data_root).resolve()),
               '--output', str(Path(output_root).resolve())]
    if method == 'pldm':
        command += ['--phase', 'formal', '--mode', 'train']
    if workers is not None:
        command += ['--workers', str(workers)]
    if prefetch is not None:
        command += ['--prefetch', str(prefetch)]
    return command


def build_environment(data_root, output_root, environ=None):
    """Use absolute sealed data paths; place native outputs in the fresh root."""
    env = dict(os.environ if environ is None else environ)
    code = Path(__file__).resolve().parents[2]
    paths = [str(code), str(code / 'third_party/stable-worldmodel')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['STABLEWM_HOME'] = str(Path(output_root).resolve())
    env['SPT_CACHE_DIR'] = str(Path(output_root).resolve() / 'checkpoints/_spt_cache')
    return env
