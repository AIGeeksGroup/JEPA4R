"""Bind task, paths and encoder width around the preserved training/boundary loops."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bind_recipe, validate_cache_protocol
from ._source import VENDOR, install_impls, load_source, verify_sources


def runtime_environment(stage, action='train', cuda_visible_devices='0'):
    """The original launcher's backend/RNG bindings, before importing JAX/Torch."""
    if stage not in ('gciql', 'native-sr') or action not in ('train', 'gate', 'reload'):
        raise ValueError('Unknown downstream stage/action')
    cpu_only = stage == 'gciql' and action == 'reload'
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices:
        raise ValueError('An explicit visible CUDA device selection is required')
    environment = dict(CUDA_VISIBLE_DEVICES='' if cpu_only else cuda_visible_devices,
        JAX_PLATFORMS='cpu' if cpu_only or stage == 'native-sr' else 'cuda,cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_THREEFRY_PARTITIONABLE='false',
        JAX_DEFAULT_PRNG_IMPL='threefry2x32', JAX_ENABLE_X64='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', XLA_FLAGS='--xla_gpu_autotune_level=0')
    if stage == 'native-sr':
        environment['MUJOCO_GL'] = 'egl'
    return environment


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def install(config):
    """Keep run/update/save/gate/reload untouched; inject only input/provenance hooks."""
    install_impls()
    for name in ('reader',):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != VENDOR/(name+'.py'):
            raise RuntimeError('Conflicting preserved source import: '+name)
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from . import trainer
    from .sampler import DistanceInputs, load_cached_features, read_json
    trainer.DistanceInputs = lambda *args, **kwargs: DistanceInputs(
        *args, **kwargs, latent_dim=config['latent_dim'])
    def feature_loader(**arguments):
        arguments = dict(arguments)
        family = arguments.pop('encoder_family', config['encoder_family'])
        dim = arguments.pop('feature_dim', config['latent_dim'])
        if (family, dim) != (config['encoder_family'], config['latent_dim']):
            raise ValueError('Cache encoder family/width differs from selected recipe')
        if arguments['checkpoint_sha256'] != config['encoder_binding']['checkpoint_sha256']:
            raise ValueError('Cache checkpoint differs from selected encoder')
        manifest = read_json(Path(arguments['cache_dir'])/'cache_manifest.json', arguments['manifest_sha256'])
        validate_cache_protocol(manifest, config)
        return load_cached_features(**arguments, feature_dim=dim, encoder_family=family)
    trainer.load_cached_features = feature_loader
    original_prepare = getattr(trainer, '_unified_original_prepare', trainer.prepare)
    trainer._unified_original_prepare = original_prepare
    def prepare(cfg, shared_datasets=None):
        samplers, resources, binding = original_prepare(cfg, shared_datasets)
        binding['source_manifest'] = verify_sources()
        binding['runtime'].update(encoder_family=config['encoder_family'], latent_dim=config['latent_dim'],
                                  task_encoder_parameter_binding=True,source_string_rewrites=False)
        binding['code'][str(Path(__file__))] = sha(__file__)
        return samplers, resources, binding
    trainer.prepare = prepare
    # Boundary imports these hooks by value. Bind the same objects explicitly.
    name = '_cube_shared_original_boundary'
    spec = importlib.util.spec_from_file_location(name, VENDOR/'gciql_boundary.py')
    boundary = importlib.util.module_from_spec(spec)
    sys.modules[name] = boundary
    previous = sys.modules.get('trainer_runtime')
    sys.modules['trainer_runtime'] = trainer
    try:
        spec.loader.exec_module(boundary)
    finally:
        if previous is None:
            del sys.modules['trainer_runtime']
        else:
            sys.modules['trainer_runtime'] = previous
    return trainer, boundary


def resolve_paths(config, data_root, output):
    config = copy.deepcopy(config)
    config['output'] = str(Path(output).resolve())
    for split in ('train', 'val'):
        binding = config['data_binding']['splits'][split]
        path = Path(binding['path'])
        binding['path'] = str(path if path.is_absolute() else Path(data_root).resolve()/path)
        if config['modality'] != 'state':
            arguments = config['feature_loader_arguments'][split]
            if arguments['checkpoint_sha256'] != config['encoder_binding']['checkpoint_sha256']:
                raise ValueError('Feature cache checkpoint binding mismatch')
            if Path(arguments['source_npz']).resolve() != Path(binding['path']).resolve():
                raise ValueError('Feature cache must identify the exact bound official-state NPZ')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--stage', choices=('gciql', 'native-sr'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    from ..contracts import get_profile
    config = bind_recipe(get_profile(args.profile), args.method, json.loads(args.config.read_text()))
    if 'jax' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Run this entry in a fresh process so backend/RNG settings apply before numerical imports')
    environment = runtime_environment(args.stage, config.get('action', 'train'),
        config.get('runtime', {}).get('cuda_visible_devices', '0'))
    os.environ.update(environment)
    os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    config = resolve_paths(config, args.data_root, args.output_root)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    driver = output.with_name(output.name+'_driver')
    driver.mkdir(parents=True, exist_ok=False)
    config_path = driver/'resolved_recipe.json'
    atomic_json(config_path, config)
    state = dict(stage='preparing', started_at=time.time(), actual_exit_code=None,
                 config_sha256=sha(config_path), original_config_sha256=sha(args.config), argv=sys.argv[:],
                 environment=environment, jax_default_matmul_precision=None)
    atomic_json(driver/'status.json', state)
    code = 1
    with (driver/'stdout.log').open('x', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                trainer, boundary = install(config)
                if args.stage == 'native-sr':
                    from .native import install_native
                    native, group = install_native(config, trainer, boundary, config_path)
                    native.main(group, config['native_protocol'].get('mode', 'full'), output)
                else:
                    action = config.get('action', 'train')
                    if action == 'train':
                        result = trainer.run(str(config_path))
                        if result != 0:
                            raise RuntimeError('Preserved trainer returned '+str(result))
                    elif action == 'gate':
                        boundary.gate(config, output)
                    elif action == 'reload':
                        boundary.reload_gate(config, Path(config['gate_output']), output)
                    else:
                        raise ValueError('gciql action must be train, gate, or reload')
                state.update(stage='returned_pending_independent_acceptance', actual_exit_code=0)
                code = 0
            except BaseException:
                state.update(stage='failed', actual_exit_code=1, error=traceback.format_exc())
                raise
            finally:
                state['updated_at'] = time.time()
                atomic_json(driver/'status.json', state)
                (driver/'actual_exit_code').write_text(str(code)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
