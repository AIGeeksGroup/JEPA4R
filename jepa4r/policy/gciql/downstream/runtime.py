import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from . import validate_cache_protocol
from ._source import VENDOR, install_impls

def runtime_environment(stage, action='train', cuda_visible_devices='0'):
    """The original launcher's backend/RNG bindings, before importing JAX/Torch."""
    if stage not in ('gciql', 'native-sr') or action not in ('train', 'gate', 'reload'):
        raise ValueError('Unknown downstream stage/action')
    cpu_only = stage == 'gciql' and action == 'reload'
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices:
        raise ValueError('An explicit visible CUDA device selection is required')
    environment = dict(
        CUDA_VISIBLE_DEVICES='' if cpu_only else cuda_visible_devices,
        JAX_PLATFORMS='cpu' if cpu_only or stage == 'native-sr' else 'cuda,cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_THREEFRY_PARTITIONABLE='false',
        JAX_DEFAULT_PRNG_IMPL='threefry2x32',
        JAX_ENABLE_X64='false',
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0',
    )
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def install(config):
    """Keep run/update/save/gate/reload untouched; inject only input/provenance hooks."""
    install_impls()
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from . import trainer
    from .sampler import DistanceInputs, load_cached_features, read_json
    trainer.DistanceInputs = lambda *args, **kwargs: DistanceInputs(*args, **kwargs, latent_dim=config['latent_dim'])

    def feature_loader(**arguments):
        arguments = dict(arguments)
        family = arguments.pop('encoder_family', config['encoder_family'])
        dim = arguments.pop('feature_dim', config['latent_dim'])
        if (family, dim) != (config['encoder_family'], config['latent_dim']):
            raise ValueError('Cache encoder family/width differs from selected recipe')
        if arguments['checkpoint_sha256'] != config['encoder_binding']['checkpoint_sha256']:
            raise ValueError('Cache checkpoint differs from selected encoder')
        manifest = read_json(Path(arguments['cache_dir']) / 'cache_manifest.json', arguments['manifest_sha256'])
        validate_cache_protocol(manifest, config)
        return load_cached_features(**arguments, feature_dim=dim, encoder_family=family)
    trainer.load_cached_features = feature_loader
    original_prepare = getattr(trainer, '_unified_original_prepare', trainer.prepare)
    trainer._unified_original_prepare = original_prepare

    def prepare(cfg, shared_datasets=None):
        samplers, resources, binding = original_prepare(cfg, shared_datasets)
        binding['runtime'].update(
            encoder_family=config['encoder_family'],
            latent_dim=config['latent_dim'],
            task_encoder_parameter_binding=True,
            source_string_rewrites=False,
        )
        binding['code'][str(Path(__file__))] = sha(__file__)
        return (samplers, resources, binding)
    trainer.prepare = prepare
    name = '_cube_shared_original_boundary'
    spec = importlib.util.spec_from_file_location(name, VENDOR / 'boundary.py')
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
    return (trainer, boundary)
