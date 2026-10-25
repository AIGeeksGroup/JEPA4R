"""Sealed RGB/PCD data routes around the preserved production LeWM entry.

Command inspection imports only the standard library. Execution calls the
original Hydra function with an explicit train/validation recipe; the original
JEPA, loss, optimizer, scheduler, callback and SPT/Lightning loop are reused.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


CODE = Path(__file__).resolve().parents[2]
NATIVE_DIRECTORY = Path(__file__).with_name('lewm_native')
NATIVE_TRAIN = NATIVE_DIRECTORY / 'train.py'
NATIVE_SHA256 = '5a91153791f2d3a46eb6d463ebf97cc20d6777b6fdd83b6ee64c25646dc1c699'
ORIGINAL_TRAIN = CODE / 'train.py'
ORIGINAL_SHA256 = '733564c824ce4023abdcdcf0a65ad72f5b71efe9d21d7958a466f87c7cdfe6e5'
RECIPE_SHA256 = {
    '2d': '72c5ac2434c8212b8a7c2441a00b312293ba9a1810fa9be7eaa54ee29e3ac608',
    '3d': 'b60ac2e2b83fb83e03baee29975e361786e54dc5ce56270f9e5a0a1340d2c558',
}
SHARED_SHA256 = {
    'jepa.py': '7e6680f9941268b13526082fa888a46de7dca9385bfc837e77ee649e5c034135',
    'module.py': '0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620',
    'utils.py': 'a6a386523779ede2ef4453b23198e46e371a0c73200e8f957ae272b5d7b63259',
    'encoders/__init__.py': '90e5ab276c362b6a5c1e4cd2c2363335eba8c5c34bf72fa48b4942ffc4ac5788',
    'encoders/vit.py': '829efe4f2c11884253d236050a9fd2146732c91393547f92475a862c8fcd669a',
    'encoders/pointnet_dexart.py': '78a545f7f41f42ed9714101d203a1ec14875495cf43fb86e72ff729fb29b184b',
}
METHOD_MODES = {'lewm_rgb': '2d', 'lewm_pcd': '3d'}


def _resources(workers, prefetch):
    if workers is not None and (type(workers) is not int or workers < 0):
        raise ValueError('workers must be a nonnegative integer')
    if prefetch is not None and (type(prefetch) is not int or prefetch < 1):
        raise ValueError('prefetch must be a positive integer')


def _single_protocol(profile_name, single_original):
    if (profile_name == 'single200') != bool(single_original):
        raise ValueError('Single LeWM requires the explicit original protocol; it is invalid for other profiles')


def load_recipe(mode, config_path=None, *, single_original=False):
    if single_original:
        from .single_original import load_single_recipe
        return load_single_recipe(mode, config_path)
    if mode not in ('2d', '3d'):
        raise ValueError('mode must be 2d or 3d')
    raw = (NATIVE_DIRECTORY / (mode + '.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECIPE_SHA256[mode]:
        raise ValueError('preserved LeWM recipe source SHA differs')
    recipe = json.loads(raw)
    if config_path is not None:
        path = Path(config_path)
        if path.suffix.lower() != '.json':
            raise ValueError('LeWM configuration must be a frozen recipe JSON')
        supplied = json.loads(path.read_text(encoding='utf8'))
        if supplied != recipe:
            raise ValueError('LeWM configuration must equal the packaged ' + mode + ' recipe; use resource flags for workers/prefetch')
    return recipe


def build_command(profile_name, method, data_root, output_root, workers=None,
                  prefetch=None, python=None, config_path=None, *, single_original=False):
    from .lewm_data import describe
    _single_protocol(profile_name, single_original)
    if method not in METHOD_MODES:
        raise ValueError('method must be lewm_rgb or lewm_pcd')
    mode = METHOD_MODES[method]
    describe(profile_name, mode, data_root)
    _resources(workers, prefetch)
    load_recipe(mode, config_path, single_original=single_original)
    command = [python or sys.executable, '-m', 'bench.ogbench.lewm', '--profile', profile_name,
               '--mode', mode, '--data-root', str(Path(data_root).resolve()),
               '--output', str(Path(output_root).resolve())]
    if single_original:
        command += ['--single-original-protocol']
    for flag, value in (('--workers', workers), ('--prefetch', prefetch)):
        if value is not None:
            command += [flag, str(value)]
    if config_path is not None:
        command += ['--config', str(Path(config_path).resolve())]
    return command


def build_config(profile_name, mode, data_root, workers=None, prefetch=None, config_path=None, *,
                 single_original=False):
    from .lewm_data import describe
    _single_protocol(profile_name, single_original)
    _resources(workers, prefetch)
    binding = describe(profile_name, mode, data_root)
    cfg = load_recipe(mode, config_path, single_original=single_original)
    cfg['output_model_name'] = cfg['subdir'] = 'lewm_' + profile_name + '_' + mode + '_seed' + str(cfg['seed'])
    if single_original:
        cfg['data']['dataset']['name'] = binding['dataset_path']
    else:
        for role, path_key in (('dataset', 'train_path'), ('validation_dataset', 'validation_path')):
            cfg['data'][role] = {
                'name': binding[path_key], 'format': 'bench_lewm_sealed',
                'num_steps': binding['num_steps'], 'frameskip': binding['frameskip'],
                'keys_to_load': binding['keys_to_load'], 'keys_to_cache': binding['keys_to_cache'],
                'raw_chunk_cache_nbytes': 8388608,
            }
    if binding['action_chunk_dim'] != cfg['model']['action_encoder']['input_dim']:
        raise ValueError('sealed action chunk disagrees with the accepted LeWM recipe')
    if workers is not None:
        cfg['loader']['num_workers'] = workers
    cfg['num_workers'] = cfg['loader']['num_workers']
    if not single_original:
        cfg['validation_num_workers'] = cfg['num_workers']
    if cfg['num_workers'] == 0:
        cfg['loader'].pop('prefetch_factor', None)
        cfg['loader'].pop('persistent_workers', None)
    elif prefetch is not None:
        cfg['loader']['prefetch_factor'] = prefetch
    return cfg


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def pinned_native(*, single_original=False):
    """Resolve historical bare imports to this checkout, then restore the caller."""
    train_path, train_sha = ((ORIGINAL_TRAIN, ORIGINAL_SHA256) if single_original else
                             (NATIVE_TRAIN, NATIVE_SHA256))
    for path, expected in [(train_path, train_sha)] + [(CODE / p, h) for p, h in SHARED_SHA256.items()]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('preserved LeWM source differs: ' + str(path))
    names = ('module', 'utils', 'jepa', 'encoders', 'bench.ogbench.lewm_native.train')
    old = {n: m for n, m in sys.modules.items() if n in names or n.startswith('encoders.')}
    for name in old:
        del sys.modules[name]
    try:
        _load('module', CODE / 'module.py')
        _load('utils', CODE / 'utils.py')
        _load('encoders', CODE / 'encoders/__init__.py', package=True)
        _load('jepa', CODE / 'jepa.py')
        entry = _load('bench.ogbench.lewm_native.train', train_path)
        yield entry
    finally:
        for name in list(sys.modules):
            if name in names or name.startswith('encoders.'):
                del sys.modules[name]
        sys.modules.update(old)


@contextmanager
def route_datasets(native, binding, data_root):
    """Route only the two bound paths; preserve the native transforms and loaders."""
    from .lewm_data import make_reader
    original = native.swm.data.load_dataset
    if binding.get('protocol') == 'single_original_random_split_v1':
        from .single_original import make_single_original_reader
        bound_path = Path(binding['dataset_path']).resolve()
        def load_dataset(name, **kwargs):
            if Path(name).resolve() != bound_path:
                raise ValueError('unbound dataset path is outside the original Single source')
            return make_single_original_reader(binding['mode'], data_root, loader=original, **kwargs)
    else:
        paths = {Path(binding['train_path']).resolve(): 'train',
                 Path(binding['validation_path']).resolve(): 'validation'}
        if len(paths) != 2:
            raise ValueError('sealed train and validation paths must be distinct')
        def load_dataset(name, **kwargs):
            split = paths.get(Path(name).resolve())
            if split is None:
                raise ValueError('unbound dataset path is outside the sealed train/validation pair')
            return make_reader(binding['profile_name'], binding['mode'], split, data_root, **kwargs)
    native.swm.data.load_dataset = load_dataset
    try:
        yield
    finally:
        native.swm.data.load_dataset = original


def invoke_native(cfg, binding, data_root, *, single_original=False):
    """Call the preserved Hydra function, with no alternate training loop."""
    from omegaconf import OmegaConf
    _single_protocol(binding['profile_name'], single_original)
    with pinned_native(single_original=single_original) as native:
        with route_datasets(native, binding, data_root):
            native.run.__wrapped__(OmegaConf.create(cfg))


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def run_job(profile_name, mode, data_root, output_root, workers=None, prefetch=None, config_path=None, *,
            single_original=False):
    from .lewm_data import validate_binding
    from .training import build_environment
    data_root, output = Path(data_root).resolve(), Path(output_root).resolve()
    # Validate the public command before claiming an output directory.
    cfg = build_config(profile_name, mode, data_root, workers, prefetch, config_path,
                       single_original=single_original)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='lewm_validating', started_at=_stamp(), updated_at=_stamp(),
                 profile_name=profile_name, mode=mode, config=cfg, argv=sys.argv[:],
                 native_train_sha256=ORIGINAL_SHA256 if single_original else NATIVE_SHA256,
                 protocol='single_original_random_split_v1' if single_original else 'sealed_train_validation',
                 completed_count=0, failed_count=0,
                 actual_exit_code=None, formal_acceptance_verified=False)
    _save(output / 'status.json', state)
    _save(output / 'config.json', cfg)
    env_keys = ('STABLEWM_HOME', 'SPT_CACHE_DIR', 'PYTHONPATH')
    old_env = {key: os.environ.get(key) for key in env_keys}
    with (output / 'stdout_stderr.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                binding = validate_binding(profile_name, mode, data_root)
                state['binding'] = binding
                os.environ.update(build_environment(data_root, output))
                import torch
                import importlib.metadata
                state['versions'] = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda}
                for package in ('lightning', 'stable-pretraining', 'torchvision', 'hydra-core', 'omegaconf'):
                    state['versions'][package] = importlib.metadata.version(package)
                if not torch.cuda.is_available():
                    raise RuntimeError('the accepted BF16 GPU training recipe requires CUDA')
                probe = torch.ones(2, 2, device='cuda')
                if not torch.equal(probe @ probe, probe * 2):
                    raise RuntimeError('CUDA tensor preflight failed')
                del probe
                state.update(stage='lewm_native_running', updated_at=_stamp())
                _save(output / 'status.json', state)
                invoke_native(cfg, binding, data_root, single_original=single_original)
                state.update(stage='lewm_native_returned', actual_exit_code=0, completed_count=1)
            except BaseException:
                state.update(stage='lewm_failed', actual_exit_code=1, failed_count=1, error=traceback.format_exc())
                traceback.print_exc()
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    state['finished_at'] = state['updated_at'] = _stamp()
    _save(output / 'status.json', state)
    (output / 'actual_exit_code').write_text(str(state['actual_exit_code']) + '\n', encoding='utf8')
    return state['actual_exit_code']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--mode', required=True, choices=('2d', '3d'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prefetch', type=int)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--single-original-protocol', action='store_true',
                        help='required for Single original random-window split; invalid for other profiles')
    args = parser.parse_args(argv)
    return run_job(args.profile, args.mode, args.data_root, args.output,
                   args.workers, args.prefetch, args.config,
                   single_original=args.single_original_protocol)


if __name__ == '__main__':
    raise SystemExit(main())
