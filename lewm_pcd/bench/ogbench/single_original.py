"""Bind Single's historical HDF5 and recipe to the unchanged SWM loader.

The original loader retains its list of 182 windows per 201-row episode,
including the final NaN-action window. Splitting, normalization and forward
remain in the original training entry. No tensor library is needed to plan.
"""
import hashlib
import json
from pathlib import Path


_DIRECTORY = Path(__file__).with_suffix('')
_ASSET_SHA256 = {
    '2d.json': 'aea4f45bf7b940b26129e6d6287fdc4faf5ebb9041fb20d677081b7a5de768fa',
    '3d.json': '5f55570d568d9561ec4ac7788a3d62a65b2b30425c08d68c42d189079754bcd4',
    'raw_bindings.json': '2902894290f966bb977a10b34a2fb2d5cf14c4798407355929df251fdaadb66d',
}


def _read_asset(name):
    raw = (_DIRECTORY / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _ASSET_SHA256[name]:
        raise ValueError('original Single asset checksum differs: ' + name)
    return json.loads(raw)


_BINDINGS = _read_asset('raw_bindings.json')


def _mode(mode):
    if mode not in ('2d', '3d'):
        raise ValueError("Single mode must be '2d' or '3d'")


def load_single_recipe(mode, config_path=None):
    """Read the exact historical YAML semantics, with no fresh-seed hook."""
    _mode(mode)
    recipe = _read_asset(mode + '.json')
    if config_path is not None:
        path = Path(config_path)
        if path.suffix.lower() != '.json' or json.loads(path.read_text(encoding='utf8')) != recipe:
            raise ValueError('Single configuration must equal the frozen original recipe JSON')
    return recipe


def describe_single_original(mode, data_root):
    """Describe the one original training file without opening raw data."""
    _mode(mode)
    if '://' in str(data_root):
        raise ValueError('original Single data requires local paths')
    binding = _BINDINGS[mode]
    result = dict(profile_name='single200', profile='single200', mode=mode,
                  protocol='single_original_random_split_v1', split_scope='window',
                  seed=3072, train_split=0.9, train_fraction=0.9,
                  validation_fraction=1 - 0.9, manager_seed_argument='omitted',
                  global_seed_before_model=False, data_root=str(data_root),
                  dataset_path=str((Path(data_root) / binding['filename']).resolve()),
                  dataset_sha256=binding['sha256'],
                  source_config_sha256=binding['source_config_sha256'],
                  identity_verification=binding['hash_semantics'],
                  source_episodes=binding['episodes'], raw_rows=binding['rows'],
                  raw_rows_per_episode=201, num_steps=4, frameskip=5, span=20,
                  action_dim=5, action_chunk_dim=25, clips_per_episode=182,
                  starts=[0, 181], terminal_nan_action_window_preserved=True,
                  keys_to_load=list(binding['keys_to_load']),
                  keys_to_cache=list(binding['keys_to_cache']))
    if 'keys_to_merge' in binding:
        result['keys_to_merge'] = dict(binding['keys_to_merge'])
    return result


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _check_stat(path, expected):
    observed = path.stat()
    if observed.st_size != expected['bytes'] or observed.st_mtime_ns != expected['mtime_ns']:
        raise ValueError('original Single data stat changed: ' + str(path))


def _check_columns(handle, columns):
    for key, expected in columns.items():
        if key not in handle:
            raise ValueError('original Single column missing: ' + key)
        column = handle[key]
        if (list(column.shape) != expected['shape'] or str(column.dtype) != expected['dtype']
                or bool(column.is_virtual) != expected['virtual']):
            raise ValueError('original Single column schema differs: ' + key)


def _virtual_sources(handle, keys):
    """Record original VDS source paths and exact source/target selections."""
    result = {}
    for key in keys:
        rows = []
        for mapping in handle[key].virtual_sources():
            filename, dataset = mapping.file_name, mapping.dset_name
            if isinstance(filename, bytes):
                filename = filename.decode()
            if isinstance(dataset, bytes):
                dataset = dataset.decode()
            rows.append(dict(source_file=filename, source_dset=dataset,
                             source_bounds=[list(x) for x in mapping.src_space.get_select_bounds()],
                             virtual_bounds=[list(x) for x in mapping.vspace.get_select_bounds()],
                             source_points=mapping.src_space.get_select_npoints(),
                             virtual_points=mapping.vspace.get_select_npoints()))
        result[key] = rows
    return result


def validate_single_original_binding(mode, data_root):
    """Reuse full-file seals with live stat/schema/index and small VDS hashes.

    Large RGB and backing HDF5 files are not rehashed at startup. Their full
    hashes are historical seal evidence, with current size/mtime and schema
    checks. Terminal action NaNs are part of this original protocol.
    """
    description = describe_single_original(mode, data_root)
    binding = _BINDINGS[mode]
    path = Path(description['dataset_path'])
    _check_stat(path, binding['stat'])
    if mode == '3d' and _sha(path) != binding['sha256']:
        raise ValueError('original Single PCD VDS checksum differs')
    import h5py
    with h5py.File(path, 'r', swmr=True, rdcc_nbytes=1 << 20) as handle:
        _check_columns(handle, binding['columns'])
        for key, expected in binding['index'].items():
            value = handle[key][:]
            if (list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
                raise ValueError('original Single episode index differs: ' + key)
        if mode == '3d' and _virtual_sources(handle, binding['columns']) != binding['virtual_sources']:
            raise ValueError('original Single PCD VDS source mapping differs')
    if mode == '3d':
        backing = binding['backing']
        backing_path = Path(backing['path'])
        _check_stat(backing_path, backing['stat'])
        with h5py.File(backing_path, 'r', swmr=True, rdcc_nbytes=1 << 20) as handle:
            _check_columns(handle, backing['columns'])
        _check_stat(backing_path, backing['stat'])
    _check_stat(path, binding['stat'])
    return description


def make_single_original_reader(mode, data_root, *, loader, **dataset_kwargs):
    """Validate, then invoke the passed original ``swm.data.load_dataset``."""
    description = describe_single_original(mode, data_root)
    options = {key: description[key] for key in
               ('num_steps', 'frameskip', 'keys_to_load', 'keys_to_cache')}
    if 'keys_to_merge' in description:
        options['keys_to_merge'] = description['keys_to_merge']
    for key, value in dataset_kwargs.items():
        if key not in options and key not in ('transform', 'cache_dir'):
            raise ValueError('unsupported original Single reader option: ' + key)
        if key in options and value != options[key]:
            raise ValueError('original Single sampling/input contract override: ' + key)
        options[key] = value
    validate_single_original_binding(mode, data_root)
    return loader(description['dataset_path'], **options)
