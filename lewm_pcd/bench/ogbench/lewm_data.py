"""LeWM modality/sampling seam over sealed sources and the shared SWM reader.

Plan descriptions are metadata only. Opening a reader first checks both split
identities and backing-source metadata, then delegates every scalar slice,
transform and action reshape to the unchanged stable-worldmodel implementation.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from . import contracts


_HERE = Path(__file__).resolve().parent
_BINDINGS = json.loads((_HERE / 'lewm_data_bindings.json').read_text(encoding='utf8'))['profiles']


def describe(profile_name, mode, data_root):
    """Describe an explicit LeWM source without opening data or tensor imports."""
    if profile_name == 'single200':
        from .single_original import describe_single_original
        return describe_single_original(mode, data_root)
    profile = contracts._collected_profile(profile_name)
    if mode not in ('2d', '3d'):
        raise ValueError("LeWM mode must be '2d' or '3d'")
    binding = _BINDINGS[profile['name']]
    if not binding['enabled']:
        raise ValueError(binding['reason'])
    if '://' in str(data_root):
        raise ValueError('sealed LeWM data requires local paths')
    selected = binding[mode]
    directory = Path(data_root) / profile['view_directory']
    result = dict(profile_name=profile['name'], profile=profile['name'], mode=mode,
                  data_root=str(data_root), producer=binding['producer_guard'],
                  binding_source=binding['producer_source'],
                  binding_source_sha256=binding['producer_source_sha256'],
                  split='explicit_train_validation', view_rows=profile['view_rows'],
                  train_episodes=profile['train_episodes'], validation_episodes=profile['validation_episodes'],
                  manifest_path=str(directory / 'training_view_index.json'),
                  manifest_sha256=profile['manifest_sha256'],
                  num_steps=4, frameskip=5, span=20, action_dim=5, action_chunk_dim=25,
                  clips_per_episode=profile['view_rows'] - 19,
                  keys_to_load=['pixels', 'action'] if mode == '2d' else ['pcd_xyz', 'pcd_rgb', 'action'],
                  keys_to_cache=['action'])
    for split in ('train', 'validation'):
        result[split + '_path'] = str(directory / selected[split + '_view'])
        result[split + '_sha256'] = (profile[split + '_sha256'] if mode == '2d'
                                    else selected.get(split + '_sha256'))
    return result


def _manifest_row(manifest, description, split):
    """Read a modality's identity from its SHA-locked original manifest."""
    path = Path(description[split + '_path'])
    rows = [row for row in manifest.get('views', []) if Path(row['path']).name == path.name]
    if len(rows) != 1:
        raise ValueError('exact LeWM modality/split manifest entry required')
    row = rows[0]
    expected_keys = description['keys_to_load']
    if 'keys' in row:
        if row['keys'] != expected_keys:
            raise ValueError('LeWM manifest modality keys differ')
    elif (row.get('modality'), row.get('split')) != (description['mode'], split):
        raise ValueError('LeWM manifest modality/split differs')
    episodes = description[split + '_episodes']
    if row.get('episodes') != episodes or row.get('rows') != episodes * description['view_rows']:
        raise ValueError('LeWM manifest split episode quota differs')
    value = row.get('sha256')
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('LeWM manifest requires its own view sha256')
    pinned = description[split + '_sha256']
    if pinned is not None and value != pinned:
        raise ValueError('LeWM manifest view sha256 differs from pinned modality')
    return row


def _check_sources(manifest):
    """Keep accepted file sizes/mtimes and small source-manifest SHA guards.

    Large raw HDF5 hashes are producer seal evidence; startup checks their
    accepted sizes/mtimes rather than rehashing the entire dataset.
    """
    files = dict(manifest.get('raw_data_files', {}))
    for part in manifest.get('recipe', {}).get('source_parts', []):
        files.update(part['files'])
    if not files:
        raise ValueError('LeWM manifest lacks accepted backing-source bindings')
    for name, expected in files.items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != expected['bytes'] or ('mtime_ns' in expected and stat.st_mtime_ns != expected['mtime_ns']):
            raise ValueError('sealed LeWM backing source stat changed: ' + name)
        if not name.endswith('.h5') and contracts._sha(path) != expected['sha256']:
            raise ValueError('sealed LeWM source metadata sha256 differs: ' + name)


def _validate_open_file(file, description, split, expected, manifest):
    """Retain the original producer guards; add the corresponding PCD schema."""
    import numpy as np
    profile = contracts.get_profile(description['profile_name'])
    if description['mode'] == '2d' or profile['name'] == 'triple_front600':
        contracts._validate_open_file(file, profile, expected)
    count = description[split + '_episodes']
    rows = count * description['view_rows']
    schema = ({'pixels': ((rows, 224, 224, 3), 'uint8')} if description['mode'] == '2d' else
              {'pcd_xyz': ((rows, 1200, 3), 'float32'), 'pcd_rgb': ((rows, 1200, 3), 'uint8')})
    schema['action'] = ((rows, 5), 'float32')
    if set(file) != set(schema) | {'ep_len', 'ep_offset'}:
        raise ValueError('LeWM visual/action keys differ from the sealed modality')
    for key, (shape, dtype) in schema.items():
        column = file[key]
        if column.shape != shape or column.dtype != np.dtype(dtype) or not column.is_virtual:
            raise ValueError('LeWM sealed column schema differs: ' + key)
        # An absent VDS source may silently read fill values in HDF5.
        for mapping in column.virtual_sources():
            source = Path(mapping.file_name)
            source = source if source.is_absolute() else Path(file.filename).parent / source
            if not source.is_file():
                raise ValueError('LeWM VDS backing source is missing: ' + str(source))
    if (not np.array_equal(file['ep_len'][:], np.full(count, description['view_rows']))
            or not np.array_equal(file['ep_offset'][:], np.arange(count) * description['view_rows'])):
        raise ValueError('LeWM sealed episode lengths/offsets differ')
    if profile['name'] != 'triple_front600':
        attrs = file.attrs
        horizon = description['view_rows']
        if (attrs.get('view_recipe') != description['producer']
                or attrs.get('canonical_index_sha256') != manifest['canonical_index']['sha256']
                or attrs.get('independent_seal_sha256') != manifest['independent_seal']['sha256']
                or attrs.get('raw_terminal_observation_row' + str(horizon)) != 'preserved_in_raw_archive'):
            raise ValueError('LeWM original producer provenance attributes differ')
        if profile['name'] == 'quadruple600' and (attrs.get('env_type'), attrs.get('physical_cube_count'),
                                                attrs.get('horizon_actions')) != ('quadruple', 4, 600):
            raise ValueError('LeWM Quadruple producer contract differs')


def _view_identity(path, profile_name, mode, split, data_root, expected_sha256=None):
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    description = describe(profile_name, mode, data_root)
    path = Path(path).resolve()
    if path != Path(description[split + '_path']).resolve():
        raise ValueError('LeWM view path differs from explicit profile, mode and split')
    manifest = contracts.validate_manifest(data_root, profile_name)
    row = _manifest_row(manifest, description, split)
    expected = row['sha256']
    if expected_sha256 is not None and expected_sha256 != expected:
        raise ValueError('requested LeWM sha256 differs from sealed modality')
    if contracts._sha(path) != expected:
        raise ValueError('LeWM view sha256 checksum differs')
    return path, description, expected, manifest


def validate_binding(profile_name, mode, data_root):
    """Verify both original split identities and guards before model allocation."""
    if profile_name == 'single200':
        from .single_original import validate_single_original_binding
        return validate_single_original_binding(mode, data_root)
    description = describe(profile_name, mode, data_root)
    manifest = contracts.validate_manifest(data_root, profile_name)
    _check_sources(manifest)
    import h5py
    for split in ('train', 'validation'):
        path, _, expected, _ = _view_identity(description[split + '_path'], profile_name, mode, split, data_root)
        with h5py.File(path, 'r', swmr=True, rdcc_nbytes=1 << 20) as handle:
            _validate_open_file(handle, description, split, expected, manifest)
        description[split + '_sha256'] = expected
    return deepcopy(description)


def make_reader(profile_name, mode, split, data_root, **dataset_kwargs):
    """Open LeWM through the sole shared scalar reader with immutable sampling."""
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    description = describe(profile_name, mode, data_root)
    for name in ('num_steps', 'frameskip', 'keys_to_load', 'keys_to_cache'):
        if name in dataset_kwargs and dataset_kwargs.pop(name) != description[name]:
            raise ValueError('LeWM sampling/input contract override: ' + name)
    if dataset_kwargs.pop('allow_fixture', False):
        raise ValueError('LeWM production reader does not accept fixture data')
    if dataset_kwargs.pop('format', 'hdf5_train_bounded') not in ('hdf5_train_bounded', 'bench_lewm_sealed'):
        raise ValueError('LeWM reader format differs from the sealed source')
    if dataset_kwargs.get('compact_clip_indices', True) is not True:
        raise ValueError('LeWM reader requires accepted compact clip indices')
    if int(dataset_kwargs.get('raw_chunk_cache_nbytes', 1 << 20)) <= 0:
        raise ValueError('raw_chunk_cache_nbytes must be positive')
    if set(dataset_kwargs) - {'transform', 'cache_dir', 'raw_chunk_cache_nbytes', 'compact_clip_indices'}:
        raise ValueError('unsupported LeWM reader options: ' + repr(sorted(dataset_kwargs)))
    description = validate_binding(profile_name, mode, data_root)
    from .reader import reader_class
    return reader_class()(description[split + '_path'], profile_name, split, 4,
                          description[split + '_sha256'], lewm_mode=mode,
                          data_root=data_root, **dataset_kwargs)
