"""Sealed Cube RGB/action contracts shared by PLDM and native Tiny LeJEPA.

Profiles are names or exact copies of registered metadata. ``data_root`` retains
each original experiment's layout; ``validate_view`` takes the HDF5 file itself.
Official Play profiles describe a separate downstream source and cannot be
opened through this collected-data reader.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PROFILES = json.loads((_HERE / 'profiles.json').read_text(encoding='utf8'))


def profile_names():
    return tuple(_PROFILES)


def get_profile(name):
    """Return independent metadata for one explicitly named source contract."""
    if isinstance(name, dict):
        identity = name.get('name')
        if not isinstance(identity, str) or name != _PROFILES.get(identity):
            raise ValueError('profile dict must exactly match a registered profile')
        name = identity
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError('unknown Cube data profile: ' + repr(name))
    return deepcopy(_PROFILES[name])


def _collected_profile(name):
    profile = get_profile(name)
    if profile['source'] != 'self_collected':
        raise ValueError('official Play data requires its original official reader')
    return profile


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _manifest(view_directory, profile):
    path = Path(view_directory) / 'training_view_index.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != profile['manifest_sha256']:
        raise ValueError('sealed ' + profile['name'] + ' manifest checksum differs')
    value = json.loads(raw)
    if profile['name'] == 'triple_front600':
        contract = value['recipe']['contract']
        if (value.get('actual_exit_code') != 0 or value.get('stage') != 'passed'
                or (contract.get('physical_cube_count'), contract.get('horizon_actions'),
                    contract.get('train_episodes'), contract.get('validation_episodes'))
                != (3, 600, 10000, 1000)):
            raise ValueError('Triple600 production manifest contract differs')
    else:
        if (value.get('actual_exit_code') != 0
                or value.get('raw_data_unchanged') is not True
                or value.get('raw_rows_per_episode') != profile['raw_rows']
                or value.get('view_rows_per_episode') != profile['view_rows']
                or value.get('split_counts') != {'train': 10000, 'validation': 1000}
                or any(value.get(key) is not True for key in
                       ('no_GT_model_input', 'no_physical_visual_merge', 'no_feature_cache'))):
            raise ValueError('sealed RGB/action manifest collection contract differs')
        views = {Path(item['path']).name: item for item in value.get('views', [])
                 if item.get('keys') == ['pixels', 'action']}
        expected_names = {profile['train_view'], profile['validation_view']}
        if set(views) != expected_names:
            raise ValueError('exact sealed RGB/action manifest splits required')
        for split in ('train', 'validation'):
            item = views[profile[split + '_view']]
            episodes = profile[split + '_episodes']
            if (item.get('sha256') != profile[split + '_sha256']
                    or item.get('episodes') != episodes
                    or item.get('rows') != episodes * profile['view_rows']
                    or item.get('bytes') is None):
                raise ValueError('sealed view manifest provenance differs: ' + split)
        if profile['name'] == 'single200' and value.get('window') != {
                'clips_per_episode': 185, 'frameskip': 1, 'num_steps': 16,
                'span': 16, 'terminal_row_excluded': True}:
            raise ValueError('SingleCube terminal-free rows200 window differs')
        if profile['name'] == 'double500' and value.get('window') != {
                'clips_per_episode': 481, 'frameskip': 5, 'num_steps': 4,
                'span': 20, 'starts': [0, 480]}:
            raise ValueError('DoubleCube accepted source window differs')
        if profile['name'] == 'quadruple600' and value.get('raw_terminal_row') != 600:
            raise ValueError('QuadrupleCube terminal observation binding differs')

    # Preserve the original SingleCube backing-store mutation guard. These
    # sealed views retain their original VDS paths and are not relocation tools.
    backing = profile.get('backing_store')
    if backing:
        stat = Path(backing['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (backing['bytes'], backing['mtime_ns']):
            raise ValueError('SingleCube backing RGB HDF5 stat changed')
    for item in profile.get('collection_provenance', {}).values():
        if _sha(item['path']) != item['sha256']:
            raise ValueError('sealed collection provenance checksum differs: ' + item['path'])
    return value


def validate_manifest(data_root, profile):
    """Validate the original manifest and collection provenance before allocation."""
    selected = _collected_profile(profile)
    return _manifest(Path(data_root) / selected['view_directory'], selected)


def _view_identity(path, profile, split, expected_sha256=None):
    selected = _collected_profile(profile)
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    if '://' in str(path):
        raise ValueError('sealed training views require local paths')
    path = Path(path).resolve()
    if path.name != selected[split + '_view']:
        raise ValueError('sealed view name/identity differs from profile and split')
    expected = selected[split + '_sha256']
    if expected_sha256 is not None and expected_sha256 != expected:
        raise ValueError('requested sha256 differs from sealed profile')
    if _sha(path) != expected:
        raise ValueError('sealed view sha256 checksum differs')
    _manifest(path.parent, selected)
    return path, selected, expected


def _validate_open_file(file, selected, expected):
    """Apply the byte-preserved validator for the original producer's attrs."""
    name = selected['validator']
    spec = importlib.util.spec_from_file_location(
        'bench.ogbench._original_' + name[:-3], _HERE / 'contracts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if selected['name'] == 'triple_front600':
        return module.validate_view_attributes(file, allow_fixture=False)
    return module.validate_view_attributes(file, expected_sha256=expected)


def validate_view(path, profile, split, *, expected_sha256=None):
    """Check exact split/hash, sealed manifest and original HDF5 schema/attrs.

    Returns a small provenance record. This never modifies files or permits
    fixture manifests. A relocated path remains subject to the original VDS and
    Triple consumer path bindings.
    """
    path, selected, expected = _view_identity(path, profile, split, expected_sha256)
    import h5py
    with h5py.File(path, 'r', swmr=True, rdcc_nbytes=1 << 20) as handle:
        _validate_open_file(handle, selected, expected)
    return dict(profile=selected['name'], split=split, path=str(path), sha256=expected,
                manifest_sha256=selected['manifest_sha256'],
                episodes=selected[split + '_episodes'],
                rows=selected[split + '_episodes'] * selected['view_rows'])
