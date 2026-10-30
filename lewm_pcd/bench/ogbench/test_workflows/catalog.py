"""Read-only model/task/stage index over byte-preserved native Test recipes."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

PACKAGE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
STAGES = ('encode', 'cache-audit', 'deliver', 'gate', 'reload', 'gciql',
          'accept', 'native-sr', 'native-audit')
ALIASES = {'lewm2d': 'lewm_rgb', 'lewm3d': 'lewm_pcd',
           'visreg_vitb768': 'visreg', 'vjepa21_vitl1024': 'vjepa',
           'lejepa_tiny512': 'lejepa'}


def load_catalog():
    return json.loads((PACKAGE/'catalog.json').read_text(encoding='utf8'))


def load_sources():
    return json.loads((PACKAGE/'source_map.json').read_text(encoding='utf8'))


def get_group(model, cube_count):
    model = ALIASES.get(model, model)
    if type(cube_count) is not int or cube_count not in (1, 2, 3, 4):
        raise ValueError('Cube count must be an integer in 1..4')
    for group in load_catalog()['groups']:
        if (group['model'], group['cube_count']) == (model, cube_count):
            return group
    raise ValueError('Unknown Test workflow model: '+str(model))


def _portable_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Source names must be portable POSIX relative paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value or value == '.':
        raise ValueError('Source names must stay inside their declared root')
    return path


def _repository_source(record):
    relative = _portable_path(record['repository_path'])
    path = REPO.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError('Repository source resolves outside the checkout')
    return path


def read_source(logical_name):
    _portable_path(logical_name)
    record = load_sources()['files'].get(logical_name)
    if record is None:
        raise ValueError('Native source was not included: '+logical_name)
    return _read_record(logical_name, record)


def _read_record(logical_name, record):
    data = _repository_source(record).read_bytes()
    if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Native source SHA/size mismatch: '+logical_name)
    return data


def verify_sources():
    sources = load_sources()['files']
    mismatches, verified = [], set()
    for logical, record in sources.items():
        _portable_path(logical)
        path = _repository_source(record)
        identity = (str(path), record['sha256'], record['size'])
        if identity in verified:
            continue
        data = path.read_bytes()
        if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            mismatches.append(logical)
        verified.add(identity)
    return dict(native_files=len(sources), unique_repository_files=len(verified),
                reused_v3_files=sum(record['reuses_v3'] for record in sources.values()),
                hash_mismatches=mismatches, executed=False)


def build_plan(model, cube_count, stage=None):
    if stage is not None and stage not in STAGES:
        raise ValueError('Unknown Test workflow stage: '+str(stage))
    group = get_group(model, cube_count)
    stages = copy.deepcopy(group['stages'])
    if stage is not None:
        stages = {stage: stages[stage]}
    # These are literal archived invocations, including their old output roots.
    # They are never rewritten or launched by the catalog.
    return dict(group=group['id'], matrix=group['matrix'], model=group['model'],
        cube_count=cube_count, native_group_id=group['native_group_id'],
        input_contract=group['input_contract'], config_sources=group['config_sources'],
        native_root=load_sources()['native_root'], stages=stages,
        external_asset_paths=group['external_asset_paths'], external_assets_verified=False,
        native_absolute_paths_preserved=True, recorded_outputs_are_historical=True,
        formal_result_revalidated=False, executed=False,
        usage='Inspect the pinned recipe and prepare a fresh reviewed output namespace before any native execution.')
