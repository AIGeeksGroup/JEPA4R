"""Explicit Triple data contract; independent of model math and Torch imports."""
import json
import hashlib
import os
from pathlib import Path


EXPECTED = dict(env_type='triple', physical_cube_count=3, horizon_actions=1000,
                raw_rows_per_episode=1001, observation_dim=46, qpos_dim=35,
                qvel_dim=32, action_dim=5, train_episodes=10000,
                validation_episodes=1000)
PREFIX_SCHEMA = 'triple_front600_v1'
PREFIX_EXPECTED = dict(EXPECTED, horizon_actions=600, raw_rows_per_episode=601)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_sha(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _prefix_binding(file, *, allow_fixture=False):
    import numpy as np
    if file.attrs.get('prefix_view_schema') != PREFIX_SCHEMA:
        raise ValueError('Unknown explicit prefix view schema')
    source = validate_contract(json.loads(file.attrs.get('source_triple_contract', '{}')))
    contract = json.loads(file.attrs.get('triple_contract', '{}'))
    if contract != PREFIX_EXPECTED:
        raise ValueError('Prefix consumer requires explicit H600/601 contract')
    scope = file.attrs.get('training_scope')
    if scope != 'production' and not (allow_fixture and scope == 'fixture'):
        raise ValueError('Prefix fixture cannot pass production training')
    for key in ('env_type', 'physical_cube_count', 'horizon_actions'):
        if file.attrs.get(key) != contract[key]:
            raise ValueError('Prefix consumer attribute differs: ' + key)
    if file.attrs.get('consumer_manifest') != 'training_view_index.json':
        raise ValueError('Prefix requires its explicit consumer manifest')
    path = Path(file.filename).resolve()
    manifest_path = path.parent / 'training_view_index.json'
    manifest = json.loads(manifest_path.read_text())
    recipe = manifest['recipe']
    if (manifest.get('stage'), manifest.get('actual_exit_code'), recipe.get('schema'),
            recipe.get('scope'), recipe.get('source_contract'), recipe.get('contract')) != (
            'passed', 0, PREFIX_SCHEMA, scope, source, contract):
        raise ValueError('Prefix consumer manifest contract/scope differs')
    identity = recipe_sha(recipe)
    if manifest.get('consumer_recipe_sha256') != identity or file.attrs.get('consumer_recipe_sha256') != identity:
        raise ValueError('Prefix consumer manifest identity differs')
    parent = recipe['parent']
    for key in ('source_freeze_sha256', 'canonical_index_sha256', 'independent_seal_sha256'):
        if file.attrs.get('parent_' + key) != parent[key]:
            raise ValueError('Prefix parent provenance differs: ' + key)
    matches = [item for item in manifest['views'] if Path(item['path']).resolve() == path]
    if len(matches) != 1 or _sha(path) != matches[0]['sha256']:
        raise ValueError('Prefix consumer view hash/path differs')
    item = matches[0]
    count = item['episodes']
    expected_count = contract['train_episodes' if item['split'] == 'train' else 'validation_episodes']
    if ((scope == 'production' and count != expected_count) or
            (scope == 'fixture' and not 0 < count <= 2)):
        raise ValueError('Prefix episode quota differs')
    keys = {'pixels', 'action'} if item['modality'] == '2d' else {'pcd_xyz', 'pcd_rgb', 'action'}
    if (item['split'] not in ('train', 'validation') or item['modality'] not in ('2d', '3d') or
            set(file) != keys | {'ep_len', 'ep_offset'} or
            not np.array_equal(file['ep_len'][:], np.full(count, 600)) or
            not np.array_equal(file['ep_offset'][:], np.arange(count) * 600) or
            any(file[key].shape[0] != count * 600 or not file[key].is_virtual for key in keys)):
        raise ValueError('Prefix VDS rows/index/keys differ')
    return contract, dict(consumer_contract=contract, consumer_manifest_sha256=_sha(manifest_path),
        consumer_recipe_sha256=identity, view_sha256=matches[0]['sha256'],
        parent_source_freeze_sha256=parent['source_freeze_sha256'],
        parent_canonical_index_sha256=parent['canonical_index_sha256'],
        parent_independent_seal_sha256=parent['independent_seal_sha256'])


def validate_contract(contract):
    contract = dict(contract)
    for key, expected in EXPECTED.items():
        if contract.get(key) != expected:
            raise ValueError(f'Triple contract requires {key}={expected!r}, got {contract.get(key)!r}')
    return contract


def validate_view_attributes(file, *, allow_fixture=False):
    if file.attrs.get('prefix_view_schema') is not None:
        return _prefix_binding(file, allow_fixture=allow_fixture)[0]
    contract = validate_contract(json.loads(file.attrs.get('triple_contract', '{}')))
    scope = file.attrs.get('training_scope')
    if scope != 'production' and not (allow_fixture and scope == 'fixture'):
        raise ValueError('Triple training requires production-sealed views; fixture is smoke-only')
    for key in ('env_type', 'physical_cube_count', 'horizon_actions'):
        if file.attrs.get(key) != contract[key]:
            raise ValueError(f'Triple view attribute differs from contract: {key}')
    return contract


def validate_profile(cfg):
    """Run before allocating a fresh output; cannot accept old Double profiles."""
    validate_contract(cfg.triple_contract)
    root = Path(cfg.migration_root).resolve()
    if Path(os.environ.get('STABLEWM_HOME', '')).resolve() != root / 'outputs':
        raise ValueError('Set STABLEWM_HOME=$TRIPLE_ROOT/outputs; old output roots are forbidden')
    for key in ('dataset', 'validation_dataset'):
        data = cfg.data[key]
        if not Path(data.name).resolve().is_relative_to(root / 'data/views'):
            raise ValueError('Triple training views must be under $TRIPLE_ROOT/data/views')
        if data.get('format') != 'hdf5_train_bounded' or data.get('allow_fixture', False):
            raise ValueError('Formal Triple profile requires bounded production view reader')
        expected = {'pixels', 'action'} if cfg.model.obs_key == 'pixels' else {'pcd_xyz', 'pcd_rgb', 'action'}
        if set(data.keys_to_load) != expected or list(data.keys_to_cache) != ['action']:
            raise ValueError('Triple WM inputs must be original visual/action columns only')
        if data.num_steps != 4 or data.frameskip != 5:
            raise ValueError('Triple WM preserves four observations and frameskip five')
    if cfg.model.action_encoder.input_dim != 25 or cfg.history_size != 3 or cfg.num_preds != 1:
        raise ValueError('Original history/action chunk contract changed')
    if cfg.get('resume_checkpoint') is not None:
        import torch
        path = Path(cfg.resume_checkpoint)
        if not path.is_absolute() or not path.is_file() or path.suffix != '.ckpt':
            raise ValueError('Explicit existing full Triple .ckpt is required')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        validate_resume_binding(checkpoint, profile_binding(cfg))


def profile_binding(cfg):
    import h5py
    from omegaconf import OmegaConf
    views = {}
    for role in ('dataset', 'validation_dataset'):
        with h5py.File(cfg.data[role].name, 'r') as file:
            if file.attrs.get('prefix_view_schema') is not None:
                views[role] = _prefix_binding(file)[1]
            else:
                validate_view_attributes(file)
                views[role] = dict(canonical_index_sha256=file.attrs['canonical_index_sha256'],
                                   independent_seal_sha256=file.attrs['independent_seal_sha256'])
    return dict(contract=validate_contract(cfg.triple_contract), views=views,
                model=OmegaConf.to_container(cfg.model, resolve=True),
                optimizer=OmegaConf.to_container(cfg.optimizer, resolve=True))


def validate_resume_binding(checkpoint, expected):
    if checkpoint.get('triple_training_binding') != expected:
        raise ValueError('Full checkpoint lacks the same Triple model/data/optimizer binding')
    if 'state_dict' not in checkpoint or not checkpoint.get('optimizer_states'):
        raise ValueError('Triple resume requires full trainer and optimizer state, not model-only weights')
