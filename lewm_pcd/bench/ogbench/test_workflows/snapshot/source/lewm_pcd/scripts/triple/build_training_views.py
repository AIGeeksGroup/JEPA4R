"""Small VDS views over independently sealed Triple H1000 physical shards; raw files stay intact."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triple_training_contract import validate_contract


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stat(path):
    value = Path(path).stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build(index_path, seal_path, output, *, fixture=False):
    index_path, seal_path, output = Path(index_path), Path(seal_path), Path(output)
    seal, index = read(seal_path), read(index_path)
    contract = validate_contract(index.get('contract', {}))
    require(seal.get('contract') == contract, 'Triple index and seal contract differ')
    scope = 'fixture' if fixture else 'production'
    require(seal.get('scope', 'production') == scope, 'fixture seal cannot pass production quota')
    horizon, rows = contract['horizon_actions'], contract['raw_rows_per_episode']
    cubes = contract['physical_cube_count']
    split_counts = seal['split_counts']
    expected_counts = {'train': contract['train_episodes'], 'validation': contract['validation_episodes']}
    require(set(split_counts) == set(expected_counts), 'Triple split names differ')
    if not fixture:
        require(split_counts == expected_counts, 'Triple production episode quota differs')
        require(seal['parts'] == 125 and len(index['parts']) == 125, 'Triple production requires original 125 parts')
    else:
        require(all(isinstance(v, int) and 0 < v <= 2 for v in split_counts.values()), 'fixture quota must be at most two episodes per split')
    require(seal['episodes'] == sum(split_counts.values()) and seal['raw_rows'] == sum(split_counts.values()) * rows, 'Triple sealed rows or episodes differ')
    require(seal['parts'] == len(index['parts']), 'Triple sealed part count differs')
    require(seal['actual_exit_code'] == 0, 'independent seal has not passed')
    require(seal['canonical_index'] == {'path': str(index_path), 'sha256': sha256(index_path)}, 'canonical index differs from seal')
    require(seal['source_freeze_sha256'] == index['source_freeze_sha256'], 'sealed source differs from index')
    require(index['no_physical_merge'], 'Triple raw physical shards must remain separate')
    expected_files = {path: item for part in index['parts'] for path, item in part['files'].items()}
    require(set(expected_files) == set(seal['accepted_files']), 'sealed file set differs from index')
    for path, item in expected_files.items():
        accepted = seal['accepted_files'][path]
        require(item['sha256'] == accepted['sha256'] and item['bytes'] == accepted['bytes'], f'sealed QC binding differs: {path}')
        require(stat(path) == {k: accepted[k] for k in ('bytes', 'mtime_ns')}, f'sealed file stat changed: {path}')
    require(not output.exists(), f'view output already exists; refusing overwrite: {output}')
    output.mkdir(parents=True)
    created, recipe_parts = [], []
    try:
        for split, episodes in seal['split_counts'].items():
            parts = [part for part in index['parts'] if part['spec']['split'] == split]
            require(sum(part['spec']['episodes'] for part in parts) == episodes, f'{split} episode quota changed')
            for modality, keys in (('2d', ('pixels', 'action')), ('3d', ('pcd_xyz', 'pcd_rgb', 'action'))):
                path = output / f'{modality}_{split}_action_rows{horizon}.h5'
                layouts, shapes, dtypes, episode_offset = {}, {}, {}, 0
                for part in parts:
                    count = part['spec']['episodes']
                    require(part['split_episode_offset'] == episode_offset and part['split_row_offset'] == episode_offset * rows, 'index offsets are not consecutive')
                    candidates = [p for p in part['files'] if p.endswith('_pix.h5' if modality == '2d' else '_pcd.h5')]
                    require(len(candidates) == 1, f'expected one {modality} source in {part["spec"]["name"]}')
                    source_path = candidates[0]
                    with h5py.File(source_path, 'r') as source:
                        for key, dim in (('observation', contract['observation_dim']), ('qpos', contract['qpos_dim']), ('qvel', contract['qvel_dim'])):
                            require(source[key].shape == (count * rows, dim), f'Triple state schema differs: {key}')
                        require(source['collection_goal_xyzs'].shape == (count * rows, cubes, 3), 'Triple full goal schema differs')
                        actions = source['action'][:].reshape(count, rows, 5)
                        require(np.isfinite(actions[:, :horizon]).all() and np.isnan(actions[:, horizon]).all(), 'Triple executed action or terminal sentinel differs')
                        require(np.array_equal(source['ep_len'][:], np.full(count, rows)), 'raw lengths changed')
                        require(np.array_equal(source['ep_offset'][:], np.arange(count) * rows), 'raw offsets changed')
                        for key in keys:
                            dataset = source[key]
                            require(dataset.shape[0] == count * rows, f'raw row quota changed: {source_path}/{key}')
                            shape, dtype = dataset.shape[1:], dataset.dtype
                            expected = {'pixels': ((224, 224, 3), np.dtype('uint8')), 'pcd_xyz': ((1200, 3), np.dtype('float32')), 'pcd_rgb': ((1200, 3), np.dtype('uint8'))}
                            require(key == 'action' and shape == (5,) and dtype.kind == 'f' or key in expected and (shape, dtype) == expected[key], f'unexpected input schema: {key}')
                            if key not in layouts:
                                shapes[key], dtypes[key] = shape, dtype
                                layouts[key] = h5py.VirtualLayout(shape=(episodes * horizon, *shape), dtype=dtype)
                            require((shapes[key], dtypes[key]) == (shape, dtype), f'schema differs between shards: {key}')
                            virtual = h5py.VirtualSource(source_path, key, shape=dataset.shape)
                            source_rows = h5py.MultiBlockSlice(start=0, stride=rows, count=count, block=horizon)
                            dst = episode_offset * horizon
                            layouts[key][dst:dst + count * horizon] = virtual[source_rows]
                    if modality == '2d':
                        recipe_parts.append({'spec': part['spec'], 'attempt': part['attempt'], 'view_episode_offset': episode_offset, 'view_row_offset': episode_offset * horizon, 'raw_rows_per_episode': rows, 'selected_source_rows': [0, horizon], 'producer_source_freeze_sha256': part['producer_source_freeze_sha256'], 'validator_source_sha256': part.get('validator_source_sha256'), 'source_files': part['files']})
                    episode_offset += count
                require(episode_offset == episodes, 'view episode count changed')
                with h5py.File(path, 'x', libver='latest') as view:
                    for key, layout in layouts.items():
                        view.create_virtual_dataset(key, layout)
                    view.create_dataset('ep_len', data=np.full(episodes, horizon, dtype=np.int32))
                    view.create_dataset('ep_offset', data=np.arange(episodes, dtype=np.int64) * horizon)
                    view.attrs['view_recipe'] = 'triple_H1000_action_bearing_rows1000_v1'
                    view.attrs['triple_contract'] = json.dumps(contract, sort_keys=True)
                    view.attrs['training_scope'] = scope
                    for key in ('env_type', 'physical_cube_count', 'horizon_actions'):
                        view.attrs[key] = contract[key]
                    view.attrs['canonical_index_sha256'] = sha256(index_path)
                    view.attrs['independent_seal_sha256'] = sha256(seal_path)
                    view.attrs['raw_terminal_observation_row1000'] = 'preserved_in_raw_archive'
                with h5py.File(path, 'r') as view:
                    require(set(view) == set(keys) | {'ep_len', 'ep_offset'}, 'unexpected view fields')
                    require(all(view[key].is_virtual for key in keys), 'visual/action inputs must be virtual')
                created.append({'path': str(path), **stat(path), 'sha256': sha256(path), 'episodes': episodes, 'rows': episodes * horizon, 'clips': episodes * (horizon - 20 + 1), 'keys': list(keys)})
        for path, accepted in seal['accepted_files'].items():
            require(stat(path) == {k: accepted[k] for k in ('bytes', 'mtime_ns')}, f'raw stat changed while building views: {path}')
        receipt = {'stage': 'VDS_metadata_built_pending_original_reader_and_loader_gate', 'actual_exit_code': 0, 'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'builder_sha256': sha256(__file__), 'canonical_index': seal['canonical_index'], 'independent_seal': {'path': str(seal_path), 'sha256': sha256(seal_path)}, 'raw_data_files': seal['accepted_files'], 'source_freeze_sha256': index['source_freeze_sha256'], 'split_counts': seal['split_counts'], 'view_rows_per_episode': horizon, 'raw_rows_per_episode': rows, 'window': {'frameskip': 5, 'num_steps': 4, 'span': 20, 'starts': [0, horizon - 20], 'clips_per_episode': horizon - 20 + 1}, 'views': created, 'parts': recipe_parts, 'raw_data_unchanged': True, 'no_physical_visual_merge': True, 'no_GT_model_input': True, 'no_feature_cache': True}
        receipt.update(contract=contract, scope=scope, production_accepted=False)
        (output / 'training_view_index.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        return receipt
    except Exception as error:
        (output / 'build_failure.json').write_text(json.dumps({'stage': 'failed', 'actual_exit_code': 1, 'error': f'{type(error).__name__}: {error}', 'builder_did_not_write_raw': True, 'raw_integrity_status': 'not accepted', 'partial_outputs_preserved': True}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical-index', required=True, type=Path)
    parser.add_argument('--independent-seal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture', action='store_true', help='isolated CPU/smoke fixtures only; never production quota')
    args = parser.parse_args()
    result = build(args.canonical_index, args.independent_seal, args.output, fixture=args.fixture)
    print(json.dumps({'actual_exit_code': result['actual_exit_code'], 'views': result['views']}, indent=2))
