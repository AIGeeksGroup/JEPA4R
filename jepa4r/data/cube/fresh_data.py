"""Current paper Cube data binding over canonical SWM scalar HDF5 slicing."""
import argparse
import hashlib
import json
import os
import operator
from pathlib import Path
import sys
from datetime import datetime, timezone

HORIZONS = {1: 200, 2: 500, 3: 1000, 4: 600}
VIEW_KEYS = ('train_rgb', 'validation_rgb', 'train_pcd', 'validation_pcd')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def _datasets(handle):
    import h5py
    result = {}
    handle.visititems(lambda name, item: result.update({name: item}) if isinstance(item, h5py.Dataset) else None)
    return result


def _source_files(path, records, accepted):
    import h5py
    path = Path(path).resolve()
    if str(path) in records:
        return
    before = file_stat(path)
    previous = accepted.get(str(path), {})
    reuse = (previous.get('bytes'), previous.get('mtime_ns')) == (before['bytes'], before['mtime_ns'])
    digest = previous.get('sha256') if reuse else sha256(path)
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('source SHA256 missing or malformed: ' + str(path))
    records[str(path)] = dict(path=str(path), stat=before, sha256=digest,
                              hash_evidence='collection_QC_receipt' if reuse else 'fresh_full_file_hash')
    with h5py.File(path, 'r') as handle:
        columns = _datasets(handle)
        records[str(path)]['schema'] = {key: dict(shape=list(col.shape), dtype=str(col.dtype), virtual=col.is_virtual)
                                        for key, col in columns.items()}
        for column in columns.values():
            if column.is_virtual:
                for mapping in column.virtual_sources():
                    source = Path(mapping.file_name)
                    source = source if source.is_absolute() else path.parent / source
                    if source.resolve() == path:
                        raise ValueError('self-referencing source VDS is unsupported')
                    _source_files(source, records, accepted)
    if file_stat(path) != before:
        raise ValueError('source changed while binding: ' + str(path))


def _validate_raw_pair(rgb, pcd, cubes, horizon, count, progress=None):
    import h5py
    import numpy as np
    rows = count * (horizon + 1)
    with h5py.File(rgb, 'r') as a, h5py.File(pcd, 'r') as b:
        for handle in (a, b):
            if (not np.array_equal(handle['ep_len'][:], np.full(count, horizon + 1)) or
                    not np.array_equal(handle['ep_offset'][:], np.arange(count) * (horizon + 1))):
                raise ValueError('raw source episode index/count differs')
            expected = {'action': ((rows, 5), 'float32'),
                        'observation': ((rows, 19 + 9 * cubes), 'float32'),
                        'qpos': ((rows, 14 + 7 * cubes), 'float32'),
                        'qvel': ((rows, 14 + 6 * cubes), 'float32')}
            expected.update({'pixels': ((rows, 224, 224, 3), 'uint8')} if handle is a else
                            {'pcd_xyz': ((rows, 1200, 3), 'float32'),
                             'pcd_rgb': ((rows, 1200, 3), 'uint8')})
            for key, (shape, dtype) in expected.items():
                actual_dtype = str(handle[key].dtype) if key in handle else None
                allowed = ('float32', 'float64') if key in ('observation', 'qpos', 'qvel') else (dtype,)
                if key not in handle or handle[key].shape != shape or actual_dtype not in allowed:
                    raise ValueError('raw source schema differs: ' + key)
        common = set(_datasets(a)) & set(_datasets(b)) - {'ep_len', 'ep_offset'}
        for start in range(0, rows, horizon + 1):
            end = start + horizon + 1
            actions = a['action'][start:end]
            if not np.isfinite(actions[:-1]).all() or not np.isnan(actions[-1]).all():
                raise ValueError('finite executed actions and terminal NaN row required')
            for key in common:
                left, right = a[key][start:end], b[key][start:end]
                if not np.array_equal(left, right, equal_nan=left.dtype.kind == 'f'):
                    raise ValueError('RGB/PCD row alignment differs: ' + key)
                if key == 'reward' and left.dtype.kind == 'f':
                    # Reset has no preceding action/reward; only this row may be NaN.
                    if np.isinf(left).any() or not np.isfinite(left[1:]).all():
                        raise ValueError('nonfinite raw reward: only reset-row NaN is permitted')
                elif key != 'action' and left.dtype.kind == 'f' and not np.isfinite(left).all():
                    raise ValueError('nonfinite raw source: ' + key)
            for offset in range(start, end, 128):
                if not np.isfinite(b['pcd_xyz'][offset:min(end, offset + 128)]).all():
                    raise ValueError('nonfinite raw point cloud')
            episode = start // (horizon + 1) + 1
            if progress is not None and (episode % 100 == 0 or episode == count):
                progress(episode)


def _view(source, output, horizon, count, modality, collection_sha):
    import h5py
    import numpy as np
    keys = ['pixels', 'action'] if modality == 'rgb' else ['pcd_xyz', 'pcd_rgb', 'action']
    with h5py.File(source, 'r') as raw, h5py.File(output, 'x', libver='latest') as view:
        raw_rows = int(raw['ep_len'][0])
        for key in keys:
            column = raw[key]
            layout = h5py.VirtualLayout((count * horizon, *column.shape[1:]), column.dtype)
            src = h5py.VirtualSource(str(source), key, shape=column.shape)
            for episode in range(count):
                layout[episode * horizon:(episode + 1) * horizon] = src[episode * raw_rows:episode * raw_rows + horizon]
            view.create_virtual_dataset(key, layout)
        view['ep_len'] = np.full(count, horizon, np.int32)
        view['ep_offset'] = np.arange(count, dtype=np.int64) * horizon
        view.attrs['schema'] = 'cube_current_training_view_v1'
        view.attrs['collection_sha256'] = collection_sha
    os.chmod(output, 0o444)
    return dict(path=str(output), sha256=sha256(output), stat=file_stat(output),
                episodes=count, rows_per_episode=horizon, keys=keys)


def create_binding(collection_manifest, output_root):
    """Validate real paired sources and create a new, independent VDS manifest."""
    source = Path(collection_manifest).resolve()
    collected = json.loads(source.read_text(encoding='utf8'))
    if (
        collected.get('schema') != 'cube_fixed_collection_v1'
        or type(collected.get('actual_exit_code')) is not int
        or collected['actual_exit_code'] != 0
    ):
        raise ValueError('successful collection schema and actual exit0 required')
    protocol = collected.get('collection_protocol')
    if (collected.get('protocol') != 'paper_cube_fixed_mix_v1' or not isinstance(protocol, dict) or
            protocol.get('name') != 'paper_cube_fixed_mix_v1' or
            type(collected.get('collection_seed')) is not int or collected['collection_seed'] != 0):
        raise ValueError('current paper collection protocol and collection seed0 are required')
    cubes, mode = collected.get('cube_count'), collected.get('mode')
    if type(cubes) is not int or cubes not in HORIZONS or mode not in ('smoke', 'formal'):
        raise ValueError('explicit Cube1..4 and smoke/formal mode required')
    horizon = HORIZONS[cubes]
    if collected.get('raw_horizon') != horizon:
        raise ValueError('raw horizon differs from fixed collection protocol')
    counts = {'train': collected.get('train_episodes'), 'validation': collected.get('validation_episodes')}
    expected = {'train': 10000, 'validation': 1000} if mode == 'formal' else {'train': 2, 'validation': 1}
    if counts != expected or collected.get('split_counts', counts) != counts:
        raise ValueError('formal requires 10000/1000; smoke requires explicit 2/1 episodes')
    files = collected.get('producer_binding', {}).get('files', {})
    if not files or not collected.get('collection_protocol'):
        raise ValueError('actual producer files and collection protocol required')
    for name, digest in files.items():
        if sha256(name) != digest:
            raise ValueError('producer source checksum differs: ' + name)
    paths = {key: Path(collected['paths'][key]).resolve() for key in VIEW_KEYS}
    if len(set(paths.values())) != 4:
        raise ValueError('four distinct paired source files required')
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(stage='validating_sources', actual_exit_code=None, completed_count=0, failed_count=0,
                  started_at=datetime.now(timezone.utc).isoformat(), collection_manifest=str(source),
                  cube_count=cubes, mode=mode, split_counts=counts)
    save(output / 'status.json', status)
    try:
        records = {}
        for path in paths.values():
            _source_files(path, records, collected.get('data_files', {}))
            status.update(updated_at=datetime.now(timezone.utc).isoformat(), validated_source_files=len(records))
            save(output / 'status.json', status)
        for split, count in counts.items():
            def progress(completed):
                status.update(stage='validating_' + split, completed_count=completed,
                              updated_at=datetime.now(timezone.utc).isoformat())
                save(output / 'status.json', status)
            _validate_raw_pair(paths[split + '_rgb'], paths[split + '_pcd'], cubes, horizon, count, progress)
        status['stage'] = 'building_views'
        save(output / 'status.json', status)
        collection_sha = sha256(source)
        training_rows = horizon
        views = {key: _view(path, output / (key + '.h5'), training_rows,
                            counts['train' if key.startswith('train_') else 'validation'],
                            key.rsplit('_', 1)[1], collection_sha) for key, path in paths.items()}
        result = dict(
            schema='cube_current_training_v1',
            cube_count=cubes,
            mode=mode,
            raw_horizon=horizon,
            training_rows_per_episode=training_rows,
            train_episodes=counts['train'],
            validation_episodes=counts['validation'],
            collection_manifest=str(source),
            collection_manifest_sha256=collection_sha,
            producer_binding=collected['producer_binding'],
            collection_protocol=collected['collection_protocol'],
            source_files=records,
            raw={key: str(path) for key, path in paths.items()},
            views=views,
            validator_source=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
            training_seed=0,
            training_protocol='episode_heldout_v1',
            protocol=collected['protocol'],
            collection_seed=collected['collection_seed'],
            seed_schedule=collected.get('seed_schedule', {}),
            actual_exit_code=0,
        )
        save(output / 'data_manifest.json', result)
        status.update(stage='views_bound', actual_exit_code=0, completed_count=sum(counts.values()))
    except BaseException as error:
        status.update(stage='failed', actual_exit_code=1, failed_count=1, error=repr(error))
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        save(output / 'status.json', status)
        (output / 'actual_exit_code').write_text(str(status['actual_exit_code']) + '\n')
    return output / 'data_manifest.json'


def load_binding(path, *, verify=True):
    """Recheck bound VDS bytes and backing identities, without rewriting data."""
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf8'))
    if value.get('schema') != 'cube_current_training_v1' or value.get('actual_exit_code') != 0:
        raise ValueError('completed fresh training binding required')
    if (value.get('protocol') != 'paper_cube_fixed_mix_v1' or value.get('training_seed') != 0 or
            value.get('training_protocol') != 'episode_heldout_v1' or value.get('collection_seed') != 0):
        raise ValueError('current data protocol/seed binding differs')
    cubes, mode = value.get('cube_count'), value.get('mode')
    expected = (10000, 1000) if mode == 'formal' else (2, 1) if mode == 'smoke' else None
    if (type(cubes) is not int or cubes not in HORIZONS or expected is None or
            (value.get('train_episodes'), value.get('validation_episodes')) != expected or
            value.get('training_rows_per_episode') != HORIZONS[cubes] or
            set(value.get('views', {})) != set(VIEW_KEYS)):
        raise ValueError('fresh manifest budget/schema differs from its declared mode')
    if verify:
        validator = value['validator_source']
        if sha256(validator['path']) != validator['sha256']:
            raise ValueError('fresh binding validator source changed')
        if sha256(value['collection_manifest']) != value['collection_manifest_sha256']:
            raise ValueError('collection manifest checksum differs')
        for name, row in value['source_files'].items():
            if file_stat(name) != row['stat']:
                raise ValueError('fresh source stat changed: ' + name)
        for row in value['views'].values():
            if not Path(row['path']).resolve().is_relative_to(path.parent) or sha256(row['path']) != row['sha256']:
                raise ValueError('fresh view checksum/path differs')
        for name, digest in value['producer_binding']['files'].items():
            if sha256(name) != digest:
                raise ValueError('producer source checksum differs')
    return value


def profile(path):
    value = load_binding(path, verify=False)
    return dict(name='cube' + str(value['cube_count']), cube_count=value['cube_count'],
                source='fresh_self_collected', view_rows=value['training_rows_per_episode'],
                train_episodes=value['train_episodes'], validation_episodes=value['validation_episodes'],
                manifest_sha256=sha256(path), mode=value['mode'])


class ReadOnlyClipIndex:
    """Original compact storage, scalar tuples and slice-list semantics."""
    def __init__(self, pairs):
        import numpy as np
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._pairs.setflags(write=False)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [tuple(map(int, row)) for row in self._pairs[key]]
        episode, start = self._pairs[operator.index(key)]
        return int(episode), int(start)

    def __getstate__(self):
        return self._pairs

    def __setstate__(self, pairs):
        self.__init__(pairs)


def reader_class():
    """Use the canonical scalar slices/normalization-before-action-reshape path."""
    if 'FreshHDF5Dataset' in globals():
        return globals()['FreshHDF5Dataset']
    from jepa4r.models.cube.training_runtime import code_root
    code = code_root()
    sys.path.insert(0, str(code / 'third_party/stable-worldmodel'))
    import h5py
    from stable_worldmodel.data.formats import hdf5
    if Path(hdf5.__file__).resolve() != code / 'third_party/stable-worldmodel/stable_worldmodel/data/formats/hdf5.py':
        raise ValueError('another SWM reader was imported')

    class FreshHDF5Dataset(hdf5.HDF5Dataset):
        def __init__(self, manifest, key, **kwargs):
            self.manifest, self.key = str(Path(manifest).resolve()), key
            self.manifest_sha256 = sha256(self.manifest)
            binding = load_binding(self.manifest)
            path = binding['views'][key]['path']
            super().__init__(path=path, **kwargs)
            self.clip_indices = ReadOnlyClipIndex(self.clip_indices)

        def _open_h5(self):
            if sha256(self.manifest) != self.manifest_sha256:
                raise ValueError('fresh manifest changed after reader construction')
            load_binding(self.manifest)
            return h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=8 << 20)

        def _open(self):
            if self.h5_file is not None and getattr(self, '_owner_pid', None) != os.getpid():
                self.close()
            super()._open()
            self._owner_pid = os.getpid()

        def close(self):
            if getattr(self, 'h5_file', None) is not None:
                self.h5_file.close()
            self.h5_file = None
            self._owner_pid = None

        def __del__(self):
            self.close()

    FreshHDF5Dataset.__qualname__ = 'FreshHDF5Dataset'
    globals()['FreshHDF5Dataset'] = FreshHDF5Dataset
    return FreshHDF5Dataset


def make_reader(manifest, key, *, num_steps=16, frameskip=1, **kwargs):
    load_binding(manifest)
    return reader_class()(manifest, key, num_steps=num_steps, frameskip=frameskip, **kwargs)


def __getattr__(name):
    if name == 'FreshHDF5Dataset':
        return reader_class()
    raise AttributeError(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(create_binding(args.collection_manifest, args.output))
