"""CPU-only readers for parent-sampled IDs; no future-batch prefetch."""
import importlib.util
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_bridge = None
_readers = None


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev)


def cache_modality(manifest):
    modality = manifest.get('source_modality')
    if modality is None and 'source_rgb_index' in manifest and 'source_pcd_index' not in manifest:
        return 'rgb'  # Original sealed RGB manifest predates the explicit modality field.
    if modality == 'pixels':
        return 'rgb'  # Encoder kind pixels has the original bridge route/cache modality rgb.
    if modality not in ('rgb', 'pcd'):
        raise ValueError('Unknown or missing unambiguous cache modality')
    return modality


class CanonicalLatentLookup:
    """Read accepted valid-frame cache plus real terminal sidecar into readonly RAM.

    Addresses are the official sampler's raw source IDs, never BC tuple plans.
    No sampling, DataLoader, tail32 override, HDF read or encoder inference.
    """
    def __init__(self, cache_manifest, terminal_manifest, split):
        cache_manifest, terminal_manifest = Path(cache_manifest), Path(terminal_manifest)
        seal_path = cache_manifest.with_name('cache_seal.json')
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        manifest = json.loads(cache_manifest.read_text())
        modality = cache_modality(manifest)
        seal = json.loads(seal_path.read_text())
        terminal = json.loads(terminal_manifest.read_text())
        if split not in ('train', 'val') or any(x['status'] != 'passed' or x['exit_code'] != 0
                                               for x in (manifest, seal, terminal)):
            raise ValueError('Complete accepted independent cache split required')
        if (seal['manifest_sha256'] != sha(cache_manifest) or
            terminal['cache_manifest_sha256'] != sha(cache_manifest) or
            terminal['cache_seal_sha256'] != sha(seal_path) or
            terminal['source_modality'] != modality or
            terminal['source_checkpoint_sha256'] != manifest['source_checkpoint_sha256']):
            raise ValueError('Cache/terminal source binding changed')
        for key in ('preprocess_source_sha256', 'encoder_hashes', 'precision'):
            if terminal[key] != manifest[key]:
                raise ValueError(f'Terminal/cache {key} differs')
        if terminal['raw_id_mapping'] != 'episode*1001+1000':
            raise ValueError('Terminal raw source mapping differs')
        entries = manifest['splits'][split]
        if [entry['episode'] for entry in entries] != list(range(len(entries))):
            raise ValueError('Canonical episode order differs')
        self.latents = np.empty((len(entries)*1000, 192), dtype=np.float32)
        for episode, entry in enumerate(entries):
            path = Path(entry['path']).resolve()
            if (not path.is_relative_to(cache_manifest.parent.resolve()) or
                entry['shape'] != [1000, 192] or entry['dtype'] != 'float32' or
                entry['source_row_start'] != episode*1001 or
                entry['source_row_end_exclusive'] != episode*1001+1000):
                raise ValueError('Canonical cache shape/path/raw source mapping differs')
            before = file_stat(path)
            values = np.load(path, allow_pickle=False)
            if before != seal['cache_stats'][str(path)] or file_stat(path) != before:
                raise ValueError('Sealed latent stat changed during RAM load')
            if values.shape != (1000, 192) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError('Invalid canonical latent values')
            self.latents[episode*1000:(episode+1)*1000] = values
        record = terminal['splits'][split]
        path = Path(record['path']).resolve()
        if not path.is_relative_to(terminal_manifest.parent.resolve()) or sha(path) != record['sha256']:
            raise ValueError('Terminal sidecar path/SHA changed')
        before = file_stat(path)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'raw_ids', 'latents'}:
                raise ValueError('Terminal sidecar fields differ')
            ids, values = data['raw_ids'], data['latents']
        expected = np.arange(len(entries), dtype=np.int64)*1001+1000
        if (ids.dtype != np.int64 or not np.array_equal(ids, expected) or
            hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest() != record['raw_ids_sha256'] or
            record['raw_ids_shape'] != [len(entries)] or record['raw_ids_dtype'] != 'int64' or
            record['shape'] != [len(entries), 192] or record['dtype'] != 'float32' or
            values.shape != (len(entries), 192) or values.dtype != np.float32 or not np.isfinite(values).all()):
            raise ValueError('Invalid complete raw terminal sidecar')
        if file_stat(path) != before or sha(path) != record['sha256']:
            raise ValueError('Terminal sidecar changed during RAM load')
        self.terminals = np.ascontiguousarray(values)
        self.latents.flags.writeable = self.terminals.flags.writeable = False
        self.size = len(entries)*1001
        self.binding = dict(cache_manifest_sha256=sha(cache_manifest), cache_seal_sha256=sha(seal_path),
                            terminal_manifest_sha256=sha(terminal_manifest), split=split,
                            source_modality=modality,
                            WM_checkpoint_sha256=manifest['source_checkpoint_sha256'],
                            runtime='readonly_RAM_no_workers_no_runtime_encoder', BC_tail32_override=False)

    def lookup(self, role, raw_ids, *, paired_current_ids=None):
        if role not in ('observations', 'next_observations', 'value_goals', 'actor_goals'):
            raise ValueError('Unknown official role')
        ids = np.asarray(raw_ids)
        if ids.dtype != np.int64 or ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[1] != 1):
            raise ValueError('Raw IDs must be int64 (B,) or (B,1)')
        ids = ids.reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError('Raw ID outside source split')
        episode, step = np.divmod(ids, 1001)
        if role == 'next_observations':
            if paired_current_ids is None:
                raise ValueError('next_observations requires paired current IDs')
            current = np.asarray(paired_current_ids)
            if (current.dtype != np.int64 or current.shape != np.asarray(raw_ids).shape or
                np.any(ids != current.reshape(-1)+1) or np.any(episode != current.reshape(-1)//1001)):
                raise ValueError('Invalid same-episode current/next source pair')
        elif np.any(step == 1000):
            raise ValueError('Raw terminal allowed only as next observation')
        terminal = step == 1000
        output = np.empty((len(ids), 192), np.float32)
        output[~terminal] = self.latents[episode[~terminal]*1000+step[~terminal]]
        output[terminal] = self.terminals[episode[terminal]]
        return output


class CachedLatents:
    """Explicit input/reward ablation around unchanged official GCIQL mathematics."""
    def __init__(self, lookups, compacts, visual_mode, goal_mode='object_xyz'):
        if visual_mode not in ('cache', 'zero192'):
            raise ValueError('Explicit cache or same-capacity zero192 control required')
        if goal_mode not in ('object_xyz', 'latent_object_xyz') or (goal_mode == 'latent_object_xyz' and visual_mode != 'cache'):
            raise ValueError('Explicit cached visual goal or original XYZ goal required')
        self.lookups, self.compacts, self.visual_mode = lookups, compacts, visual_mode
        self.goal_mode = goal_mode
        self.reader = SimpleNamespace(split='train')
        for compact in compacts.values():
            states = compact['observations']
            if states.ndim != 2 or states.shape[1] != 28 or states.dtype != np.float32 or not np.isfinite(states).all():
                raise ValueError('Original finite float32 state28 required')
            states.flags.writeable = False
        self.source_binding = dict(caches={split: lookup.binding for split, lookup in lookups.items()},
            visual_mode=visual_mode, current_next='canonical192_plus_original28',
            goal='canonical192_plus_original_scaled_object_xyz_19_22' if goal_mode == 'latent_object_xyz' else 'original_scaled_object_xyz_19_22',
            reward='current_object_distance_to_value_goal_le_0.4; r=success-1; m=1-success',
            original_source_id_reward=False, normalization='no_added_normalization')
        self.binding_hash = hashlib.sha256(json.dumps(self.source_binding, sort_keys=True).encode()).hexdigest()

    def guard(self):
        if any(lookup.latents.flags.writeable or lookup.terminals.flags.writeable for lookup in self.lookups.values()):
            raise ValueError('Readonly cached feature violation')
        if any(compact['observations'].flags.writeable for compact in self.compacts.values()):
            raise ValueError('Readonly original state violation')

    def batch(self, raw):
        started = time.perf_counter()
        split = self.reader.split
        lookup, states = self.lookups[split], self.compacts[split]['observations']
        result = {key: raw[key] for key in ('actions', 'rewards', 'masks')}
        for role in ('observations', 'next_observations', 'value_goals', 'actor_goals'):
            ids = raw[role]
            if (ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[1] != 1 or
                np.any(ids < 0) or np.any(ids >= len(states)) or
                (role != 'next_observations' and np.any(ids % 1001 == 1000))):
                raise ValueError('Invalid raw source ID for official role')
            state = states[np.asarray(ids).reshape(-1)]
            if role in ('observations', 'next_observations'):
                kwargs = {'paired_current_ids': raw['observations']} if role == 'next_observations' else {}
                latent = lookup.lookup(role, ids, **kwargs)
                if self.visual_mode == 'zero192': latent = np.zeros_like(latent)
                result[role] = np.concatenate((latent, state), axis=-1)
            else:
                result[role] = np.ascontiguousarray(state[:, 19:22])
                if self.goal_mode == 'latent_object_xyz':
                    result[role] = np.concatenate((lookup.lookup(role, ids), result[role]), axis=-1)
        # The reward time remains current observation, not post-action next observation.
        success = (np.linalg.norm(result['observations'][:, 192+19:192+22] - result['value_goals'][:, -3:], axis=-1) <= 0.4)
        result['rewards'] = success.astype(np.float32)-1.0
        result['masks'] = 1.0-success.astype(np.float32)
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError('Nonfinite cached policy batch')
        self.last_batch_timing = dict(io_seconds=0.0, preprocessing_transfer_seconds=0.0,
                                     encoder_copy_seconds=0.0, cache_lookup_seconds=time.perf_counter()-started)
        return result


def load_bridge(config):
    global _bridge
    if _bridge is None:
        spec = importlib.util.spec_from_file_location('pcd_gciql_frozen_bridge', config['bridge_path'])
        _bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bridge)
        if _bridge.sha256(config['bridge_path']) != config['bridge_sha256']:
            raise ValueError('Frozen bridge changed')
    return _bridge


def initialize(config, source_config, metadata):
    global _readers
    bridge = load_bridge(config)
    _readers = {split: bridge.RawRows(source_config, 'pcd', compact, split)
                for split, compact in metadata.items()}


def read_part(job):
    split, ids, role = job
    return _readers[split].read(ids, role)


class ReadPool:
    def __init__(self, config, source_config, compacts, workers=None):
        self.workers = workers or config['workers']
        metadata = {split: {k: compact[k] for k in ('valids', 'actions')}
                    for split, compact in compacts.items()}
        self.pool = mp.get_context('spawn').Pool(self.workers, initialize,
                                               (config, source_config, metadata))
        self.worker_processes = tuple(self.pool._pool)
        self.processes = {p.pid: p for p in self.worker_processes}
        self.joined = False

    def record_processes(self):
        if not hasattr(self, 'processes'):
            self.processes = {p.pid: p for p in self.worker_processes}
        self.processes.update({p.pid: p for p in self.pool._pool})
        self.exit_codes = {pid: process.exitcode for pid, process in self.processes.items()}

    def check_workers(self):
        self.record_processes()
        if any(code is not None and code != 0 for code in self.exit_codes.values()):
            # Only this owned CPU reader pool is terminated. No IDs are resampled.
            self.pool.terminate(); self.pool.join(); self.joined = True
            self.record_processes()
            raise RuntimeError(f'Reader worker exit failure during current role: {self.exit_codes}')

    def read(self, split, ids, role):
        parts = [part for part in np.array_split(ids, min(len(ids), self.workers)) if len(part)]
        # map returns task order, including duplicate IDs. Only this current role is submitted.
        self.check_workers()
        pending = self.pool.map_async(read_part, [(split, part, role) for part in parts])
        while True:
            self.check_workers()
            try:
                loaded = pending.get(timeout=0.25)
                break
            except mp.TimeoutError:
                pass
        self.check_workers()
        return {key: np.concatenate([part[key] for part in loaded]) for key in loaded[0]}

    def close(self):
        self.record_processes()
        if not getattr(self, 'joined', False):
            self.pool.close(); self.pool.join(); self.joined = True
        self.record_processes()
        if any(code != 0 for code in self.exit_codes.values()):
            raise RuntimeError(f'Reader worker exit failure: {self.exit_codes}')
        return self.exit_codes


class ParallelRows:
    def __init__(self, pool, split='train'):
        self.pool, self.split = pool, split
        self.io_seconds = 0.0

    def read(self, ids, role):
        started = time.perf_counter()
        rows = self.pool.read(self.split, ids, role)
        self.io_seconds += time.perf_counter() - started
        return rows


def encode_batch(frozen, raw_batch, microbatch):
    if isinstance(frozen, CachedLatents):
        return frozen.batch(raw_batch)
    bridge = _bridge
    result = {key: raw_batch[key] for key in bridge.AGENT_FIELDS}
    io_before = frozen.reader.io_seconds
    preprocessing, encoder = 0.0, 0.0
    for key in bridge.OBS_FIELDS:
        started = time.perf_counter(); previous_io = frozen.reader.io_seconds
        inputs = frozen.inputs(raw_batch[key], key)
        preprocessing += time.perf_counter() - started - (frozen.reader.io_seconds - previous_io)
        started = time.perf_counter()
        # This is encoder inference chunking, not optimizer accumulation.
        result[key] = np.concatenate([frozen.encoder(part).logits.cpu().numpy()
                                      for part in inputs.split(microbatch, dim=0)]).astype(np.float32, copy=False)
        if result[key].shape != (len(raw_batch['actions']), 192) or not np.isfinite(result[key]).all():
            raise ValueError('Invalid frozen latent batch')
        encoder += time.perf_counter() - started
    frozen.last_batch_timing = dict(io_seconds=frozen.reader.io_seconds-io_before,
                                   preprocessing_transfer_seconds=preprocessing, encoder_copy_seconds=encoder)
    return result
