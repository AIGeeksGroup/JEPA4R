"""PCD-only thin producer: canonical state poses, original frame/factory, reused atomic IO."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from common import episode_seed, legacy, load_arrays, load_source, object_sha, protocol_payload, validate_plan
from pcd_renderer import init_worker, render_episode, render_rows

HERE = Path(__file__).resolve().parent


def execute(plan, pilot):
    import fcntl
    import torch
    old = legacy(plan['workspace'])
    sha, stat, save = old.writer_helpers(plan['workspace'])
    validate_plan(plan, sha)
    out = Path(plan['cache_dir'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'cache_seal.json').exists(), 'Sealed cache stays immutable'
    gate_path = Path(plan['gate_dir']) / 'fresh_report.json'
    gate = json.loads(gate_path.read_text())
    assert gate['actual_exit_code'] == 0 and gate['fresh_reload_bitexact'] and gate['finite']
    assert gate['checkpoint_sha256'] == plan['checkpoint_sha256'] and gate['plan_sha256'] == sha(HERE / 'producer_plan.json')
    arrays_path, arrays = load_arrays(plan, old, sha, stat)
    arrays_binding = dict(path=str(arrays_path), sha256=sha(arrays_path))
    payload = protocol_payload(plan)
    protocol = object_sha(payload)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == payload
    else:
        save(protocol_path, payload)
    started, completed, rows = time.time(), 0, 0

    def status(stage, **fields):
        save(out / 'status.json', dict(stage=stage, started=started, updated=time.time(),
            completed_episodes=completed, completed_rows=rows, failed_episodes=0,
            protocol_sha256=protocol, actual_exit_code=None, **fields))

    status('strict_loading_original_frozen_model')
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-_RESOURCE_PROFILE['affinity_cores']:])
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    encoder = load_source('_cache_original_pcd_encoder', Path(plan['workspace']) / 'checkpoint_encoder.py')
    model = encoder.load_frozen(plan['checkpoint'], device='cuda')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled()
    from gate import tensor_digest
    frozen = tensor_digest(model.state_dict())
    assert frozen == gate['frozen_model_state_sha256']

    def infer(points, colors):
        assert points.shape == colors.shape and points.shape[1:] == (1200, 3)
        assert points.dtype == np.float32 and colors.dtype == np.uint8
        pieces, padding = [], 0
        for start in range(0, len(points), 32):
            xyz, rgb = points[start:start+32], colors[start:start+32]
            valid = len(xyz)
            if valid < 32:
                indices = np.arange(32-valid) % valid
                xyz, rgb = np.concatenate((xyz, xyz[indices])), np.concatenate((rgb, rgb[indices]))
                padding += 32-valid
            batch = dict(pcd_xyz=torch.from_numpy(xyz), pcd_rgb=torch.from_numpy(rgb))
            prepared = encoder.preprocess(batch, '3d')[model.obs_key]
            assert prepared.shape == (32, 1200, 6) and prepared.dtype == torch.float32
            pieces.append(encoder.encode(model, prepared.to('cuda')).cpu().numpy()[:valid].copy())
        result = np.concatenate(pieces)
        assert result.shape == (len(points), 192) and result.dtype == np.float32 and np.isfinite(result).all()
        return result, padding

    records, pending, boundary = {'train': [], 'val': []}, [], {}
    for split, count in plan['split_episodes'].items():
        for episode in range(1 if pilot else count):
            receipt_path = out / split / f'episode_{episode:06d}/receipt.json'
            if receipt_path.exists():
                record = json.loads(receipt_path.read_text())
                assert record['protocol_sha256'] == protocol and record['source_arrays_receipt'] == arrays_binding
                assert sha(record['latent']['path']) == record['latent']['sha256']
                records[split].append(dict(episode=episode, receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path)))
                completed += 1
                rows += 1001
            else:
                pending.append((split, episode))
    try:
        with ProcessPoolExecutor(max_workers=_RESOURCE_PROFILE['renderer_workers'], mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker, initargs=(plan['workspace'], str(arrays_path), plan['camera'], plan['pcd'])) as pool:
            iterator, queue = iter(pending), deque()
            for _ in range(min(_RESOURCE_PROFILE['prefetch_jobs'], len(pending))):
                job = next(iterator)
                queue.append((job, pool.submit(render_episode, job)))
            while queue:
                (split, episode), future = queue.popleft()
                rendered = future.result()
                try:
                    job = next(iterator)
                    queue.append((job, pool.submit(render_episode, job)))
                except StopIteration:
                    pass
                ids = old.canonical_ids(episode)
                assert np.array_equal(rendered['raw_ids'], ids)
                encoding_started = time.time()
                values, padding = infer(rendered['pcd_xyz'], rendered['pcd_rgb'])
                epdir = out / split / f'episode_{episode:06d}'
                epdir.mkdir(parents=True, exist_ok=True)
                target = epdir / 'latent.npy'
                assert not target.exists(), 'Uncommitted file requires explicit inspection; never overwrite'
                temporary = epdir / f'latent.partial_{os.getpid()}.npy'
                with temporary.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(target)
                readback = np.load(target, mmap_mode='r', allow_pickle=False)
                assert np.array_equal(readback, values) and np.isfinite(readback).all()
                record = {key: rendered[key] for key in ('sampling_seed', 'capture_boundaries',
                    'capture_receipt_stream_sha256', 'RGB_sha256', 'PCD_xyz_sha256', 'PCD_rgb_sha256',
                    'camera', 'render_geometry', 'worker_pid', 'render_started', 'render_completed')}
                record.update(status='episode_committed', actual_exit_code=0, split=split, episode=episode,
                    raw_row_start=int(ids[0]), raw_row_count=1001, main_rows=1000, terminal_raw_id=int(ids[-1]),
                    canonical_raw_ids_sha256=hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest(),
                    latent=dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target), shape=[1001,192], dtype='float32'),
                    protocol_sha256=protocol, checkpoint_sha256=plan['checkpoint_sha256'],
                    source_arrays_receipt=arrays_binding, source_npz=dict(path=arrays['splits'][split]['source_npz'],
                    accepted_stat=arrays['splits'][split]['accepted_stat']),
                    encode_seconds=time.time()-encoding_started, encoder_rows_valid=1001,
                    padding_rows=padding, encoder_rows_executed=1001+padding)
                record['runtime_resource_profile'] = _RESOURCE_PROFILE
                save(epdir / 'receipt.json', record)
                records[split].append(dict(episode=episode, receipt_path=str(epdir/'receipt.json'), receipt_sha256=sha(epdir/'receipt.json')))
                if pilot:
                    boundary[split] = dict(clouds=rendered['boundary_clouds'], latents=values[[999,1000]].copy())
                completed += 1
                rows += 1001
                status('encoding', split=split, episode=episode)
                del rendered, values, readback, future
            if pilot:
                for split in ('train', 'val'):
                    assert split in boundary, 'Pilot resume after partial diagnostic publication needs explicit inspection'
                    next_row = pool.submit(render_rows, (split, [1001])).result()
                    z, _ = infer(next_row['pcd_xyz'], next_row['pcd_rgb'])
                    xyz = np.stack([boundary[split]['clouds'][key]['pcd_xyz'] for key in ('999','1000')])
                    rgb = np.stack([boundary[split]['clouds'][key]['pcd_rgb'] for key in ('999','1000')])
                    path = out / f'pilot_{split}_boundary.npz'
                    with path.open('xb') as stream:
                        np.savez(stream, raw_ids=np.array([999,1000,1001],np.int64),
                            pcd_xyz=np.concatenate((xyz,next_row['pcd_xyz'])),
                            pcd_rgb=np.concatenate((rgb,next_row['pcd_rgb'])),
                            latents=np.concatenate((boundary[split]['latents'],z)))
        assert tensor_digest(model.state_dict()) == frozen
        for item in arrays['splits'].values():
            assert old.raw_stat(item['source_npz']) == item['accepted_stat']
        if pilot:
            save(out/'pilot_receipt.json', dict(actual_exit_code=0, status='pilot_pending_outer_acceptance',
                completed_episodes=completed, completed_rows=rows, source_encoder_frozen=True,
                protocol_sha256=protocol, source_arrays_receipt=arrays_binding, splits=records))
            status('pilot_pending_outer_acceptance')
            return
        assert completed == 3300 and rows == 3303300
        for split, count in plan['split_episodes'].items():
            records[split].sort(key=lambda item:item['episode'])
            assert [item['episode'] for item in records[split]] == list(range(count))
        save(out/'cache_manifest.json', dict(schema_version=1, actual_exit_code=0, protocol=payload,
            protocol_sha256=protocol, source_arrays_receipt=arrays_binding, splits=records,
            source_encoder_state_sha256=frozen, checkpoint_sha256=plan['checkpoint_sha256'],
            full_raw_RGB_or_PCD_persisted=False, training_updates=0, SR_episodes=0, world_model_dynamics_calls=0))
        save(out/'cache_seal.json', dict(actual_exit_code=0, status='producer_closed_pending_outer_and_independent_audit',
            ready_for_downstream=False, manifest_sha256=sha(out/'cache_manifest.json'), protocol_sha256=protocol,
            train_rows=3003000, val_rows=300300, full_raw_ID_coverage=True))
        status('producer_closed_pending_outer_and_independent_audit')
    except BaseException as exc:
        save(out/'status.json', dict(stage='failed', started=started, updated=time.time(),
            completed_episodes=completed, completed_rows=rows, failed_episodes=1, error=repr(exc), actual_exit_code=1))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.execute:
        assert all(Path(plan[key]).is_absolute() and '\\' not in plan[key]
                   for key in ('workspace','shared_state_root','checkpoint','cache_dir','gate_dir'))
        print(json.dumps(dict(status='prepared_only_no_model_render_GPU', training_updates=0, SR_episodes=0)))
        return
    execute(plan, args.pilot)


if __name__ == '__main__':
    main()
