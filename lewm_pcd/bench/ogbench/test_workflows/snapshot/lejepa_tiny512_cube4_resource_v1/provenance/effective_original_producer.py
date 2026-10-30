"""Resume canonical raw1001/episode frozen 2D embeddings; no learning or SR."""
import argparse
import ast
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from render_adapter import init_worker, render_episode, render_rows, missing_state_witness

HERE = Path(__file__).resolve().parent
CONTRACT_KEYS = ('checkpoint_sha256', 'data_acceptance_sha256', 'source_stat_freeze_sha256',
                 'source_pins', 'camera', 'renderer_workers', 'encoder_batch', 'torch_cpu_threads',
                 'split_episodes', 'raw_rows_per_episode')


def canonical_ids(episode):
    return np.arange(episode * 1001, (episode + 1) * 1001, dtype=np.int64)


def protocol_payload(plan, producer_hash, render_hash):
    return dict({key: plan[key] for key in CONTRACT_KEYS}, producer_source_sha256=producer_hash,
                render_adapter_source_sha256=render_hash, latent_dim=512, dtype='float32',
                pixel_preprocess_device=plan.get('pixel_preprocess_device', 'cpu'),
                preprocess_gate=plan.get('preprocess_gate', {}))


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol_digest(plan, producer_hash, render_hash):
    return object_sha(protocol_payload(plan, producer_hash, render_hash))


def writer_helpers(workspace):
    path = Path(workspace) / ('reference/downstream/source_closure/ogbench-official-single-play-20261006/'
        'official_lewm_gcbc_20261006/goal_xyz_mask_100k_b256_20261011_v1/rgb_latent_cache_20261012_v1/cache_producer.py')
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'f40dea29f18346e538edd15db2d035967245d6841c02a7afba781ed2b1cab745'
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in ('sha', 'stat', 'save')]
    namespace = dict(hashlib=hashlib, Path=Path, json=json, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sha'], namespace['stat'], namespace['save']


def raw_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def source_arrays(plan, protocol, sha, stat, save, contract):
    root = Path(plan['shared_state_root'])
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'source_arrays.json'
    frozen = json.loads(Path(plan['source_stat_freeze']).read_text())
    for item in frozen['datasets'].values():
        assert raw_stat(item['path']) == item['accepted_stat']
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        assert record['protocol_sha256'] == protocol and record['contract'] == contract
        for item in record['splits'].values():
            assert raw_stat(item['source_npz']) == item['accepted_stat']
            for key in ('qpos', 'qvel'):
                assert stat(item[key]['path']) == item[key]['stat']
        return receipt_path, record
    ownership = root / 'extraction_ownership.json'
    owner = dict(protocol_sha256=protocol, source_stat_freeze_sha256=plan['source_stat_freeze_sha256'])
    if ownership.exists():
        assert json.loads(ownership.read_text()) == owner
    else:
        assert not list(root.glob('*_qpos.npy')) and not list(root.glob('*_qvel.npy'))
        save(ownership, owner)
    record = dict(protocol_sha256=protocol, contract=contract,
                  source_stat_freeze_sha256=plan['source_stat_freeze_sha256'],
                  original_NPZ_rehashed=False, splits={})
    for split, count in plan['split_episodes'].items():
        frozen_item = frozen['datasets']['validation' if split == 'val' else 'train']
        path, rows = Path(frozen_item['path']), count * 1001
        item = dict(source_npz=str(path), accepted_stat=frozen_item['accepted_stat'], rows=rows)
        with np.load(path, allow_pickle=False) as raw:
            terminal = raw['terminals']
            assert terminal.shape == (rows,) and terminal.dtype == np.bool_
            assert np.array_equal(np.flatnonzero(terminal), np.arange(count, dtype=np.int64) * 1001 + 1000)
            del terminal
            for key, width in (('qpos', 35), ('qvel', 32)):
                target = root / f'{split}_{key}.npy'
                if target.exists():
                    # An earlier extraction stopped before its receipt; preserve it.
                    target.rename(target.with_name(target.name + '.uncommitted_' + str(time.time_ns())))
                value = raw[key]  # Decompress this field once, not once per worker.
                assert value.shape == (rows, width) and value.dtype == np.float32 and np.isfinite(value).all()
                temporary = target.with_name(target.name + '.partial_' + str(os.getpid()))
                with temporary.open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
                del value
                temporary.replace(target)
                item[key] = dict(path=str(target), shape=[rows, width], dtype='float32',
                                 stat=stat(target), sha256=sha(target))
        assert raw_stat(path) == item['accepted_stat']
        record['splits'][split] = item
    save(receipt_path, record)
    return receipt_path, record


def source_observation_rows(npz_path, ids):
    """Read only early pilot observation rows from the compressed NPY stream."""
    with zipfile.ZipFile(npz_path) as archive, archive.open('observations.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        assert not fortran and shape[1] == 55 and dtype == np.float32
        offset = stream.tell()
        result = []
        for raw_id in ids:
            stream.seek(offset + int(raw_id) * 55 * 4)
            result.append(np.frombuffer(stream.read(55 * 4), np.float32).copy())
        return np.stack(result)


def execute(plan, pilot):
    import fcntl
    import torch
    workspace, out = Path(plan['workspace']), Path(plan['cache_dir'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha, stat, save = writer_helpers(workspace)
    producer_hash, renderer_hash = sha(__file__), sha(HERE / 'render_adapter.py')
    base_payload = protocol_payload(plan, producer_hash, renderer_hash)
    base_digest = object_sha(base_payload)
    policy = plan.get('pixel_preprocess_device', 'cpu')
    assert policy == 'shared_cpu'
    selection_path = out / 'preprocess_selection.json'
    protocol, payload, selected_device = None, None, None
    assert plan['renderer_workers'] == 2 and plan['encoder_batch'] == 32 and plan['torch_cpu_threads'] == 1
    assert plan['split_episodes'] == {'train': 5000, 'val': 500} and plan['raw_rows_per_episode'] == 1001
    assert not (out / 'cache_seal.json').exists(), 'Sealed cache is immutable'
    assert sha(plan['data_acceptance']) == plan['data_acceptance_sha256']
    assert sha(plan['source_stat_freeze']) == plan['source_stat_freeze_sha256']
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    for name, expected in plan['source_pins'].items():
        assert sha(workspace / name) == expected, name
    def activate(device, selection_sha):
        nonlocal protocol, payload, selected_device
        assert device in ('cpu', 'cuda')
        selected_device = device
        payload = dict(base_payload, pixel_preprocess_device=device,
                       preprocessing_selection_sha256=selection_sha)
        protocol = object_sha(payload)
        protocol_path = out / 'protocol.json'
        if protocol_path.exists():
            assert json.loads(protocol_path.read_text()) == payload
        else:
            save(protocol_path, payload)

    if policy == 'shared_cpu':
        activate('cpu', None)
    elif selection_path.exists():
        selection = json.loads(selection_path.read_text())
        assert selection['base_contract_sha256'] == base_digest and selection['actual_exit_code'] == 0
        assert selection['selected_device'] in ('cpu', 'cuda')
        if selection['selected_device'] == 'cuda':
            assert selection['numeric_passed'] and selection['cuda_faster_by_required_gain']
        activate(selection['selected_device'], sha(selection_path))
    else:
        assert pilot, 'Run the declared one-batch selection gate in the pilot first'
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])
    started, completed, total_rows = time.time(), 0, 0

    def status(phase, **fields):
        save(out / 'status.json', dict(phase=phase, started=started, updated=time.time(),
            completed_episodes=completed, completed_rows=total_rows, failed_episodes=0,
            actual_exit_code=None, protocol_sha256=protocol, **fields))

    status('extracting_shared_readonly_state')
    arrays_contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
        source_stat_freeze_sha256=plan['source_stat_freeze_sha256'], split_episodes=plan['split_episodes'],
        raw_rows_per_episode=1001, qpos_width=42, qvel_width=38, extractor_source_sha256=plan['source_arrays_extractor_sha256'])
    arrays_protocol = object_sha(arrays_contract)
    arrays_path, arrays = source_arrays(plan, arrays_protocol, sha, stat, save, arrays_contract)
    arrays_binding = dict(path=str(arrays_path), sha256=sha(arrays_path))
    spec = importlib.util.spec_from_file_location('_cache_original_encoder', Path(plan['encoder_adapter']))
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    model = encoder.load_encoder(plan['encoder_spec'], device='cuda')
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert not torch.is_autocast_enabled()
    assert all(layer.track_running_stats and layer.running_mean is not None
               for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d))
    encoder_runtime = dict(torch=torch.__version__, CUDA=torch.version.cuda, device=torch.cuda.get_device_name(),
        CPU_threads=torch.get_num_threads(), dtype='float32', AMP=False,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, settings_modified_for_precision=False)

    def model_hash():
        value = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            array = tensor.detach().cpu().numpy()
            value.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        return value.hexdigest()

    frozen_hash = model_hash()
    assert sha(plan['encoder_spec']) == plan['encoder_spec_sha256']
    assert sha(plan['encoder_adapter']) == plan['encoder_adapter_sha256']

    alignment_checked = False
    def prepared_and_encoded(images, device):
        nonlocal alignment_checked
        # Renderer produces original BHWC uint8; the shared encoder API owns preprocessing.
        tensor = torch.from_numpy(images).permute(0,3,1,2).contiguous()
        assert tensor.dtype == torch.uint8 and tuple(tensor.shape) == (32,3,224,224)
        latent = model.encode_uint8(tensor)
        assert latent.dtype == torch.float32 and tuple(latent.shape) == (32,plan['latent_dim'])
        if not alignment_checked:
            alignment_path=out/'batch1_alignment.json'
            if alignment_path.exists():
                proof=json.loads(alignment_path.read_text())
                assert proof['passed'] is True and proof['actual_exit_code']==0
                assert proof['encoder_spec_sha256']==plan['encoder_spec_sha256']
                assert proof['frozen_encoder_state_sha256']==frozen_hash
            else:
                single=model.encode_uint8(tensor[:1])
                finite=bool(torch.isfinite(single).all() and torch.isfinite(latent).all())
                passed=finite and bool(torch.allclose(single,latent[:1],atol=1e-5,rtol=1e-5))
                save(alignment_path,dict(passed=passed,actual_exit_code=0 if passed else 1,
                    encoder_spec_sha256=plan['encoder_spec_sha256'],frozen_encoder_state_sha256=frozen_hash,
                    finite=finite,atol=1e-5,rtol=1e-5,max_abs_difference=float((single-latent[:1]).abs().max()),
                    first_real_uint8_RGB_sha256=hashlib.sha256(memoryview(images[0])).hexdigest(),
                    batch1_shape=list(single.shape),batch32_shape=list(latent.shape),extra_B1_forwards=1))
                assert passed,'Real B1/B32 alignment failed; preserve proof and stop'
            alignment_checked=True
        return tensor, latent

    def infer(images):
        assert selected_device == 'cpu'  # shared encoder API; no CPU/GPU speed selection
        pieces, padded = [], 0
        for start in range(0, len(images), 32):
            image = images[start:start + 32]
            valid = len(image)
            if valid < 32:
                image = np.concatenate((image, np.tile(image, (32 // valid + 1, 1, 1, 1))[:32 - valid]))
                padded += 32 - valid
            _, latent = prepared_and_encoded(image, selected_device)
            pieces.append(latent.float().cpu().numpy()[:valid].copy())
        result = np.concatenate(pieces)
        assert result.shape == (len(images), 512) and result.dtype == np.float32 and np.isfinite(result).all()
        return result, padded

    records, pending, boundary = {'train': [], 'val': []}, [], {}
    for split, count in plan['split_episodes'].items():
        for episode in range(1 if pilot else count):
            epdir = out / split / f'episode_{episode:06d}'
            receipt_path = epdir / 'receipt.json'
            if receipt_path.exists():
                record = json.loads(receipt_path.read_text())
                assert record['protocol_sha256'] == protocol and record['source_arrays_receipt'] == arrays_binding
                assert sha(record['latent']['path']) == record['latent']['sha256']
                records[split].append(dict(episode=episode, receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path)))
                completed += 1
                total_rows += 1001
            else:
                pending.append((split, episode))
    context = multiprocessing.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=init_worker,
                initargs=(str(workspace), str(arrays_path), plan['camera'])) as pool:
            iterator, queue = iter(pending), deque()
            for _ in range(min(2, len(pending))):
                job = next(iterator)
                queue.append((job, pool.submit(render_episode, job)))
            while queue:
                (split, episode), future = queue.popleft()
                rendered = future.result()
                ids = canonical_ids(episode)
                assert np.array_equal(rendered['raw_ids'], ids)
                encoding_started = time.time()
                values, padding = infer(rendered['pixels'])
                epdir = out / split / f'episode_{episode:06d}'
                epdir.mkdir(parents=True, exist_ok=True)
                target = epdir / 'latent.npy'
                assert not target.exists(), 'Uncommitted existing latent requires explicit inspection'
                temporary = epdir / f'latent.partial_{os.getpid()}.npy'
                with temporary.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                temporary.replace(target)
                readback = np.load(target, mmap_mode='r', allow_pickle=False)
                assert np.array_equal(readback, values) and np.isfinite(readback).all()
                record = dict(status='episode_committed', actual_exit_code=0, split=split, episode=episode,
                    raw_row_start=int(ids[0]), raw_row_count=1001, main_rows=1000, terminal_raw_id=int(ids[-1]),
                    canonical_raw_ids_sha256=hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest(),
                    latent=dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target), shape=[1001, 512], dtype='float32'),
                    producer_source_sha256=producer_hash, render_adapter_source_sha256=renderer_hash,
                    protocol_sha256=protocol, checkpoint_sha256=plan['checkpoint_sha256'],
                    pixel_preprocess_device=selected_device,
                    preprocessing_selection_sha256=payload['preprocessing_selection_sha256'],
                    preprocess_source_sha256=plan['encoder_adapter_sha256'],
                    checkpoint_encoder_sha256=plan['encoder_adapter_sha256'],
                    source_arrays_receipt=arrays_binding,
                    source_npz=dict(path=arrays['splits'][split]['source_npz'],
                                     accepted_stat=arrays['splits'][split]['accepted_stat']),
                    camera_contract_sha256=object_sha(plan['camera']), camera=rendered['camera'],
                    render_geometry=rendered['render_geometry'], RGB_sha256=rendered['RGB_sha256'],
                    worker_pid=rendered['worker_pid'], render_started=rendered['render_started'],
                    render_completed=rendered['render_completed'], encode_seconds=time.time() - encoding_started,
                    encoder_rows_valid=1001, padding_rows=padding, encoder_rows_executed=1001 + padding)
                # Publish the receipt only after actual feature readback succeeds.
                save(epdir / 'receipt.json', record)
                records[split].append(dict(episode=episode, receipt_path=str(epdir / 'receipt.json'),
                                           receipt_sha256=sha(epdir / 'receipt.json')))
                if pilot:
                    boundary[split] = dict(pixels=rendered['pixels'][[999, 1000]].copy(),
                                           latents=values[[999, 1000]].copy(),
                                           xyz=rendered['cube_xyz_witness'])
                completed += 1
                total_rows += 1001
                status('encoding', split=split, episode=episode)
                del rendered, values, readback, future
                try:
                    job = next(iterator)
                    queue.append((job, pool.submit(render_episode, job)))
                except StopIteration:
                    pass
            if pilot:
                for split in ('train', 'val'):
                    ids = np.array([999, 1000, 1001], np.int64)
                    if split not in boundary:
                        prior = pool.submit(render_rows, (split, ids[:2])).result()
                        cached = np.load(out / split / 'episode_000000/latent.npy', mmap_mode='r', allow_pickle=False)
                        boundary[split] = dict(pixels=prior['pixels'], latents=cached[[999, 1000]].copy(),
                                               xyz=prior['cube_xyz_witness'])
                    result = pool.submit(render_rows, (split, ids[2:])).result()
                    observations = source_observation_rows(arrays['splits'][split]['source_npz'], ids)
                    expected_xyz = np.stack([observations[:, 19 + 9*i:22 + 9*i] for i in range(4)], axis=1) / 10 + [0.425, 0., 0.]
                    xyz = dict(boundary[split]['xyz'], **result['cube_xyz_witness'])
                    actual_xyz = np.array([xyz[str(int(raw_id))] for raw_id in ids])
                    np.testing.assert_allclose(actual_xyz, expected_xyz, atol=1e-6, rtol=0)
                    next_latent, _ = infer(result['pixels'])
                    latent = np.concatenate((boundary[split]['latents'], next_latent))
                    pixels = np.concatenate((boundary[split]['pixels'], result['pixels']))
                    path = out / f'pilot_{split}_boundary.npz'
                    assert not path.exists(), 'Pilot diagnostic already exists; do not overwrite'
                    with path.open('xb') as stream:
                        np.savez(stream, raw_ids=ids, pixels=pixels, latents=latent)
                witness = pool.submit(missing_state_witness, ('train', 999)).result()
                save(out / 'pilot_missing_control_mocap_witness.json', witness)
        assert model_hash() == frozen_hash, 'Frozen model state changed'
        for item in arrays['splits'].values():
            assert raw_stat(item['source_npz']) == item['accepted_stat']
        if pilot:
            save(out / 'pilot_receipt.json', dict(status='pilot_complete_pending_outer_and_audit',
                actual_exit_code=0, protocol_sha256=protocol, completed_episodes=completed,
                completed_rows=total_rows, not_full_cache=True, source_encoder_frozen=True,
                source_arrays_receipt=arrays_binding, encoder_runtime=encoder_runtime, splits=records))
            status('pilot_complete_pending_outer_and_audit')
            return
        assert completed == 5500 and total_rows == 5505500
        for split, count in plan['split_episodes'].items():
            records[split].sort(key=lambda item: item['episode'])
            assert [item['episode'] for item in records[split]] == list(range(count))
        manifest = dict(schema_version=1, protocol=payload, protocol_sha256=protocol,
            base_protocol=base_payload, base_contract_sha256=base_digest,
            source_arrays_receipt=arrays_binding, source_arrays_protocol_sha256=arrays_protocol,
            preprocessing_selection=dict(path=str(selection_path), sha256=sha(selection_path)) if selection_path.exists() else None,
            checkpoint_sha256=plan['checkpoint_sha256'],
            source_encoder_state_sha256=frozen_hash, encoder_runtime=encoder_runtime,
            splits=records, training_updates=0, SR_episodes=0,
            world_model_dynamics_calls=0, raw_RGB_persisted=False, actual_exit_code=0)
        save(out / 'cache_manifest.json', manifest)
        save(out / 'cache_seal.json', dict(status='producer_closed_pending_outer_and_independent_audit',
            ready_for_downstream=False, actual_exit_code=0, protocol_sha256=protocol,
            manifest_sha256=sha(out / 'cache_manifest.json'), train_rows=5005000, val_rows=500500,
            full_raw_ID_coverage=True, source_encoder_frozen=True, per_episode_readback_exact=True,
            original_NPZ_rehashed=False, training_updates=0, SR_episodes=0))
        status('producer_closed_pending_outer_and_independent_audit')
    except BaseException as exc:
        save(out / 'status.json', dict(phase='failed', started=started, updated=time.time(),
            completed_episodes=completed, completed_rows=total_rows, failed_episodes=1,
            error=repr(exc), actual_exit_code=1, protocol_sha256=protocol))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.execute:
        assert Path(plan['workspace']).resolve() == Path(plan['root_workspace'])
        for key in ('workspace', 'cache_dir', 'shared_state_root', 'checkpoint',
                    'data_acceptance', 'source_stat_freeze'):
            assert Path(plan[key]).is_absolute() and '\\' not in plan[key], key
        for key in ('checkpoint', 'data_acceptance', 'source_stat_freeze'):
            assert Path(plan[key]).is_file(), key
        print(json.dumps(dict(status='prepared_only_no_renderer_model_GPU', pilot=args.pilot,
                              output=plan['cache_dir'], training_updates=0, SR_episodes=0)))
        return
    try:
        execute(plan, args.pilot)
    except BaseException as exc:
        # Also cover extraction/factory failures before the pool's local guard.
        import traceback
        out = Path(plan['cache_dir'])
        if out.is_dir() and not (out / 'cache_seal.json').exists():
            _, _, save = writer_helpers(plan['workspace'])
            path = out / 'status.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(phase='failed', updated=time.time(), error=repr(exc),
                          traceback=traceback.format_exc(), actual_exit_code=1)
            save(path, record)
        raise


if __name__ == '__main__':
    main()
