"""Frozen LeWM GCBC extension: held-out validation and epoch-boundary resume.

The original GoalDataset, image/PCD preprocessing, GCRL and action Predictor
are reused. This extension uses constant AdamW 1e-4, not the upstream scheduler.
An interrupted epoch is replayed from the last completed epoch snapshot.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture_rng(include_cuda=True):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if include_cuda else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def validate_data_audit(report_path, train_path, val_path, modality):
    """Check sealed receipts/stat/VDS dependencies without rehashing 132 GB."""
    import h5py
    from gcbc_smoke import sha_file
    report_path = Path(report_path).resolve()
    seal = json.loads(report_path.read_text())
    if seal.get('status') != 'passed' or seal.get('exit_code') != 0:
        raise ValueError('Formal data audit must have passed with exit_code=0')
    for stage in ('contracts', 'hashes', 'geometry'):
        receipt = seal['receipts'][stage]
        document = Path(receipt['document_path'])
        status_file = document.with_name(stage + '_status.json')
        if receipt['exit_code'] != 0 or sha_file(document) != receipt['document_sha256'] or sha_file(status_file) != receipt['status_sha256']:
            raise ValueError(f'Sealed {stage} receipt hash/exit mismatch')
        stage_status = json.loads(status_file.read_text())
        if stage_status.get('status') != 'passed' or stage_status.get('exit_code') != 0:
            raise ValueError(f'{stage} stage has not passed')
    role_train, role_val = ('train_pix', 'val_pix') if modality == 'pixels' else ('train_pcd', 'val_pcd')
    if Path(seal['files'][role_train]['path']).resolve() != Path(train_path).resolve() or Path(seal['files'][role_val]['path']).resolve() != Path(val_path).resolve():
        raise ValueError('Requested train/validation files differ from sealed paths')
    for role, entry in seal['files'].items():
        path = Path(entry['path']).resolve()
        current = path.stat()
        if current.st_size != entry['bytes'] or current.st_mtime_ns != entry['mtime_ns']:
            raise ValueError(f'Sealed file/backing stat changed: {role}')
        if role in ('manifest', 'train_pcd', 'val_pix', 'val_pcd') and sha_file(path) != entry['sha256']:
            raise ValueError(f'Sealed VDS/manifest content changed: {role}')
    for role in (role_train, role_val):
        if not seal['dependencies'].get(role):
            raise ValueError(f'Sealed dependency chain missing for {role}')
        bindings = seal['virtual_source_bindings'].get(role, {})
        with h5py.File(seal['files'][role]['path'], 'r') as dataset:
            actual = {}
            for column in dataset:
                if not dataset[column].is_virtual:
                    continue
                maps = []
                for item in dataset[column].virtual_sources():
                    source_file = item.file_name.decode() if isinstance(item.file_name, bytes) else item.file_name
                    source_file = Path(source_file)
                    if not source_file.is_absolute():
                        source_file = Path(seal['files'][role]['path']).parent / source_file
                    source_dset = item.dset_name.decode() if isinstance(item.dset_name, bytes) else item.dset_name
                    maps.append({'source_file': str(source_file.resolve()), 'source_dset': source_dset,
                                 'source_bounds': [list(value) for value in item.src_space.get_select_bounds()],
                                 'virtual_bounds': [list(value) for value in item.vspace.get_select_bounds()],
                                 'source_points': int(item.src_space.get_select_npoints()),
                                 'virtual_points': int(item.vspace.get_select_npoints())})
                actual[column] = maps
            if actual != bindings:
                raise ValueError(f'Actual HDF5 virtual-source bindings differ from seal: {role}')
    if seal['paired_splits']['train']['episodes'] != 10000 or seal['paired_splits']['val']['episodes'] != 1000:
        raise ValueError('Audited episode split differs')
    geometry = seal['geometry']
    if geometry['status'] != 'passed' or not geometry['all_xyz_finite'] or geometry['all_rgb_dtype'] != 'uint8' or geometry['checked_xyz_frames'] != 2211000:
        raise ValueError('Full paired geometry/color audit has not passed')
    return {'path': str(report_path), 'sha256': sha_file(report_path), 'report': seal,
            'preflight': 'receipts + all file/backing stats + small metadata SHA + requested VDS mappings'}


def build_dataset(path, modality, preprocessor, scaler=None, seed=42):
    import stable_worldmodel as swm
    from stable_worldmodel.data.normalization import get_scaler
    from gcbc_smoke import DataTransform
    keys = ['pixels', 'action'] if modality == 'pixels' else ['pcd_xyz', 'pcd_rgb', 'action']
    raw = swm.data.load_dataset(str(Path(path).resolve()), num_steps=3, frameskip=1,
                               keys_to_load=keys, keys_to_cache=['action'], transform=None)
    if scaler is None:
        scaler = get_scaler('zscore').fit(raw.get_col_data('action'))
    raw.transform = DataTransform(modality, preprocessor, scaler)
    goal_key = 'goal_pixels' if modality == 'pixels' else 'goal_pcd'
    goals = swm.data.GoalDataset(raw, goal_probabilities=(0., 0., 1., 0.),
                                current_goal_offset=3, goal_keys={modality: goal_key}, seed=seed)
    return goals, scaler


def load_data_pair(train_path, val_path, modality, preprocessor, seed=42):
    import h5py
    import numpy as np
    train_path, val_path = Path(train_path).resolve(), Path(val_path).resolve()
    if train_path == val_path:
        raise ValueError('Training and validation must use distinct files')
    training, scaler = build_dataset(train_path, modality, preprocessor, seed=seed)
    validation, _ = build_dataset(val_path, modality, preprocessor, scaler=scaler, seed=seed)
    contract = {'training_path': str(train_path), 'validation_path': str(val_path),
                'training_episodes': len(training.dataset.lengths),
                'validation_episodes': len(validation.dataset.lengths),
                'training_clips': len(training), 'validation_clips': len(validation),
                'action_scaler_fit_source': 'training dataset only',
                'frameskip': 1, 'history': 3, 'action_dim': 5,
                'seed_disjoint_verified': False}
    with h5py.File(train_path, 'r') as first, h5py.File(val_path, 'r') as second:
        if 'collection_reset_seed' in first and 'collection_reset_seed' in second:
            train_seeds = np.unique(np.asarray(first['collection_reset_seed']))
            val_seeds = np.unique(np.asarray(second['collection_reset_seed']))
            if np.intersect1d(train_seeds, val_seeds).size:
                raise ValueError('Training and held-out validation contain overlapping reset seeds')
            contract.update(seed_disjoint_verified=True,
                            training_seed_count=len(train_seeds), validation_seed_count=len(val_seeds))
        for name, raw in (('training', first), ('validation', second)):
            contract[name + '_columns'] = {key: {'shape': list(raw[key].shape), 'dtype': str(raw[key].dtype)}
                                           for key in training.dataset.column_names}
    return training, validation, scaler, contract


def seed_worker(worker_id, seed_parts):
    import numpy as np
    import torch
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, worker_id]))


def make_loader(dataset, batch_size, workers, split, seed, epoch):
    import numpy as np
    import torch
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    parts = [int(seed), int(epoch) if split == 'train' else 0, 0 if split == 'train' else 1]
    dataset.rng = np.random.default_rng(np.random.SeedSequence(parts))
    generated_seed = int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
    generator = torch.Generator().manual_seed(generated_seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=workers,
        drop_last=split == 'train', generator=generator, pin_memory=True,
        persistent_workers=False,
        worker_init_fn=functools.partial(seed_worker, seed_parts=parts),
        **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if workers else {}))
    return loader, generator


def prepare_batch(batch, modality, goal_key, device):
    import torch
    from gcbc_smoke import verify_batch
    action = batch['action']
    if not torch.isfinite(action).all():
        raise ValueError('Nonfinite GCBC target actions; original uniform-future sampling excludes terminal sentinels')
    verify_batch(batch, modality, goal_key)
    batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    return batch, 0


def forward_loss(policy, batch, modality, goal_key, device, precision):
    import torch
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    with torch.autocast(device_type=device, dtype=dtype, enabled=precision != 'fp32'):
        latent = policy.encode({'pixels': batch[modality]}, target='embed')['embed']
        goal = policy.encode({'pixels': batch[goal_key]}, target='embed')['embed']
        if latent.shape != (len(batch['action']), 3, 1, 192) or goal.shape != (len(batch['action']), 1, 1, 192):
            raise ValueError('Actual LeWM latent/goal contract differs')
        means, _ = policy.predict_actions(latent, goal)
        loss = torch.nn.functional.mse_loss(means.float(), batch['action'].float())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite train/validation loss')
    return loss


def train_step(policy, batch, optimizer, modality, goal_key, device='cuda', precision='fp32', grad_scaler=None):
    import torch
    batch, sentinels = prepare_batch(batch, modality, goal_key, device)
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(policy, batch, modality, goal_key, device, precision)
    scale_before = float(grad_scaler.get_scale()) if grad_scaler is not None else 1.
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
    else:
        loss.backward()
    gradients = [value.grad for value in policy.action_predictor.parameters() if value.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value.detach().float()) for value in gradients]))
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError('Action head gradients are nonfinite or all zero')
    if any(parameter.grad is not None or parameter.requires_grad for parameter in policy.encoder.parameters()):
        raise ValueError('Frozen encoder/projector received gradients')
    lr = float(optimizer.param_groups[0]['lr'])
    if lr != 1e-4:
        raise ValueError('Actual constant learning rate differs from 1e-4')
    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    scale_after = float(grad_scaler.get_scale()) if grad_scaler is not None else 1.
    if scale_after < scale_before:
        raise ValueError(f'AMP scale decreased/optimizer step skipped: {scale_before} -> {scale_after}')
    return {'mse': float(loss.detach()), 'batch_samples': len(batch['action']),
            'gradient_l2': float(norm), 'actual_lr': lr, 'reset_action_sentinels': sentinels,
            'amp_scale_before': scale_before, 'amp_scale_after': scale_after, 'optimizer_step_skipped': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--smoke-utils-root', required=True)
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--train-dataset', required=True)
    parser.add_argument('--validation-dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Total target epochs, including completed epochs')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark-batches', type=int, help='Diagnostic only: limits train/val; never saves a formal epoch')
    parser.add_argument('--data-audit-report', required=True, help='Passed sealed dataset audit JSON; full backing hashes are not reread')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('Positive epochs/batch size and nonnegative workers required')
    if args.device == 'cpu' and args.precision == 'fp16':
        parser.error('FP16 CUDA GradScaler requires CUDA')
    if args.benchmark_batches is not None and args.benchmark_batches < 1:
        parser.error('benchmark-batches must be positive')
    output = Path(args.output).resolve()
    existed = output.exists()
    output.mkdir(parents=True, exist_ok=args.resume)
    protocol_accepted = False
    status = {'status': 'running', 'stage': 'setup', 'started_at': now(), 'updated_at': now(),
              'completed_epochs': 0, 'global_updates': 0, 'exit_code': None,
              'resume_semantics': 'exact completed-epoch snapshot; replay an unfinished epoch'}
    def persist():
        status['updated_at'] = now()
        atomic_json(output / 'status.json', status)
    if not existed:
        persist()
    with (output / 'run.log').open('a', buffering=1) as logfile, contextlib.redirect_stdout(logfile), contextlib.redirect_stderr(logfile):
        handler = logging.StreamHandler(logfile)
        logging.getLogger().addHandler(handler)
        try:
            import numpy as np
            import torch
            import hydra
            from omegaconf import OmegaConf
            lewm = Path(args.repo_root).resolve() / 'lewm_pcd'
            sys.path.insert(0, str(lewm))
            sys.path.insert(0, str(lewm / 'third_party/stable-worldmodel'))
            sys.path.insert(0, str(Path(args.smoke_utils_root).resolve()))
            import stable_worldmodel as swm
            from lewm_gcbc_adapter import create_policy
            from gcbc_smoke import sha_file, state_hash
            source = Path(args.source_checkpoint).resolve()
            cfg = OmegaConf.load(source.parent / 'config.yaml')
            modality = str(cfg.model.obs_key)
            if modality not in ('pixels', 'pcd') or cfg.history_size != 3 or cfg.data.dataset.frameskip != 5:
                raise ValueError('Source world-model contract differs')
            audited_data = validate_data_audit(args.data_audit_report, args.train_dataset,
                                              args.validation_dataset, modality)
            goals, val_goals, scaler, data_contract = load_data_pair(
                args.train_dataset, args.validation_dataset, modality,
                hydra.utils.instantiate(cfg.data.preprocessor), seed=args.seed)
            if data_contract['training_episodes'] != 10000 or data_contract['validation_episodes'] != 1000 or not data_contract['seed_disjoint_verified']:
                raise ValueError('Formal data requires 10000 training and independent 1000 validation episodes with disjoint reset seeds')
            goal_key = 'goal_pixels' if modality == 'pixels' else 'goal_pcd'
            factory = {'_target_': 'lewm_gcbc_adapter.create_policy', 'source_checkpoint': str(source),
                       'history_size': 3, 'action_dim': 5,
                       'predictor_config': {'depth': 6, 'heads': 16, 'mlp_dim': 2048, 'dim_head': 64,
                                            'dropout': 0., 'emb_dropout': 0.}}
            protocol = {'source_checkpoint': str(source), 'source_sha256': sha_file(source),
                        'source_config_sha256': sha_file(source.parent / 'config.yaml'),
                        'data': data_contract, 'batch_size': args.batch_size, 'workers': args.workers,
                        'precision': args.precision, 'device': args.device, 'seed': args.seed,
                        'action_scaler': {'mean': scaler.mean.tolist(), 'std': scaler.std.tolist(), 'eps': scaler.eps},
                        'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.01, 'scheduler': 'constant'},
                        'scope': 'frozen LeWM→GCBC extension, not official DINO baseline',
                        'goal_sampling': {'algorithm': 'original uniform-future GoalDataset',
                                          'worker_rng': 'epoch-scoped seed(seed,epoch,split,worker); new workers every epoch',
                                          'multiprocessing_context': 'spawn',
                                          'validation': 'fixed seed and fixed ordering across all epochs'},
                        'resume': 'completed epoch only; no exact mid-epoch resume',
                        'factory': factory, 'adapter_sha256': sha_file(Path(args.smoke_utils_root) / 'lewm_gcbc_adapter.py'),
                        'runner_sha256': sha_file(__file__), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'benchmark_only': args.benchmark_batches is not None,
                        'training_drop_last': True, 'dropped_training_clips_per_epoch': len(goals) % args.batch_size,
                        'validation_drop_last': False,
                        'imported_swm_path': str(Path(swm.__file__).resolve())}
            protocol['external_data_audit'] = audited_data
            protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            protocol['sha256'] = protocol_hash
            if (output / 'protocol.json').exists():
                if json.loads((output / 'protocol.json').read_text()) != protocol:
                    raise ValueError('Immutable protocol mismatch; existing artifacts preserved')
            else:
                atomic_json(output / 'protocol.json', protocol)
            protocol_accepted = True
            source_model = swm.wm.utils.load_pretrained(str(source), cache_dir='/')
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            policy = create_policy(world_model=source_model, history_size=3, action_dim=5,
                                   predictor_config=factory['predictor_config']).to(args.device)
            del source_model
            optimizer = torch.optim.AdamW(policy.action_predictor.parameters(), lr=1e-4)
            amp = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16') if args.device == 'cuda' else None
            frozen_hash = state_hash(policy.encoder)
            manifest = {'protocol_sha256': protocol_hash, 'epochs': [], 'last': None, 'best': None}
            start_epoch, global_updates = 0, 0
            if (output / 'checkpoint_manifest.json').exists():
                manifest = json.loads((output / 'checkpoint_manifest.json').read_text())
            if manifest['last'] is not None:
                last_path = output / manifest['last']['directory'] / 'resume.pt'
                if sha_file(last_path) != manifest['last']['resume_sha256']:
                    raise ValueError('Committed resume checkpoint hash differs')
                snapshot = torch.load(last_path, map_location='cpu', weights_only=False)
                if snapshot['protocol_sha256'] != protocol_hash or snapshot['frozen_state_sha256'] != frozen_hash:
                    raise ValueError('Resume source/trunk differs')
                policy.action_predictor.load_state_dict(snapshot['head'], strict=True)
                optimizer.load_state_dict(snapshot['optimizer'])
                if amp is not None:
                    amp.load_state_dict(snapshot['grad_scaler'])
                restore_rng(snapshot['rng'])
                start_epoch, global_updates = snapshot['next_epoch'], snapshot['global_updates']
                status['completed_epochs'] = start_epoch
                status['global_updates'] = global_updates
            else:
                atomic_json(output / 'checkpoint_manifest.json', manifest)
            atomic_json(output / 'action_scaler.json', protocol['action_scaler'])
            status.update(stage='training', target_epochs=args.epochs, frozen_state_sha256=frozen_hash)
            persist()
            for epoch in range(start_epoch, args.epochs):
                free_bytes = shutil.disk_usage(output).free
                checkpoint_estimate = int(1.2 * (
                    sum(value.numel() * value.element_size() for value in policy.state_dict().values())
                    + 3 * sum(value.numel() * value.element_size() for value in policy.action_predictor.parameters())))
                needed_bytes = checkpoint_estimate * (args.epochs - epoch) + 5 * 1024**3
                status['disk_gate'] = {'free_bytes': free_bytes, 'estimated_epoch_checkpoint_bytes': checkpoint_estimate,
                                       'required_remaining_bytes_with_5gib_reserve': needed_bytes}
                if args.benchmark_batches is None and free_bytes < needed_bytes:
                    raise ValueError('Insufficient disk for retained epoch checkpoints and 5 GiB reserve; existing checkpoints preserved')
                begin = time.perf_counter()
                train_loader, generator = make_loader(goals, args.batch_size, args.workers, 'train', args.seed, epoch)
                validation_loader, val_generator = make_loader(val_goals, args.batch_size, args.workers, 'validation', args.seed, epoch)
                policy.train()
                train_total, train_samples, reset_sentinels, last_grad = 0., 0, 0, None
                for index, batch in enumerate(train_loader):
                    if args.benchmark_batches is not None and index >= args.benchmark_batches:
                        break
                    metrics = train_step(policy, batch, optimizer, modality, goal_key, args.device, args.precision, amp)
                    train_total += metrics['mse'] * metrics['batch_samples']
                    train_samples += metrics['batch_samples']
                    reset_sentinels += metrics['reset_action_sentinels']
                    last_grad = metrics['gradient_l2']
                    global_updates += 1
                    if index % 100 == 0:
                        status.update(epoch=epoch + 1, train_batch=index + 1, train_samples=train_samples,
                                      global_updates=global_updates, last_loss=metrics['mse'], actual_lr=metrics['actual_lr'])
                        persist()
                        print(json.dumps({'epoch': epoch + 1, 'batch': index + 1, **metrics}), flush=True)
                train_seconds = time.perf_counter() - begin
                policy.eval()
                val_total, val_samples = 0., 0
                with torch.inference_mode():
                    for index, batch in enumerate(validation_loader):
                        if args.benchmark_batches is not None and index >= args.benchmark_batches:
                            break
                        batch, _ = prepare_batch(batch, modality, goal_key, args.device)
                        loss = forward_loss(policy, batch, modality, goal_key, args.device, args.precision)
                        val_total += float(loss) * len(batch['action'])
                        val_samples += len(batch['action'])
                if not train_samples or not val_samples or state_hash(policy.encoder) != frozen_hash:
                    raise ValueError('Empty epoch or frozen parameter/buffer mutation')
                metrics = {'epoch': epoch + 1, 'global_updates': global_updates, 'train_mse': train_total / train_samples,
                           'validation_mse': val_total / val_samples, 'train_samples': train_samples,
                           'validation_samples': val_samples, 'train_seconds': train_seconds,
                           'validation_seconds': time.perf_counter() - begin - train_seconds,
                           'reset_action_sentinels': reset_sentinels, 'last_gradient_l2': last_grad,
                           'actual_lr': float(optimizer.param_groups[0]['lr']), 'frozen_state_sha256': frozen_hash}
                if args.benchmark_batches is not None:
                    atomic_json(output / 'benchmark.json', metrics)
                    status.update(status='benchmark_completed', stage='complete', exit_code=0, **metrics)
                    persist()
                    (output / 'exit_code').write_text('0\n')
                    return 0
                expected_train = (len(goals) // args.batch_size) * args.batch_size
                if train_samples != expected_train or val_samples != len(val_goals):
                    raise ValueError('Epoch did not consume the declared complete train/validation loaders')
                ckpt = output / 'checkpoints' / f'epoch_{epoch + 1:04d}_{time.time_ns()}'
                ckpt.mkdir(parents=True, exist_ok=False)
                snapshot = {'head': policy.action_predictor.state_dict(), 'optimizer': optimizer.state_dict(),
                            'grad_scaler': amp.state_dict() if amp is not None else None,
                            'rng': capture_rng(include_cuda=args.device == 'cuda'),
                            'train_goal_rng': goals.rng.bit_generator.state,
                            'validation_goal_rng': val_goals.rng.bit_generator.state,
                            'train_generator': generator.get_state(), 'validation_generator': val_generator.get_state(),
                            'next_epoch': epoch + 1, 'global_updates': global_updates,
                            'frozen_state_sha256': frozen_hash, 'protocol_sha256': protocol_hash, 'metrics': metrics}
                atomic_torch_save(ckpt / 'resume.pt', snapshot)
                atomic_torch_save(ckpt / 'modelonly.pt', policy.state_dict())
                atomic_json(ckpt / 'config.json', factory)
                entry = {'directory': str(ckpt.relative_to(output)), 'epoch': epoch + 1,
                         'resume_sha256': sha_file(ckpt / 'resume.pt'), 'modelonly_sha256': sha_file(ckpt / 'modelonly.pt'),
                         'metrics': metrics, 'retained': True}
                manifest['last'] = entry
                if manifest['best'] is None or metrics['validation_mse'] < manifest['best']['metrics']['validation_mse']:
                    manifest['best'] = entry
                manifest['epochs'].append(entry)
                atomic_json(output / 'checkpoint_manifest.json', manifest)
                status.update(completed_epochs=epoch + 1, **metrics)
                persist()
                with (output / 'epochs.jsonl').open('a') as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            status.update(status='completed', stage='complete', exit_code=0)
            persist()
            (output / 'exit_code').write_text('0\n')
            return 0
        except BaseException as error:
            traceback.print_exc()
            if not existed or protocol_accepted:
                status.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                              exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
                persist()
                (output / 'exit_code').write_text(str(status['exit_code']) + '\n')
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            logging.getLogger().removeHandler(handler)


if __name__ == '__main__':
    raise SystemExit(main())
