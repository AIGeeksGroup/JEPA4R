"""Single-frame official-head extension; fresh run, exact logical-boundary resume."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np
import torch
from dataset import CubeOfficialDataset, LogicalSampler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def epoch_counts(samples, micro, accumulation):
    logical = micro * accumulation
    return dict(samples=samples, micro_steps=math.ceil(samples / micro), optimizer_steps=math.ceil(samples / logical),
                last_logical_samples=samples % logical or logical)


def capture_rng(include_cuda=False):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if include_cuda else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None: torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def module_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def worker_init(worker):
    torch.set_num_threads(1)


def make_loader(dataset, sampler, workers, pin_memory=True):
    kwargs = dict(dataset=dataset, batch_sampler=sampler, num_workers=workers, pin_memory=pin_memory,
                  generator=torch.Generator().manual_seed(0), worker_init_fn=worker_init)
    if workers: kwargs.update(multiprocessing_context='spawn', persistent_workers=False, prefetch_factor=2)
    return torch.utils.data.DataLoader(**kwargs)


def cleanup_loader(iterator):
    if iterator is not None and hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()  # Only this owned DataLoader's workers/pin thread.


def policy_forward(policy, current, goal, batch, device):
    if 'current_state' in batch or 'goal_state' in batch:
        return policy(current, goal, batch['current_state'].to(device, non_blocking=True),
                      batch['goal_state'].to(device, non_blocking=True))
    return policy(current, goal)


def train_logical_step(policy, micro_batches, optimizer, device='cpu', precision='fp32', amp=None):
    from policy import unit_gaussian_nll
    count = sum(len(batch['action']) for batch in micro_batches)
    if count <= 0: raise ValueError('Nonempty logical batch required')
    optimizer.zero_grad(set_to_none=True)
    nll_sum, mse_sum = 0., 0.
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    scale_before = amp.get_scale() if amp is not None else 1.
    for batch in micro_batches:
        current, goal, action = [batch[key].to(device, non_blocking=True) for key in ('current', 'goal', 'action')]
        if action.ndim != 2 or action.shape[1] != 5 or not torch.isfinite(action).all(): raise ValueError('Raw action target invalid')
        with torch.autocast(device_type=device, dtype=dtype, enabled=precision != 'fp32'):
            mean = policy_forward(policy, current, goal, batch, device)
        # Keep the declared NLL/metrics in FP32 even under encoder/head AMP.
        nll = unit_gaussian_nll(mean.float(), action.float())
        mse = (mean.float() - action.float()).square().mean()
        if not torch.isfinite(nll) or not torch.isfinite(mse): raise ValueError('Nonfinite logical loss')
        weighted = nll * (len(action) / count)
        if amp is not None: amp.scale(weighted).backward()
        else: weighted.backward()
        nll_sum += float(nll.detach()) * len(action); mse_sum += float(mse.detach()) * len(action)
    if amp is not None: amp.unscale_(optimizer)
    gradients = [p.grad for p in policy.action_head.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients): raise ValueError('Nonfinite/missing accumulated head gradients')
    if any(p.requires_grad or p.grad is not None for p in policy.encoder.parameters()): raise ValueError('Frozen trunk unexpectedly trainable')
    norm = float(torch.sqrt(sum(value.float().square().sum() for value in gradients)))
    if not math.isfinite(norm) or norm <= 0: raise ValueError('Accumulated gradient norm must be finite and nonzero')
    if any(float(group['lr']) != .0003 for group in optimizer.param_groups): raise ValueError('Actual Adam LR differs from3e-4')
    if amp is not None: amp.step(optimizer); amp.update()
    else: optimizer.step()
    scale_after = amp.get_scale() if amp is not None else 1.
    skipped = scale_after < scale_before
    if skipped: raise ValueError('FP16 optimizer skip is not accepted')
    return dict(samples=count, micro_steps=len(micro_batches), nll=nll_sum / count, raw_mse=mse_sum / count,
                gradient_l2=norm, actual_lr=float(optimizer.param_groups[0]['lr']), amp_scale_before=scale_before,
                amp_scale_after=scale_after, optimizer_step_skipped=skipped)


def validate_binding(index_path, seal_path):
    seal = json.loads(Path(seal_path).read_text()); index = json.loads(Path(index_path).read_text())
    if seal['status'] != 'sealed_data_complete' or seal['exit_code'] != 0 or index['status'] != 'passed' or index['exit_code'] != 0:
        raise ValueError('Full RGB seal/index must have passed')
    if sha(index_path) != seal['files']['dataset_index.json']['sha256']: raise ValueError('Sealed full index SHA differs')
    for key in ('final_independent_audit.json', 'final_lineage_reader_audit.json'):
        entry = seal['files'][key]
        if sha(entry['path']) != entry['sha256'] or json.loads(Path(entry['path']).read_text())['exit_code'] != 0:
            raise ValueError('Sealed actual source/reader audit differs')
    for split, expected in (('train', 1000), ('val', 100)):
        if len(index[split]) != expected or [entry['episode'] for entry in index[split]] != list(range(expected)):
            raise ValueError('Full Cube split/episode scope differs')
        for entry in index[split]:
            if sha(entry['path']) != entry['sha256']: raise ValueError('Successful source shard was modified')
    for source in seal['source'].values():
        if sha(source['path']) != source['archive_sha256']: raise ValueError('Raw Cube source archive changed')
    return dict(data_index_sha256=sha(index_path), data_seal_sha256=sha(seal_path), producer_protocol_sha256=index['protocol_sha256'],
                valid_train_current=1000000, valid_val_current=100000, raw_actions_identity=True, history=1,
                goal='pcur0/traj1/random0, np.round in raw compact terminal boundary999; terminal raw1000 audit-only')


def evaluate(policy, loader, device, precision, progress=None):
    from policy import unit_gaussian_nll
    policy.eval(); count, nll, mse = 0, 0., 0.
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    iterator = iter(loader)
    try:
        with torch.inference_mode():
            for index, batch in enumerate(iterator):
                action = batch['action'].to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=dtype, enabled=precision != 'fp32'):
                    mean = policy_forward(policy, batch['current'].to(device, non_blocking=True), batch['goal'].to(device, non_blocking=True), batch, device)
                loss = unit_gaussian_nll(mean.float(), action.float()); error = (mean.float() - action).square().mean()
                if not torch.isfinite(loss) or not torch.isfinite(error): raise ValueError('Nonfinite validation')
                count += len(action); nll += float(loss) * len(action); mse += float(error) * len(action)
                if progress is not None and (index % 50 == 0 or count == 100000): progress(count)
    finally:
        cleanup_loader(iterator)
    return dict(samples=count, nll=nll / count, raw_mse=mse / count)


def publish_epochs_view(manifest, path):
    path = Path(path); temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as handle:
        for entry in manifest['epochs']: handle.write(json.dumps(entry, allow_nan=False) + '\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)  # Own generated view; manifest is authority.


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-index', 'data-seal', 'source-checkpoint', 'repo-root', 'smoke-utils-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--io-tools', type=Path, default=Path('/root/autodl-tmp/workspace/users/lewm/lewm-gcbc-formal-20261006-0054/trainer.py'))
    parser.add_argument('--epochs', type=int, default=10); parser.add_argument('--micro-batch', type=int, default=64)
    parser.add_argument('--accumulation', type=int, default=16); parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--open-files', type=int, default=16); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    args = parser.parse_args()
    lewm = args.repo_root / 'lewm_pcd'
    sys.path[:0] = [str(args.smoke_utils_root), str(lewm), str(lewm / 'third_party/stable-worldmodel')]
    spec = importlib.util.spec_from_file_location('frozen_legacy_atomic_io', args.io_tools)
    io = importlib.util.module_from_spec(spec); spec.loader.exec_module(io)
    from policy import create_policy, make_optimizer
    import hydra
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    binding = validate_binding(args.data_index, args.data_seal)
    cfg = OmegaConf.load(args.source_checkpoint.parent / 'config.yaml')
    factory = dict(_target_='policy.create_policy', source_checkpoint=str(args.source_checkpoint.resolve()), modality='pixels',
                   latent_dim=192, history_size=1, action_dim=5, seed=args.seed)
    protocol = dict(**binding, factory=factory, source_checkpoint_sha256=sha(args.source_checkpoint), policy_source_sha256=sha(Path(__file__).with_name('policy.py')),
        dataset_source_sha256=sha(Path(__file__).with_name('dataset.py')), trainer_source_sha256=sha(__file__),
        source_config_sha256=sha(args.source_checkpoint.parent / 'config.yaml'), atomic_io_sha256=sha(args.io_tools),
        frozen_adapter_sha256=sha(args.smoke_utils_root / 'lewm_gcbc_adapter.py'), torch_version=torch.__version__,
        precision=args.precision, micro_batch=args.micro_batch, accumulation=args.accumulation, workers=args.workers, open_files=args.open_files,
        optimizer='Adam', lr=.0003, weight_decay=0., betas=[.9, .999], eps=1e-8, loss='unit std Gaussian NLL meanB sumA5',
        epoch_counts=epoch_counts(1000000, args.micro_batch, args.accumulation), validation=dict(samples=100000, fixed_goal_seed=0), seed=args.seed,
        comparison_budget='10epochs9770 updates ifmicro64accum16; differs from official state1Mupdates')
    protocol.update(parent_torch_threads=1, worker_torch_threads=1, pin_memory=True, non_blocking=True,
                    loader_generator_seed=0, prefetch_factor=2 if args.workers else None,
                    torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    sampler_order_snapshot='Torch shuffle generator before/after epoch; recreate full order with pinnedTorch rather than store8MB array')
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and json.loads((args.output / 'protocol.json').read_text()) != protocol: raise ValueError('Resume protocol differs')
    if not args.resume: io.atomic_json(args.output / 'protocol.json', protocol)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    policy = create_policy(**{key: value for key, value in factory.items() if key != '_target_'}).to(args.device)
    frozen_hash = module_hash(policy.encoder); optimizer = make_optimizer(policy)
    amp = torch.amp.GradScaler('cuda', enabled=True) if args.precision == 'fp16' else None
    train = CubeOfficialDataset(args.data_index, 'train', hydra.utils.instantiate(cfg.data.preprocessor), args.open_files)
    val = CubeOfficialDataset(args.data_index, 'val', hydra.utils.instantiate(cfg.data.preprocessor), args.open_files)
    sampler = LogicalSampler(len(train), args.micro_batch, args.accumulation, args.seed)
    manifest = dict(protocol_sha256=protocol_sha, epochs=[], best=None, last=None); epoch, global_micro, global_optimizer = 0, 0, 0
    sums = dict(samples=0, nll=0., raw_mse=0.)
    if args.resume:
        manifest = json.loads((args.output / 'checkpoint_manifest.json').read_text())
        if manifest.get('protocol_sha256') != protocol_sha: raise ValueError('Resume manifest protocol differs')
        resume_path = (args.output / manifest['last']['resume']).resolve()
        if not resume_path.is_relative_to(args.output.resolve()) or sha(resume_path) != manifest['last']['resume_sha256']:
            raise ValueError('Resume path/hash binding differs')
        snapshot = torch.load(resume_path, map_location='cpu', weights_only=False)
        if snapshot['protocol_sha256'] != protocol_sha or snapshot['frozen_state_sha256'] != frozen_hash: raise ValueError('Resume source/config differs')
        policy.action_head.load_state_dict(snapshot['head'], strict=True); optimizer.load_state_dict(snapshot['optimizer'])
        if amp is not None: amp.load_state_dict(snapshot['amp'])
        sampler = LogicalSampler.from_state_dict(snapshot['sampler']); restore_rng(snapshot['rng'])
        epoch, global_micro, global_optimizer, sums = snapshot['epoch'], snapshot['global_micro_steps'], snapshot['global_optimizer_steps'], snapshot['sums']
        counts = protocol['epoch_counts']
        if global_optimizer != epoch * counts['optimizer_steps'] + math.ceil(sampler.committed / sampler.logical) or global_micro != epoch * counts['micro_steps'] + math.ceil(sampler.committed / sampler.micro) or sums['samples'] != sampler.committed:
            raise ValueError('Resume logical/epoch sample or update counters differ')
        publish_epochs_view(manifest, args.output / 'epochs.jsonl')
    state = dict(status='running', stage='train', exit_code=None, completed_epochs=epoch, epoch=epoch,
                 protocol_sha256=protocol_sha, frozen_state_sha256=frozen_hash, started_at=datetime.now(timezone.utc).isoformat())
    def persist():
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), epoch_consumed_samples=sampler.committed,
                     global_micro_steps=global_micro, global_optimizer_steps=global_optimizer, current_lr=float(optimizer.param_groups[0]['lr']))
        io.atomic_json(args.output / 'status.json', state)
    def checkpoint(end=False, metrics=None):
        folder = args.output / 'checkpoints' / f'epoch_{epoch:04d}_update_{global_optimizer:06d}_{uuid.uuid4().hex[:8]}'
        folder.mkdir(parents=True, exist_ok=False)
        snapshot = dict(head=policy.action_head.state_dict(), optimizer=optimizer.state_dict(), amp=amp.state_dict() if amp is not None else None,
            rng=capture_rng(args.device == 'cuda'), sampler=sampler.state_dict(), epoch=epoch, global_micro_steps=global_micro,
            global_optimizer_steps=global_optimizer, sums=sums, protocol_sha256=protocol_sha, frozen_state_sha256=frozen_hash)
        io.atomic_torch_save(folder / 'resume.pt', snapshot)
        entry = dict(resume=str((folder / 'resume.pt').relative_to(args.output)), resume_sha256=sha(folder / 'resume.pt'), epoch=epoch, metrics=metrics,
                     global_micro_steps=global_micro, global_optimizer_steps=global_optimizer, frozen_state_sha256=frozen_hash)
        if end:
            io.atomic_torch_save(folder / 'modelonly.pt', policy.state_dict()); io.atomic_json(folder / 'config.json', factory)
            entry.update(modelonly=str((folder / 'modelonly.pt').relative_to(args.output)), modelonly_sha256=sha(folder / 'modelonly.pt'),
                         train_samples=1000000, validation_samples=100000)
            manifest['epochs'].append(entry)
            if manifest['best'] is None or metrics['val_nll'] < manifest['best']['metrics']['val_nll']: manifest['best'] = entry
        manifest['last'] = entry; io.atomic_json(args.output / 'checkpoint_manifest.json', manifest)
        if end: publish_epochs_view(manifest, args.output / 'epochs.jsonl')
    persist()
    active_iterator = None
    try:
        while epoch < args.epochs:
            train_start = time.perf_counter()
            policy.train(); loader = make_loader(train, sampler, args.workers, pin_memory=args.device == 'cuda'); iterator = iter(loader)
            active_iterator = iterator
            while sampler.committed < len(train):
                count = min(sampler.logical, len(train) - sampler.committed)
                batches = [next(iterator) for _ in range(math.ceil(count / sampler.micro))]
                metrics = train_logical_step(policy, batches, optimizer, args.device, args.precision, amp)
                if module_hash(policy.encoder) != frozen_hash: raise ValueError('Frozen encoder/projector params/buffers changed')
                sampler.commit_logical_batch(metrics['samples']); global_optimizer += 1; global_micro += metrics['micro_steps']
                sums['samples'] += metrics['samples']; sums['nll'] += metrics['nll'] * metrics['samples']; sums['raw_mse'] += metrics['raw_mse'] * metrics['samples']
                state.update(train_nll=sums['nll'] / sums['samples'], train_raw_mse=sums['raw_mse'] / sums['samples'], latest_step=metrics)
                persist()
                if global_optimizer % args.checkpoint_interval == 0: checkpoint()
            if sums['samples'] != len(train): raise ValueError('Epoch sample count differs; no hidden drops accepted')
            cleanup_loader(iterator); active_iterator = None
            del iterator, loader
            train_elapsed = time.perf_counter() - train_start
            fixed_val = LogicalSampler(len(val), args.micro_batch, 1, seed=0)
            state.update(stage='validation', validation_consumed_samples=0); persist()
            val_start = time.perf_counter()
            def val_progress(count):
                state['validation_consumed_samples'] = count; persist()
            validation = evaluate(policy, make_loader(val, fixed_val, args.workers, pin_memory=args.device == 'cuda'), args.device, args.precision, val_progress)
            if validation['samples'] != len(val) or module_hash(policy.encoder) != frozen_hash: raise ValueError('Full val/frozen state failed')
            metrics = dict(train_nll=sums['nll'] / sums['samples'], train_raw_mse=sums['raw_mse'] / sums['samples'],
                           val_nll=validation['nll'], val_raw_mse=validation['raw_mse'], train_elapsed_seconds=train_elapsed,
                           validation_elapsed_seconds=time.perf_counter() - val_start)
            epoch += 1; sampler.next_epoch(); sums = dict(samples=0, nll=0., raw_mse=0.)
            state.update(completed_epochs=epoch, epoch=epoch, stage='train', **metrics); checkpoint(end=True, metrics=metrics); persist()
        state.update(status='passed', stage='complete', exit_code=0)
    except BaseException:
        import traceback
        state.update(status='failed', exit_code=130 if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 1, error=traceback.format_exc())
    finally:
        cleanup_loader(active_iterator)
    persist(); (args.output / 'exit_code').write_text(str(state['exit_code']) + '\n')
    return state['exit_code']


if __name__ == '__main__': raise SystemExit(main())
