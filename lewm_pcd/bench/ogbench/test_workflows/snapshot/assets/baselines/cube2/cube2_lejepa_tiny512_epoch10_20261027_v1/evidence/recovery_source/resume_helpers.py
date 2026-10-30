"""Weight-only recovery, read-only first-update evidence, verified exports.

No training model, optimizer update, data reader or process is created here.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import warnings
import zipfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes_equal(left, right):
    import torch
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def audit_checkpoint(path, expected_sha256=None, expected_bytes=None, expected_payload=None):
    import torch
    path = Path(path)
    size = path.stat().st_size
    if size <= 0 or (expected_bytes is not None and size != expected_bytes):
        raise ValueError(f'Invalid export size: {path}: {size}, expected {expected_bytes}')
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'Export SHA256 mismatch: {path}')
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'Export ZIP CRC failure: {bad}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'backbone', 'head', 'proj'}:
        raise ValueError('Export must contain exactly backbone/head/proj')
    count, floating = 0, 0
    for module_name, state in payload.items():
        if not isinstance(state, dict):
            raise ValueError(f'Invalid state dictionary: {module_name}')
        if expected_payload is not None and set(state) != set(expected_payload[module_name]):
            raise ValueError(f'Export tensor keys changed: {module_name}')
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f'Non-tensor export entry: {module_name}.{name}')
            count += 1
            if tensor.is_floating_point() or tensor.is_complex():
                floating += 1
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'Nonfinite export tensor: {module_name}.{name}')
            if expected_payload is not None and not tensor_bytes_equal(tensor, expected_payload[module_name][name]):
                raise ValueError(f'Export tensor bytes changed: {module_name}.{name}')
    return {'path': str(path.resolve()), 'bytes': size, 'sha256': digest,
            'zip_crc_ok': True, 'tensor_count': count, 'finite_float_tensor_count': floating,
            'tensor_bytes_equal_to_expected': expected_payload is not None}, payload


def require_free_space(root, minimum_bytes):
    free = shutil.disk_usage(root).free
    if free < minimum_bytes:
        raise OSError(f'Insufficient free space at {root}: {free} < {minimum_bytes}')
    return free


def _safe_child(root, name):
    root = Path(root).resolve(strict=True)
    target = root / name
    if target.resolve().parent != root:
        raise ValueError(f'Export path escapes explicit root: {name}')
    return target


def _publish_new(partial, target):
    # link is atomic and refuses existing destinations on Linux, unlike rename.
    # A filesystem without hardlinks fails visibly; the validated stage survives.
    os.link(partial, target)
    Path(partial).unlink()  # Only this call's redundant partial after publication.


def verified_export(payload, stage_root, output_root, name):
    import torch
    if not re.fullmatch(r'(weights_epoch_[1-9][0-9]*|smoke_after_[1-9][0-9]*_batches)\.pt', name):
        raise ValueError(f'Unexpected export name: {name}')
    stage_root, output_root = Path(stage_root), Path(output_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    staged, final = _safe_child(stage_root, name), _safe_child(output_root, name)
    if staged.exists() or final.exists():
        raise FileExistsError(f'Export already exists: {staged} or {final}')
    require_free_space(stage_root, 64 * 1024**2)
    local_partial = _safe_child(stage_root, f'.{name}.{uuid.uuid4().hex}.partial')
    with local_partial.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    staged_meta, _ = audit_checkpoint(local_partial, expected_payload=payload)
    _publish_new(local_partial, staged)
    staged_meta['path'] = str(staged.resolve())
    write_json_new(stage_root / f'{name}.stage_receipt.json', {
        'schema': 'verified_local_stage_v1', 'at_utc': utc_now(), 'stage': staged_meta,
        'primary_publication': 'pending', 'external_backup': 'pending'})
    require_free_space(output_root, staged_meta['bytes'] + 16 * 1024**2)
    output_partial = _safe_child(output_root, f'.{name}.{uuid.uuid4().hex}.partial')
    with staged.open('rb') as source, output_partial.open('xb') as target:
        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    audit_checkpoint(output_partial, expected_sha256=staged_meta['sha256'],
                     expected_bytes=staged_meta['bytes'], expected_payload=payload)
    _publish_new(output_partial, final)
    final_meta, _ = audit_checkpoint(final, expected_sha256=staged_meta['sha256'],
                                      expected_bytes=staged_meta['bytes'], expected_payload=payload)
    receipt = {'schema': 'verified_export_v1', 'at_utc': utc_now(), 'stage': staged_meta,
               'primary': final_meta,
               'primary_storage_role': 'configured_output_root; consult plan for local/shared location',
               'external_backup': 'pending; separate backup receipt required'}
    write_json_new(output_root / f'{name}.receipt.json', receipt)
    write_json_new(stage_root / f'{name}.published_receipt.json', receipt)
    print(json.dumps({'verified_export': name, 'sha256': final_meta['sha256'],
                      'bytes': final_meta['bytes'], 'primary_path': str(final),
                      'external_backup': 'pending'}), flush=True)
    return receipt


def validate_resume_args(args):
    if not 0 <= args.start_epoch < args.epochs:
        raise ValueError('Require 0 <= start_epoch < total epochs')
    if bool(args.resume_model) != bool(args.start_epoch):
        raise ValueError('Weight-only resume requires both resume-model and positive start-epoch')
    if args.resume_model is not None and args.verified_export_root is None:
        raise ValueError('Recovery requires a retained local verified-export-root')


def _model_payload(net):
    return {'backbone': net.backbone.state_dict(), 'head': net.head.state_dict(), 'proj': net.proj.state_dict()}


def load_resume_model(net, args, out, binding):
    if args.resume_model is None:
        return None
    if Path(args.resume_model).resolve() != Path(binding['parent_model']).resolve():
        raise ValueError('Resume model differs from frozen binding')
    metadata, payload = audit_checkpoint(args.resume_model, binding['parent_sha256'], binding['parent_bytes'])
    for contract, actual in [('parent_tensor_count', 'tensor_count'),
                             ('parent_floating_tensor_count', 'finite_float_tensor_count')]:
        if contract in binding and metadata[actual] != binding[contract]:
            raise ValueError(f'Parent tensor-count contract mismatch: {contract}')
    for name in ['backbone', 'head', 'proj']:
        getattr(net, name).load_state_dict(payload[name], strict=True)
    actual = _model_payload(net)
    for name, tensors in payload.items():
        if set(tensors) != set(actual[name]):
            raise ValueError(f'Strict load key mismatch: {name}')
        for key, value in tensors.items():
            if not tensor_bytes_equal(value, actual[name][key]):
                raise ValueError(f'Loaded model tensor bytes mismatch: {name}.{key}')
    receipt = {'schema': 'weight_only_parent_load_v1', 'at_utc': utc_now(), 'parent': metadata,
               'strict_modules': ['backbone', 'head', 'proj'], 'all_loaded_tensor_bytes_equal': True,
               'exact_resume': False, 'optimizer_scaler_rng_restored': False,
               'seed_rule': 'native torch.manual_seed(0); Python/NumPy retain new-process initialization; loader RNG rebuilt'}
    write_json_new(Path(out) / 'parent_load_receipt.json', receipt)
    return receipt


def prepare_resume(scheduler, opt, net, args, batches, out, parent_receipt, binding):
    if args.resume_model is None:
        return None
    if (args.start_epoch, args.epochs, batches) != (binding['start_epoch'], binding['total_epochs'], binding['batches_per_epoch']):
        raise ValueError('Recovery epoch/batch contract changed')
    if len(opt.state) != 0:
        raise ValueError('Rebuilt Adam state must be empty before recovery')
    logical_steps = args.start_epoch * batches
    with warnings.catch_warnings(record=True) as caught:
        for _ in range(logical_steps):
            scheduler.step()
    receipt = {'schema': 'weight_only_scheduler_position_v1', 'at_utc': utc_now(),
               'exact_resume': False, 'parent_loaded': bool(parent_receipt),
               'replayed_scheduler_calls_without_optimizer_step': logical_steps,
               'logical_steps_before': logical_steps, 'scheduler_last_epoch': scheduler.last_epoch,
               'first_new_nominal_batch_one_based': logical_steps + 1,
               'lr_before_first_new_batch': scheduler.get_last_lr(),
               'adam_state_entries_before_first_batch': len(opt.state),
               'adam_step_counter_start': 0, 'scaler': 'fresh native enabled GradScaler',
               'warnings_from_replay': [str(item.message) for item in caught]}
    if scheduler.last_epoch != logical_steps:
        raise ValueError('Scheduler did not reach the declared native logical position')
    if 'expected_lr_before' in binding and any(abs(lr - binding['expected_lr_before']) > 1e-12 for lr in scheduler.get_last_lr()):
        raise ValueError('Scheduler recovery LR differs from the frozen native position')
    write_json_new(Path(out) / 'recovery_position_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    tracker = {'logical_steps_before': logical_steps, 'observation': None, 'handles': []}
    if net is None:  # CPU scheduler-only check, no training model constructed.
        return tracker
    import torch
    named = list(net.named_parameters())

    def before_step(optimizer, positional, keyword):
        if tracker['observation'] is not None:
            return
        finite = all(bool(torch.isfinite(p.grad).all()) for _, p in named if p.grad is not None)
        selected = next(((name, p) for name, p in named if p.grad is not None and bool(torch.count_nonzero(p.grad))), None)
        if selected is None:
            raise ValueError('First optimizer submission has no nonzero gradient')
        tracker['selected'] = selected
        tracker['before'] = selected[1].detach().cpu().clone()
        tracker['gradient_finite'] = finite

    def after_step(optimizer, positional, keyword):
        name, parameter = tracker['selected']
        state_steps = [int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
        tracker['observation'] = {'gradient_finite': tracker['gradient_finite'],
                                  'weight_name': name,
                                  'weight_bytes_changed': not tensor_bytes_equal(tracker['before'], parameter),
                                  'adam_state_entries': len(optimizer.state),
                                  'adam_step_min': min(state_steps), 'adam_step_max': max(state_steps)}
    tracker['handles'] = [opt.register_step_pre_hook(before_step), opt.register_step_post_hook(after_step)]
    return tracker


def record_first_update(tracker, scheduler, out):
    if tracker is None or tracker.get('recorded'):
        return
    observation = tracker['observation']
    if observation is None:
        raise ValueError('First new nominal batch did not submit an optimizer update (GradScaler skip)')
    if not observation['gradient_finite'] or not observation['weight_bytes_changed']:
        raise ValueError('First optimizer update has nonfinite gradients or unchanged selected weight')
    if (observation['adam_step_min'], observation['adam_step_max']) != (1, 1):
        raise ValueError('Rebuilt Adam first actual update counter is not 1')
    if scheduler.last_epoch != tracker['logical_steps_before'] + 1:
        raise ValueError('First recovery scheduler logical counter mismatch')
    write_json_new(Path(out) / 'first_actual_update_receipt.json', {
        'schema': 'weight_only_first_actual_update_v1', 'at_utc': utc_now(), **observation,
        'scheduler_last_epoch': scheduler.last_epoch, 'lr_after_batch': scheduler.get_last_lr(),
        'optimizer_counter_semantics': 'new Adam actual update 1; scheduler logical batch position is separate'})
    for handle in tracker['handles']:
        handle.remove()
    tracker['recorded'] = True
    tracker.pop('before', None)


def save_epoch(net, out, epoch, verified_export_root):
    import torch
    payload = _model_payload(net)
    if verified_export_root is None:
        torch.save(payload, Path(out) / f'weights_epoch_{epoch}.pt')
        return
    return verified_export(payload, verified_export_root, out, f'weights_epoch_{epoch}.pt')


def smoke_export(net, out, args, batches):
    if args.verified_export_root is None:
        return
    name = f'smoke_after_{batches}_batches.pt'
    receipt = verified_export(_model_payload(net), args.verified_export_root, out, name)
    # Reuse the already constructed model's three module classes, without another
    # GPU model or optimizer step. Fresh CPU torch.load is checked against live weights.
    audit_checkpoint(Path(out) / name, expected_sha256=receipt['primary']['sha256'],
                     expected_bytes=receipt['primary']['bytes'], expected_payload=_model_payload(net))
    write_json_new(Path(out) / 'smoke_export_fresh_readback.json', {
        'schema': 'recovery_smoke_export_readback_v1', 'at_utc': utc_now(),
        'model_export': receipt['primary'], 'fresh_cpu_torch_load_all_tensor_bytes_equal': True,
        'fresh_model_strict_reload': 'pending independent root gate'})
