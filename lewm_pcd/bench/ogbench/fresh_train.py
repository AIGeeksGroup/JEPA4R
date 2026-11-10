"""Explicit fresh-data dispatch into original trainers; default is plan-only."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

from . import fresh_data
from .training_runtime import build_environment, load_recipe, pinned_native, load_tiny_trainer, code_root

METHODS = ('pldm', 'lejepa', 'lewm_rgb', 'lewm_pcd')


def describe_training(method, data_manifest, output_root, *, phase='formal', workers=None, prefetch=None):
    method = {'lewm2d': 'lewm_rgb', 'lewm3d': 'lewm_pcd'}.get(method, method)
    if method not in METHODS or phase not in ('formal', 'smoke'):
        raise ValueError('explicit supported method and formal/smoke phase required')
    binding = fresh_data.load_binding(data_manifest, verify=False)
    if phase != binding['mode']:
        raise ValueError('training mode must match data mode: formal requires formal data; smoke requires the 2/1 smoke binding')
    if workers is not None and (type(workers) is not int or workers < (1 if method == 'lejepa' else 0)):
        raise ValueError('invalid workers; Tiny requires positive workers')
    if prefetch is not None and (type(prefetch) is not int or prefetch < 1):
        raise ValueError('prefetch must be positive')
    cubes = binding['cube_count']
    result = dict(schema='cube_fresh_training_plan_v1', method=method, cube_count=cubes,
                  mode=phase, data_manifest=str(Path(data_manifest).resolve()),
                  data_manifest_sha256=fresh_data.sha256(data_manifest),
                  output=str(Path(output_root).resolve()), workers=workers, prefetch=prefetch,
                  formal_acceptance_verified=False, execute=False)
    if method in ('lewm_rgb', 'lewm_pcd'):
        modality = 'rgb' if method == 'lewm_rgb' else 'pcd'
        cfg = load_recipe(method)
        label = 'fresh_cube' + str(cubes) + '_' + method
        cfg['output_model_name'] = cfg['subdir'] = label
        for role, split in [('dataset', 'train'), ('validation_dataset', 'validation')]:
            cfg['data'][role] = dict(name=binding['views'][split + '_' + modality]['path'],
                num_steps=4, frameskip=5,
                keys_to_load=['pixels', 'action'] if modality == 'rgb' else ['pcd_xyz', 'pcd_rgb', 'action'],
                keys_to_cache=['action'])
        if workers is not None:
            cfg['loader']['num_workers'] = cfg['num_workers'] = cfg['validation_num_workers'] = workers
        if cfg['loader']['num_workers'] == 0:
            cfg['loader'].pop('prefetch_factor', None)
            cfg['loader'].pop('persistent_workers', None)
        elif prefetch is not None:
            cfg['loader']['prefetch_factor'] = prefetch
        if phase == 'smoke':
            cfg['trainer'].update(max_epochs=1, limit_train_batches=2, limit_val_batches=1)
        result.update(config=cfg, recipe={'epochs': cfg['trainer']['max_epochs'], 'batch_size': 128},
                      protocol='episode_heldout_v1')
    else:
        result.update(recipe=dict(epochs=10 if phase == 'formal' or method == 'lejepa' else 1,
                                  batch_size=128 if method == 'pldm' else 256),
                      protocol='episode_heldout_v1' if method == 'pldm' else 'canonical_tiny_train_only')
        if method == 'pldm':
            result['recipe']['native_scheduler_epochs'] = 10
            if phase == 'smoke':
                result['recipe'].update(limit_train_batches=2, limit_val_batches=2)
    return result


def build_plan(cube_count, method, data_manifest, output_root, mode='formal', python=None, workers=None, prefetch=None):
    plan = describe_training(method, data_manifest, output_root, phase=mode, workers=workers, prefetch=prefetch)
    if type(cube_count) is not int or cube_count != plan['cube_count']:
        raise ValueError('Cube count differs from fresh data binding')
    command = [python or sys.executable, '-m', 'bench.ogbench.fresh_train', '--method', plan['method'],
               '--data-manifest', plan['data_manifest'], '--output', plan['output'], '--mode', mode, '--execute']
    for flag, value in [('--workers', workers), ('--prefetch', prefetch)]:
        if value is not None:
            command.extend([flag, str(value)])
    plan['command'] = command
    return plan


def mark_training_returned(plan, output):
    """Record the own native function's exit0 separately from strict acceptance."""
    output = Path(output)
    source_binding = _source_binding(plan['method'], plan['cube_count'])
    previous_path = output / 'NATIVE_TRAINING.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous.get('source_binding', source_binding) != source_binding:
            raise ValueError('native training source changed while running')
    fresh_data.save(output / 'NATIVE_TRAINING.json', dict(actual_exit_code=0,
        stage='native_returned_pending_acceptance', plan_sha256=fresh_data.sha256(output / 'plan.json'),
        data_manifest_sha256=plan['data_manifest_sha256'], source_binding=source_binding,
        updated_at=datetime.now(timezone.utc).isoformat()))


def build_accept_plan(training_output, output_root, python=None):
    """Recover acceptance for exactly one completed own run; never run fit again."""
    original = Path(training_output).resolve()
    marker_path = original / 'NATIVE_TRAINING.json'
    if not marker_path.is_file():
        raise ValueError('own native training actual exit0 receipt is required')
    marker = json.loads(marker_path.read_text(encoding='utf8'))
    if type(marker.get('actual_exit_code')) is not int or marker['actual_exit_code'] != 0:
        raise ValueError('own native training actual exit0 is required; running/failed jobs cannot be accepted')
    plan = json.loads((original / 'plan.json').read_text(encoding='utf8'))
    if fresh_data.sha256(original / 'plan.json') != marker['plan_sha256']:
        raise ValueError('own native training plan changed')
    if marker['source_binding'] != _source_binding(plan['method'], plan['cube_count']):
        raise ValueError('own native training source binding changed')
    fresh_data.load_binding(plan['data_manifest'])
    if fresh_data.sha256(plan['data_manifest']) != marker['data_manifest_sha256']:
        raise ValueError('own native training data binding changed')
    if (original / 'WM_ACCEPTANCE.json').exists():
        raise ValueError('own training run already has an acceptance receipt')
    plan.update(output=str(Path(output_root).resolve()), accept_only=True, training_output=str(original),
                original_plan_sha256=marker['plan_sha256'], execute=False)
    plan['command'] = [python or sys.executable, '-m', 'bench.ogbench.fresh_train', '--accept-only',
                       '--training-output', str(original), '--output', plan['output'], '--execute']
    return plan


@contextmanager
def _argv(values):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


@contextmanager
def route_loader(native, plan):
    """Route only explicitly bound paths, then restore the original service."""
    binding = fresh_data.load_binding(plan['data_manifest'])
    original = native.swm.data.load_dataset
    modality = 'pcd' if plan['method'] == 'lewm_pcd' else 'rgb'
    paths = {Path(row['path']).resolve(): key for key, row in binding['views'].items() if key.endswith('_' + modality)}
    def load(name, **kwargs):
        key = paths.get(Path(name).resolve())
        if key is None:
            raise ValueError('unbound fresh training dataset path')
        kwargs.pop('format', None)
        return fresh_data.make_reader(plan['data_manifest'], key, **kwargs)
    native.swm.data.load_dataset = load
    try:
        yield
    finally:
        native.swm.data.load_dataset = original


@contextmanager
def _record_loaders(native, output):
    original = native.spt.data.DataModule
    def data_module(*args, **kwargs):
        fresh_data.save(output / 'native_loader_counts.json', {
            split: dict(samples=len(kwargs[split].dataset), batches=len(kwargs[split])) for split in ('train', 'val')})
        return original(*args, **kwargs)
    native.spt.data.DataModule = data_module
    try:
        yield
    finally:
        native.spt.data.DataModule = original


def _pldm(plan, output, training_root=None):
    from .pldm import train
    native_output = Path(training_root or output) / 'native'
    phase = 'gate' if plan['mode'] == 'smoke' else 'formal'
    common = ['current_pldm', '--cube-count', str(plan['cube_count']),
              '--data-manifest', plan['data_manifest'], '--phase', phase, '--run-tag', 'fresh',
              '--workers', str(plan['workers'] if plan['workers'] is not None else 20),
              '--prefetch', str(plan['prefetch'] if plan['prefetch'] is not None else 2)]
    if training_root is None:
        with _argv(common + ['--mode', 'train', '--output', str(native_output)]):
            if train.main() != 0:
                raise RuntimeError('fresh PLDM native trainer failed; inspect native/status.json')
        mark_training_returned(plan, output)
    accepted_output = output / 'native_accept'
    with _argv(common + ['--mode', 'accept', '--training-output', str(native_output), '--output', str(accepted_output)]):
        if train.main() != 0:
            raise RuntimeError('fresh PLDM strict native acceptance failed')
    receipt = json.loads((accepted_output / 'status.json').read_text())
    checkpoint = Path(receipt['epoch_exports'][-1]['path'])
    return checkpoint, checkpoint.parent / 'config.json', 1 if phase == 'gate' else 10, receipt


def _tiny(plan, output, training_root=None):
    module = load_tiny_trainer()
    binding = fresh_data.load_binding(plan['data_manifest'])
    label = 'lejepa_fresh_cube' + str(plan['cube_count'])
    values = [module.__file__, '--task', label, '--dataset', binding['views']['train_rgb']['path'],
              '--name', label, '--epochs', str(plan['recipe']['epochs'])]
    if plan['workers'] is not None:
        values.extend(['--workers', str(plan['workers'])])
    original_loader = module.DataLoader
    if plan['prefetch'] is not None:
        def loader(*args, **kwargs):
            kwargs['prefetch_factor'] = plan['prefetch']
            return original_loader(*args, **kwargs)
        module.DataLoader = loader
    try:
        if training_root is None:
            with route_loader(module, plan), _argv(values):
                module.main()
            mark_training_returned(plan, output)
    finally:
        module.DataLoader = original_loader
    epochs = plan['recipe']['epochs']
    directory = Path(training_root or output) / 'native' / 'checkpoints' / label
    checkpoint = directory / f'weights_epoch_{epochs}.pt'
    if set(directory.glob('weights_epoch_*.pt')) != {directory / f'weights_epoch_{epoch}.pt' for epoch in range(1, epochs + 1)}:
        raise ValueError('canonical Tiny native epoch export set differs')
    import torch
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = module.Encoder()
    if set(weights) != {'backbone', 'head', 'proj'}:
        raise ValueError('canonical Tiny export keys differ')
    for key, state in weights.items():
        getattr(model, key).load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError('nonfinite Tiny export')
    return checkpoint, directory / 'config.json', epochs, dict(strict_factory_reload=True,
        optimizer_resume_available=False, heldout_validation=False, completed_native_epochs=epochs)


def _lewm(plan, output, training_root=None):
    from omegaconf import OmegaConf
    cfg = plan['config']
    with pinned_native() as native:
        if training_root is None:
            with route_loader(native, plan), _record_loaders(native, output):
                native.run(OmegaConf.create(cfg))
            mark_training_returned(plan, output)
        training_root = Path(training_root or output)
        epochs = cfg['trainer']['max_epochs']
        directory = training_root / 'native' / 'checkpoints' / cfg['output_model_name']
        checkpoint = directory / f'weights_epoch_{epochs}.pt'
        import torch
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = native.hydra.utils.instantiate(OmegaConf.create(cfg).model)
        model.load_state_dict(weights, strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
            raise ValueError('nonfinite LeWM export')
        full_paths = list((training_root / 'native' / 'checkpoints' / '_spt_cache').rglob('last.ckpt'))
        if len(full_paths) != 1:
            raise ValueError('one own full native LeWM last.ckpt required')
        full = torch.load(full_paths[0], map_location='cpu', weights_only=False)
        counts = json.loads((training_root / 'native_loader_counts.json').read_text())
        expected_steps = min(counts['train']['batches'], 2) if plan['mode'] == 'smoke' else counts['train']['batches'] * epochs
        if full.get('global_step') != expected_steps or full.get('epoch') not in (epochs - 1, epochs):
            raise ValueError('native LeWM epoch/update count differs')
        progress = full['loops']['fit_loop']['epoch_loop.val_loop.batch_progress']['total']['completed']
        expected_validation = min(counts['val']['batches'], 1) if plan['mode'] == 'smoke' else counts['val']['batches']
        if progress != expected_validation:
            raise ValueError('native LeWM final validation count differs')
        steps = [float(item['step']) for optimizer in full['optimizer_states']
                 for item in optimizer['state'].values() if 'step' in item]
        if not steps or set(steps) != {expected_steps}:
            raise ValueError('native LeWM Adam counters differ')
        def finite(value):
            if isinstance(value, torch.Tensor):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return all(finite(item) for item in value)
            return True
        if not finite(full['state_dict']) or not finite(full['optimizer_states']):
            raise ValueError('native LeWM model or optimizer is nonfinite')
        extracted = {key[6:]: value for key, value in full['state_dict'].items() if key.startswith('model.')}
        if extracted.keys() != weights.keys() or any(not torch.equal(extracted[key], weights[key]) for key in weights):
            raise ValueError('LeWM full checkpoint and native export differ')
    return checkpoint, directory / 'config.json', epochs, dict(strict_factory_reload=True,
        full_checkpoint=str(full_paths[0]), global_step=expected_steps, actual_loader_counts=counts,
        episode_heldout=True)


def _source_binding(method, cubes):
    package = Path(__file__).resolve().parent
    names = ['fresh_train.py','fresh_data.py','training_runtime.py','SOURCE_MANIFEST.json']
    names += ['pldm/'+name for name in ('train.py','shared_data.py','method.py','source_manifest.json')] if method=='pldm' else ['native/tiny_train.py'] if method=='lejepa' else ['native/lewm_train.py','recipes/'+method+'.json']
    result = {str(package/name): fresh_data.sha256(package/name) for name in names}
    if method.startswith('lewm'):
        result.update({str(code_root()/name): fresh_data.sha256(code_root()/name) for name in ('jepa.py','module.py','utils.py','encoders/vit.py','encoders/pointnet_dexart.py')})
    return result


def run_job(plan):
    """Run a fresh local native job and emit a receipt only after strict reload."""
    binding = fresh_data.load_binding(plan['data_manifest'])
    if fresh_data.sha256(plan['data_manifest']) != plan['data_manifest_sha256']:
        raise ValueError('fresh training manifest changed after planning')
    if plan.get('accept_only'):
        recovered = build_accept_plan(plan['training_output'], plan['output'])
        if recovered['original_plan_sha256'] != plan['original_plan_sha256']:
            raise ValueError('own training plan changed after acceptance planning')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='preflight', started_at=datetime.now(timezone.utc).isoformat(), actual_exit_code=None,
                 formal_acceptance_verified=False, plan=plan)
    fresh_data.save(output / 'plan.json', plan)
    fresh_data.save(output / 'status.json', state)
    if not plan.get('accept_only'):
        fresh_data.save(output / 'NATIVE_TRAINING.json', dict(stage='running', actual_exit_code=None,
            source_binding=_source_binding(plan['method'], plan['cube_count'])))
    previous_env = dict(os.environ)
    code = 1
    with (output / 'stdout_stderr.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                os.environ.update(build_environment(Path(plan['data_manifest']).parent, output / 'native'))
                import torch
                state['versions'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)
                for name in ('lightning', 'stable-pretraining', 'torchvision', 'hydra-core', 'omegaconf'):
                    state['versions'][name] = importlib.metadata.version(name)
                if not plan.get('accept_only'):
                    if not torch.cuda.is_available():
                        raise RuntimeError('original BF16 trainers require CUDA')
                    probe = torch.ones(2, 2, device='cuda')
                    if not torch.equal(probe @ probe, probe * 2):
                        raise RuntimeError('CUDA tensor preflight failed')
                state['stage'] = 'native_acceptance' if plan.get('accept_only') else 'native_training'
                fresh_data.save(output / 'status.json', state)
                runner = _pldm if plan['method'] == 'pldm' else _tiny if plan['method'] == 'lejepa' else _lewm
                checkpoint, config, epochs, acceptance = runner(plan, output, training_root=plan.get('training_output'))
                if acceptance.get('strict_factory_reload') is not True:
                    raise ValueError('own strict native checkpoint acceptance is required')
                fresh_data.load_binding(plan['data_manifest'])
                training_record = json.loads((Path(plan.get('training_output', output)) / 'NATIVE_TRAINING.json').read_text())
                if training_record['source_binding'] != _source_binding(plan['method'], plan['cube_count']):
                    raise ValueError('native training source changed before acceptance')
                receipt = dict(schema='cube_fresh_wm_acceptance_v1', method=plan['method'], model=plan['method'],
                    cube_count=plan['cube_count'], mode=plan['mode'], actual_exit_code=0, actual_epochs=epochs,
                    checkpoint=dict(path=str(checkpoint.resolve()), sha256=fresh_data.sha256(checkpoint)),
                    config=dict(path=str(config.resolve()), sha256=fresh_data.sha256(config)),
                    data_manifest=dict(path=plan['data_manifest'], sha256=plan['data_manifest_sha256']),
                    producer_binding=binding['producer_binding'], collection_protocol=binding['collection_protocol'],
                    source_binding=_source_binding(plan['method'], plan['cube_count']),
                    strict_reload_verified=True, formal_acceptance_verified=plan['mode'] == binding['mode'] == 'formal' and epochs == 10,
                    diagnostic=plan['mode'] == 'smoke', protocol=plan['protocol'], native_acceptance=acceptance,
                    training_budget=plan['recipe'], versions=state['versions'],
                    acceptance_only=bool(plan.get('accept_only')), training_output=plan.get('training_output', str(output)))
                fresh_data.save(output / 'WM_ACCEPTANCE.json', receipt)
                state.update(stage='accepted', actual_exit_code=0, completed_count=1,
                             formal_acceptance_verified=receipt['formal_acceptance_verified'])
                code = 0
            except BaseException:
                state.update(stage='failed', actual_exit_code=1, failed_count=1, error=traceback.format_exc())
                traceback.print_exc()
                marker_path = output / 'NATIVE_TRAINING.json'
                if not plan.get('accept_only') and json.loads(marker_path.read_text()).get('actual_exit_code') is None:
                    fresh_data.save(marker_path, dict(stage='failed', actual_exit_code=1, error=state['error']))
            finally:
                os.environ.clear()
                os.environ.update(previous_env)
    state['updated_at'] = state['finished_at'] = datetime.now(timezone.utc).isoformat()
    fresh_data.save(output / 'status.json', state)
    (output / 'actual_exit_code').write_text(str(code) + '\n', encoding='utf8')
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS + ('lewm2d', 'lewm3d'))
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('formal', 'smoke'))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prefetch', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-only', action='store_true', help='strict acceptance of a completed own native run; never retrain')
    parser.add_argument('--training-output', type=Path)
    args = parser.parse_args(argv)
    if args.accept_only:
        if args.training_output is None:
            parser.error('--accept-only requires --training-output')
        if args.method is not None or args.data_manifest is not None or args.workers is not None or args.prefetch is not None or args.mode is not None:
            parser.error('accept-only reads the original plan; do not supply training overrides')
        plan = build_accept_plan(args.training_output, args.output)
    else:
        if args.method is None or args.data_manifest is None:
            parser.error('--method and --data-manifest are required for training')
        plan = describe_training(args.method, args.data_manifest, args.output, phase=args.mode or 'formal',
                                 workers=args.workers, prefetch=args.prefetch)
    print(json.dumps(plan, indent=2))
    return run_job(plan) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
