# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/baselines/adroit/train_lejepa.py (verbatim except: imports / paths adapted to the package)
"""Final Relocate data/output adapter; original learning methods are delegated.

An explicit smoke/formal mode is required to execute; preparation is read-only.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


LEWM_ROOT = Path(__file__).resolve().parents[3]
NATIVE_SCRIPT = LEWM_ROOT / 'jepa4r/baselines/relocate/train_lejepa_relocate.py'
RECIPE = {'backbone': 'vit_hf_tiny_patch14', 'image_size': 224, 'views': 4,
          'seed': 3072,
          'batch_size': 256, 'epochs': 10, 'lr': 0.002, 'weight_decay': 0.05,
          'lamb': 0.02, 'training_projector_dim': 16, 'downstream_dim': 512,
          'precision': 'bf16', 'augmentation': 'unchanged native CPU transforms'}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='Fresh external STABLEWM_HOME; native checkpoints are under checkpoints/NAME')
    parser.add_argument('--task', choices=('adroit_relocate',), default='adroit_relocate')
    parser.add_argument('--name', default=None, help='Checkpoint name; default lejepa_TASK')
    parser.add_argument('--mode', choices=('smoke', 'formal'),
                        help='Explicit execution switch; omitted means preparation only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-steps', type=int, help='Positive native smoke limit; default2 in smoke mode')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, choices=(1, 2, 3), default=2)
    return parser


def build_plan(args, parser):
    if args.name is None:
        args.name = 'lejepa_' + args.task
    if not args.name or args.name in ('.', '..') or any(c in args.name for c in ('/', '\\')):
        parser.error('--name must be one checkpoint-directory component')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    if args.mode == 'smoke':
        steps = 2 if args.max_steps is None else args.max_steps
        if steps < 1:
            parser.error('smoke requires a positive --max-steps limit')
    elif args.max_steps not in (None, 0):
        parser.error('--max-steps is only valid in explicit smoke mode')
    dataset, output = args.dataset.expanduser().resolve(), args.output.expanduser().resolve()
    native = ['--task', args.task, '--dataset', str(dataset), '--name', args.name,
              '--epochs', '10', '--bs', '256', '--V', '4', '--lr', '0.002',
              '--lamb', '0.02', '--proj-dim', '16', '--workers', str(args.workers), '--seed', '3072']
    if args.mode == 'smoke':
        native += ['--max-steps', str(steps)]
    return {'execution': 'PREPARED_ONLY', 'method': 'repository_Tiny_LeJEPA',
            'task': args.task,
            'mode': args.mode, 'dataset': str(dataset), 'output': str(output),
            'native_script': str(NATIVE_SCRIPT),
            'native_source_sha256_lf': hashlib.sha256(NATIVE_SCRIPT.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            'native_argv': native, 'recipe': dict(RECIPE), 'workers': args.workers,
            'prefetch_factor': args.prefetch_factor,
            'data_boundary': 'native_all_visual_frames',
            'expected_visual_frames': 2000000,
            'source_row_schema': 'pre_action_rows_with_outgoing_action',
            'checkpoint_scope': 'native_smoke_does_not_save_a_checkpoint' if args.mode == 'smoke'
                                else 'native_epoch1_to10_exports_require_separate_acceptance'}


def loader_factory(original, workers, prefetch, close_parent):
    if workers < 0 or prefetch not in (1, 2, 3):
        raise ValueError('invalid worker/prefetch scheduling budget')
    def create(*args, **kwargs):
        dataset = args[0] if args else kwargs.get('dataset')
        close_parent(getattr(dataset, 'ds', dataset))
        kwargs = dict(kwargs, num_workers=workers, persistent_workers=workers > 0)
        if workers:
            kwargs['prefetch_factor'] = prefetch
        else:
            kwargs.pop('prefetch_factor', None)
        return original(*args, **kwargs)
    return create


def _load_native():
    if str(LEWM_ROOT) not in sys.path:
        sys.path.insert(0, str(LEWM_ROOT))
    module = importlib.import_module('jepa4r.baselines.relocate.train_lejepa_relocate')
    if Path(module.__file__).resolve() != NATIVE_SCRIPT.resolve():
        raise RuntimeError('LeJEPA entry resolved outside the selected checkout')
    return module


def _load_data_helpers():
    return importlib.import_module('jepa4r.baselines.relocate.data')


def _write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def execute(plan, *, gate=None):
    dataset, output = Path(plan['dataset']), Path(plan['output'])
    if output.exists():
        raise FileExistsError(f'fresh output required: {output}')
    if output.is_relative_to(LEWM_ROOT.resolve()):
        raise ValueError('outputs must remain outside protected source directories')
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Native LeJEPA requires CUDA; preparation does not authorize a CPU substitution')
    native, helpers = _load_native(), _load_data_helpers()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'launcher_plan.json', plan)
    state = {'phase': 'calling_native', 'mode': plan['mode'], 'started': time.time(),
             'exit_code': None, 'formal_acceptance_performed': False}
    _write(output / 'state.json', state)
    old_argv, old_home, old_loader = sys.argv, os.environ.get('STABLEWM_HOME'), native.DataLoader
    try:
        sys.argv = [str(NATIVE_SCRIPT), *plan['native_argv']]
        os.environ['STABLEWM_HOME'] = str(output)
        native.DataLoader = loader_factory(old_loader, plan['workers'], plan['prefetch_factor'],
                                          helpers.close_parent_hdf)
        with helpers.dataset_handle_hook(native.swm, dataset):
            if gate is None:
                native.main()
            else:
                native.main(gate=gate)
        state.update(phase='native_returned_acceptance_pending', exit_code=0, updated=time.time())
        _write(output / 'state.json', state)
        (output / 'actual_exit_code').write_text('0\n')
    except BaseException as error:
        code = (error.code or 0) if isinstance(error, SystemExit) else 130 if isinstance(error, KeyboardInterrupt) else 1
        if gate is not None:
            from jepa4r.policy.bc.relocate.recovery import RecoveryPause
            if isinstance(error, RecoveryPause):
                code = 75
        code = code if isinstance(code, int) else 1
        state.update(phase='paused_for_recovery' if code == 75 else 'failed' if code else 'native_returned_acceptance_pending',
                     exit_code=code, updated=time.time(), error=traceback.format_exc())
        _write(output / 'state.json', state)
        (output / 'actual_exit_code').write_text(str(code) + '\n')
        raise
    finally:
        sys.argv, native.DataLoader = old_argv, old_loader
        if old_home is None:
            os.environ.pop('STABLEWM_HOME', None)
        else:
            os.environ['STABLEWM_HOME'] = old_home


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = make_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    plan = build_plan(args, parser)
    if args.mode is None or args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    execute(plan)
    print(json.dumps({'native_main_returned': True, 'mode': args.mode,
                      'checkpoint_scope': plan['checkpoint_scope'], 'acceptance': 'not_performed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
