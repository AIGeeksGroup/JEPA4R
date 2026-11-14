"""Final Relocate data/output adapter; original learning methods are delegated.

An explicit smoke/formal mode is required to execute; preparation is read-only.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types


LEWM_ROOT = Path(__file__).resolve().parents[3]
RECIPE = {'backbone': 'vit_hf_tiny_patch14', 'image_size': 224,
          'seed': 3072,
          'selected_frames': 4, 'frameskip': 5, 'raw_span': 20,
          'raw_action_dim': 30, 'action_block_dim': 150, 'latent_dim': 192,
          'history_size': 3, 'prediction_steps': 1, 'batch_size': 128,
          'epochs': 10, 'precision': 'bf16', 'optimizer': 'AdamW',
          'lr': 5e-5, 'weight_decay': 1e-3,
          'loss': 'native pred_MSE +18std +0.7std_t +12cov +0.2temp_align',
          'normalizer': 'original column_normalizer/ZScoreScaler: NaN rows excluded, population std'}


def sha_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework-root', required=True, type=Path,
                        help='External stable-worldmodel checkout at the recorded reference commit')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='Fresh external STABLEWM_HOME, never an existing run')
    parser.add_argument('--task', choices=('adroit_relocate',), default='adroit_relocate')
    parser.add_argument('--name', default=None, help='Checkpoint name; default pldm_TASK')
    parser.add_argument('--mode', choices=('smoke', 'formal'),
                        help='Explicit execution switch; omission prepares only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-steps', type=int, help='Smoke-only actual update limit; default2')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prefetch-factor', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--device', choices=('cuda',), default='cuda',
                        help='Native BF16 recipe; no silent CPU-training substitute')
    return parser


def build_plan(args, parser):
    if args.name is None:
        args.name = 'pldm_' + args.task
    if not args.name or args.name in ('.', '..') or any(c in args.name for c in ('/', '\\')):
        parser.error('--name must be one checkpoint-directory component')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    if args.mode == 'smoke':
        steps = 2 if args.max_steps is None else args.max_steps
        if steps < 1:
            parser.error('smoke requires a positive actual-step limit')
    elif args.max_steps not in (None, 0):
        parser.error('--max-steps is valid only for explicit smoke mode')
    framework = args.framework_root.expanduser().resolve()
    script = framework / 'scripts/train/pldm.py'
    config = framework / 'scripts/train/config/pldm.yaml'
    if not script.is_file() or not config.is_file():
        parser.error('--framework-root must contain the original PLDM script and configuration')
    dataset, output = args.dataset.expanduser().resolve(), args.output.expanduser().resolve()
    overrides = ['data=ogb', 'seed=3072', f'data.dataset.name={json.dumps(str(dataset))}',
                 'data.dataset.keys_to_load=[pixels,action]', 'data.dataset.keys_to_cache=[action]',
                 '~data.dataset.keys_to_merge', f'subdir={json.dumps(args.name)}',
                 f'output_model_name={json.dumps(args.name)}',
                 'trainer.max_epochs=10', f'num_workers={args.workers}',
                 'loader.persistent_workers=' + str(args.workers > 0).lower()]
    overrides += ([f'loader.prefetch_factor={args.prefetch_factor}'] if args.workers
                  else ['~loader.prefetch_factor'])
    if args.mode == 'smoke':
        overrides += [f'+trainer.max_steps={steps}', '+trainer.limit_train_batches=2',
                      '+trainer.limit_val_batches=2']
    action_dim = 30
    recipe = dict(RECIPE, raw_action_dim=action_dim,
                  action_block_dim=action_dim * RECIPE['frameskip'])
    return {
        'execution': 'PREPARED_ONLY',
        'method': 'framework_PLDM',
        'mode': args.mode,
        'task': args.task,
        'expected_visual_frames': 2000000,
        'source_row_schema': 'pre_action_rows_with_outgoing_action',
        'framework_root': str(framework),
        'native_script': str(script),
        'native_source_sha256_lf': sha_lf(script),
        'native_config_sha256_lf': sha_lf(config),
        'dataset': str(dataset),
        'output': str(output),
        'name': args.name,
        'device': args.device,
        'hydra_overrides': overrides,
        'recipe': recipe,
        'workers': args.workers,
        'prefetch_factor': args.prefetch_factor,
        'data_boundary': 'native sequence sampling and all windows preserved; original normalizer excludes NaN rows '
                         'and forward applies nan_to_num',
        'compatibility': 'isolated-memory logger scope fix only; framework source stays unchanged',
        'acceptance': 'pending real boundary-window GPU smoke and full-run checkpoint/gradient acceptance',
    }


def logger_compatible_tree(source, filename):
    """Apply the repository's logger-scope fix to a private module AST, not disk."""
    tree = ast.parse(source, filename=filename)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run'), None)
    if function is None:
        raise ValueError('framework entry has no original run function')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'info'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'logger']
    if len(calls) > 1:
        raise ValueError('logger scope compatibility must select at most one original call')
    if calls:
        calls[0].func = ast.parse('logging.getLogger(__name__).info', mode='eval').body
    return ast.fix_missing_locations(tree), {
        'applied': bool(calls), 'replacement': 'logger.info -> logging.getLogger(__name__).info',
        'disk_source_changed': False, 'checkpoint_interval_changed': False,
        'scope': 'private module AST only; native forward/model/loss/optimizers untouched'}


def load_native(path):
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    tree, receipt = logger_compatible_tree(source, str(path))
    name = '_relocate_framework_pldm_' + sha_lf(path)[:12]
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), ''
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not Path(module.swm.__file__).resolve().is_relative_to(path.parents[2].resolve()):
        raise RuntimeError('stable_worldmodel resolved outside the selected framework checkout')
    return module, receipt


def _write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def explicit_training_seed(native, seed):
    import lightning as pl
    seed=int(seed)
    pl.seed_everything(seed,workers=True)
    original=native.spt.Manager
    def manager(*args,**kwargs):
        kwargs['seed']=seed
        return original(*args,**kwargs)
    native.spt.Manager=manager
    try:yield
    finally:native.spt.Manager=original


def execute(plan):
    if plan['mode'] not in ('smoke', 'formal'):
        raise ValueError('an explicit smoke/formal mode is required for execution')
    framework, dataset, output = map(Path, (plan['framework_root'], plan['dataset'], plan['output']))
    if output.exists():
        raise FileExistsError(f'fresh output required: {output}')
    if output.is_relative_to(LEWM_ROOT.resolve()) or output.is_relative_to(framework.resolve()):
        raise ValueError('output must remain outside both protected source directories')
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    torch = importlib.import_module('torch')
    if not torch.cuda.is_available():
        raise RuntimeError('PLDM execution requires CUDA; preparation does not authorize CPU training')
    if sha_lf(plan['native_script']) != plan['native_source_sha256_lf']:
        raise RuntimeError('native PLDM source changed after preparation')
    previous_path = list(sys.path)
    previous_env = {key: os.environ.get(key) for key in ('STABLEWM_HOME', 'SPT_CACHE_DIR')}
    sys.path[:0] = [str(LEWM_ROOT), str(framework)]
    try:
        native, patch_receipt = load_native(plan['native_script'])
        hydra = importlib.import_module('hydra')
        helpers = importlib.import_module('jepa4r.baselines.relocate.data')
        with hydra.initialize_config_dir(version_base=None,
                                         config_dir=str(framework / 'scripts/train/config')):
            cfg = hydra.compose(config_name='pldm', overrides=plan['hydra_overrides'])
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'launcher_plan.json', plan)
        _write(output / 'compatibility_patch.json', patch_receipt)
        state = {'phase': 'calling_native', 'mode': plan['mode'], 'started': time.time(),
                 'exit_code': None, 'formal_acceptance_performed': False, 'data_receipts': []}
        _write(output / 'state.json', state)
        try:
            os.environ.update(STABLEWM_HOME=str(output), SPT_CACHE_DIR=str(output / '_spt_cache'))
            native.spt.set(cache_dir=str(output / '_spt_cache'),
                           cleanup={'checkpoints': False, 'logs': False, 'callback_artifacts': False})
            def record(receipt):
                state['data_receipts'].append(receipt)
                _write(output / 'state.json', state)

            with (
                helpers.dataset_handle_hook(native.swm, dataset, recorder=record),
                explicit_training_seed(native, int(cfg.seed)),
            ):
                native.run.__wrapped__(cfg)
            state.update(phase='native_returned_acceptance_pending', exit_code=0, updated=time.time())
            _write(output / 'state.json', state)
            (output / 'actual_exit_code').write_text('0\n')
        except BaseException as error:
            code = (
                (error.code or 0)
                if isinstance(error, SystemExit)
                else 130
                if isinstance(error, KeyboardInterrupt)
                else 1
            )
            code = code if isinstance(code, int) else 1
            state.update(phase='failed' if code else 'native_returned_acceptance_pending',
                         exit_code=code, updated=time.time(), error=traceback.format_exc())
            _write(output / 'state.json', state)
            (output / 'actual_exit_code').write_text(str(code) + '\n')
            raise
    finally:
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
                      'acceptance': 'not_performed; real checkpoint/gradient verification remains pending'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
