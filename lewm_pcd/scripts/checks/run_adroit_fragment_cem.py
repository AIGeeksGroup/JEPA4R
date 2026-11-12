"""Task-only facade around the fresh CURRENT mainline eval.run; no CEM copy."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import operator
from pathlib import Path
import sys


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def hdf_dependencies(path):
    # Import the owning pure-data helper without triggering environment registration.
    import importlib.util
    helper = Path(__file__).resolve().parents[2] / 'envs/adroit/hdf_binding.py'
    specification = importlib.util.spec_from_file_location('_relocate_hdf_binding', helper)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.hdf_dependencies(path)


def build_input_binding(train, test, checkpoint, initials, route, mode):
    """Bind reachable data and model contents independently of directory names."""
    dependencies = {'train': hdf_dependencies(train), 'test': hdf_dependencies(test)}
    configurations = {}
    for name in ('config.json', 'config.yaml'):
        path = Path(checkpoint).parent / name
        configurations[name] = file_sha(path) if path.is_file() else None
    content = dict(route=route, mode=mode,
                   train=dependencies['train']['content_sha256'],
                   test=dependencies['test']['content_sha256'],
                   checkpoint=file_sha(checkpoint), initials=file_sha(initials),
                   model_configurations=configurations)
    return dict(schema='relocate-cem-input-binding-v1', content=content,
                input_dependencies=dependencies)


def validate_input_binding(actual, expected):
    if (actual.get('schema') != 'relocate-cem-input-binding-v1'
            or expected.get('schema') != actual['schema']
            or expected.get('content') != actual['content']):
        raise ValueError('CEM input binding differs from the frozen data/model/route reference')
    return True


def validate_fragment_selection(episodes, start_steps, goal_offset, expected=None):
    def integer(value):
        if isinstance(value, bool):
            raise ValueError('CEM indices must be integers')
        try:
            return operator.index(value)
        except TypeError as error:
            raise ValueError('CEM indices must be integers') from error
    selected = dict(episodes=list(map(integer, episodes)), start_steps=list(map(integer, start_steps)),
                    goal_offset=integer(goal_offset))
    if (len(selected['episodes']) != len(selected['start_steps'])
            or any(value < 0 for value in selected['episodes'] + selected['start_steps'])
            or selected['goal_offset'] != 25):
        raise ValueError('invalid CEM source episode/start/future-goal selection')
    if expected is not None and selected != expected:
        raise ValueError('CEM source episode/start/future-goal selection differs from reference')
    return selected


class FullCEMTrace:
    """Passive recorder for the original solver's existing callback interface."""
    output_key = 'guide_evidence'

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.history, self.records = {}, []
        self.solve_index = 0

    def reset(self):
        self.solve_index += 1
        self.batch_index = -1
        self.history = {}

    def start_batch(self):
        self.batch_index += 1

    def __call__(self, *, step, **state):
        import numpy as np
        names = ('candidates', 'costs', 'topk_vals', 'topk_inds', 'topk_candidates',
                 'mean', 'var', 'prev_mean', 'prev_var')
        arrays = {}
        for name in names:
            value = state[name]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                raise ValueError('CEM trace requires finite numeric arrays: ' + name)
            arrays[name] = value.copy()
        arrays['first_action_block'] = arrays['mean'][:, 0].copy()
        name = f'solve_{self.solve_index:03d}_batch_{self.batch_index:03d}_step_{step:03d}.npz'
        path = self.directory / name
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        self.records.append(dict(solve=self.solve_index, batch=self.batch_index, step=step,
                                 arrays_file=name, arrays_sha256=file_sha(path),
                                 shapes={key: list(value.shape) for key, value in arrays.items()}))

    def end_solve(self):
        rows = [row for row in self.records if row['solve'] == self.solve_index]
        if not rows:
            raise ValueError('CEM solve produced no candidate evidence')
        with (self.directory / f'solve_{self.solve_index:03d}.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(producer_sha256=file_sha(__file__), records=rows), stream, indent=2)


class ModelTrace:
    """Read-only hooks on the actual solver cost, including both projections."""
    roles = ('encoder', 'projector', 'action_encoder', 'predictor', 'pred_proj')
    def __init__(self, model, directory):
        import inspect
        self.model, self.directory = model, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('CEM model must already be frozen and in eval mode')
        self.before = self.state_digest()
        self.records, self.handles, self.sources = [], [], {}
        modules = [(role, getattr(model, role)) for role in self.roles]
        for role, module in modules:
            path = Path(inspect.getfile(type(module))).resolve()
            self.sources[role] = dict(module=type(module).__module__+'.'+type(module).__name__,
                                      path=str(path), sha256=file_sha(path))
        for role, module in modules:
            self.handles.append(module.register_forward_hook(self.hook(role)))

    def state_digest(self):
        import torch
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def hook(self, role):
        def capture(module, inputs, output):
            from collections.abc import Mapping
            import numpy as np
            import torch
            arrays, dtypes = {}, {}
            def visit(value, key):
                if torch.is_tensor(value):
                    dtypes[key] = str(value.dtype)
                    value = value.detach().cpu()
                    if value.dtype == torch.bfloat16:
                        value = value.float()
                    array = value.numpy().copy()
                    if not np.isfinite(array).all():
                        raise ValueError('nonfinite actual CEM model forward: '+role+'/'+key)
                    arrays[key] = array
                elif isinstance(value, Mapping):
                    for name, child in value.items():
                        visit(child, key+'/'+str(name))
                elif isinstance(value, (list, tuple)):
                    for index, child in enumerate(value):
                        visit(child, key+'/'+str(index))
            visit(inputs, 'input')
            visit(output, 'output')
            if not any(key.startswith('output') for key in arrays):
                raise ValueError('actual CEM model hook has no output tensors')
            name = f'{len(self.records):04d}_{role}.npz'
            with (self.directory/name).open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            self.records.append(dict(role=role, arrays_file=name,
                                     arrays_sha256=file_sha(self.directory/name),
                                     torch_dtypes=dtypes,
                                     shapes={k:list(v.shape) for k,v in arrays.items()}))
        return capture

    def close(self, completed):
        for handle in self.handles:
            handle.remove()
        after = self.state_digest()
        self.summary = dict(completed=completed, producer_sha256=file_sha(__file__),
                            frozen_before_sha256=self.before, frozen_after_sha256=after,
                            frozen_unchanged=self.before == after, actual_sources=self.sources,
                            calls={role:sum(row['role'] == role for row in self.records)
                                   for role in self.roles},
                            records=self.records)
        with (self.directory/'report.json').open('x', encoding='utf-8') as stream:
            json.dump(self.summary, stream, indent=2)
        if completed and (self.before != after or any(count == 0 for count in self.summary['calls'].values())):
            raise ValueError('actual CEM model usage or frozen-state contract failed')


@contextmanager
def observe_model(model, directory):
    trace = ModelTrace(model, directory)
    completed = False
    try:
        yield trace
        completed = True
    finally:
        trace.close(completed)


def validate_model_trace_records(actual, reference):
    """Compare original Torch types as well as stored, NumPy-compatible arrays."""
    if not actual or len(actual) != len(reference):
        raise ValueError('CEM model trace event counts differ')
    for left, right in zip(actual, reference):
        for field in ('role', 'torch_dtypes', 'shapes'):
            if not left.get(field) or not right.get(field) or left[field] != right[field]:
                raise ValueError('CEM model trace '+field+' differs')
    return True


class RolloutTrace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = []

    def capture(self, world, kind, mask=None, raw_step_infos=None):
        import numpy as np
        names = ('qpos', 'qvel', 'proprio', 'native_observation', 'target_pos',
                 'fragment_goal_position', 'fragment_distance', 'fragment_success',
                 'fragment_failed', 'native_goal_achieved', 'native_done',
                 'executed_action', 'action_clip_max_abs', 'time', 'seed', 'step_idx',
                 'pixels', 'pcd_xyz', 'pcd_rgb', 'goal', 'goal_pixels', 'goal_pcd_xyz', 'goal_pcd_rgb',
                 'goal_native_observation', 'goal_observation', 'goal_qpos', 'goal_qvel')
        arrays = {}
        for name in names:
            if name in world.infos:
                value = world.infos[name]
            elif raw_step_infos and all(name in info for info in raw_step_infos.values()):
                # EnvPool deliberately retains only the reset-time keys. Read
                # step-only native evidence from the actual returned infos;
                # never add keys to the planner's live buffered observations.
                value = np.stack([raw_step_infos[i][name] for i in sorted(raw_step_infos)])[:, None]
            else:
                continue
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                raise ValueError('nonfinite/nonnumeric actual rollout field: ' + name)
            arrays[name] = value.copy()
        if not {'qpos', 'qvel'} <= set(arrays) or kind == 'step' and 'executed_action' not in arrays:
            raise ValueError('actual CEM rollout state/executed action is missing')
        if kind == 'step':
            for name in ('rewards', 'terminateds', 'truncateds'):
                value = getattr(world, name, None)
                if value is not None:
                    arrays[name] = np.asarray(value).copy()
        if mask is not None:
            arrays['active_mask'] = np.asarray(mask).copy()
        if any(value.dtype.kind not in 'biuf' or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('nonfinite/nonnumeric actual rollout state/action/event')
        name = f'{len(self.records):04d}_{kind}.npz'
        with (self.directory / name).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        self.records.append(dict(kind=kind, arrays_file=name, arrays_sha256=file_sha(self.directory / name)))


@contextmanager
def observe_rollout(world, directory):
    """Observe the existing reset/on_step interfaces; retain the original loop."""
    trace = RolloutTrace(directory)
    original_reset, original_run = world.reset, world._run
    pending, original_steps = {}, []
    bases = [env.unwrapped for env in getattr(getattr(world, 'envs', None), 'envs', [])]
    if len({id(base) for base in bases}) != len(bases):
        raise ValueError('CEM trace requires distinct native env instances')
    for index, base in enumerate(bases):
        original = base.step
        original_steps.append((base, original))
        def step(action, _original=original, _index=index):
            result = _original(action)
            pending[_index] = result[-1]
            return result
        base.step = step
    def reset(*args, **kwargs):
        pending.clear()
        result = original_reset(*args, **kwargs)
        trace.capture(world, 'initial')
        return result
    def run(*args, **kwargs):
        # Dataset-driven World.evaluate injects goals after reset, before _run.
        trace.capture(world, 'planner_initial')
        previous = kwargs.get('on_step')
        def on_step(current, mask):
            if previous is not None:
                previous(current, mask)
            kind = 'step' if pending or not bases else 'reset'
            trace.capture(current, kind, mask, raw_step_infos=pending)
            pending.clear()
        return original_run(*args, **dict(kwargs, on_step=on_step))
    world.reset, world._run = reset, run
    try:
        yield trace
    finally:
        world.reset, world._run = original_reset, original_run
        for base, original in original_steps:
            base.step = original


def validate_rgb_cache(cache, test, initials, mode):
    """Accept only a completed producer bound to these logical source bytes."""
    report_path = Path(cache).parent / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'COMPLETE' or report.get('exit_code') != 0:
        raise ValueError('RGB cache producer did not complete with exit0')
    if report.get('mode') != mode:
        raise ValueError('RGB cache producer mode differs from CEM mode')
    if file_sha(cache) != report.get('cache_sha256'):
        raise ValueError('RGB cache SHA differs from the completed producer')
    inputs = set(report.get('inputs', {}).values())
    bound = hdf_dependencies(test)
    if bound['root_sha256'] not in inputs:
        raise ValueError('RGB cache is bound to another test dataset')
    initial_sha = file_sha(initials)
    if initial_sha not in inputs:
        raise ValueError('RGB cache is bound to other initial states')
    expected = report.get('input_dependencies', {}).get('test', {})
    if expected.get('schema') != bound['schema'] or expected.get('content_sha256') != bound['content_sha256']:
        raise ValueError('RGB cache logical test dependencies/backing differ')
    return dict(passed=True, producer_report=str(report_path.resolve()), producer_report_sha256=file_sha(report_path),
                test_sha256=bound['root_sha256'], initials_sha256=initial_sha,
                logical_test_content_sha256=bound['content_sha256'], dependency_files=bound['file_count'])


def budget(mode):
    if mode == 'full':
        return dict(num_eval=50, eval_budget=50, num_samples=300, topk=30,
                    n_steps=10, train_episodes=10000, test_episodes=150)
    if mode == 'smoke':
        # One complete25-action planning horizon; only the search budget is reduced.
        return dict(num_eval=1, eval_budget=25, num_samples=30, topk=5,
                    n_steps=1, train_episodes=2, test_episodes=1)
    raise ValueError('CEM mode must be full or smoke')


def apply_budget(cfg, mode):
    selected = budget(mode)
    for section, keys in (('eval', ('num_eval', 'eval_budget')),
                          ('solver', ('num_samples', 'topk', 'n_steps'))):
        for key in keys:
            cfg[section][key] = selected[key]


def validate_config(cfg, mode='full'):
    selected = budget(mode)
    expected = {'seed': 42, 'eval.goal_offset_steps': 25,
                'eval.eval_budget': selected['eval_budget'], 'eval.num_eval': selected['num_eval'],
                'eval.img_size': 224, 'plan_config.horizon': 5, 'plan_config.receding_horizon': 5,
                'plan_config.action_block': 5, 'solver.num_samples': selected['num_samples'],
                'solver.topk': selected['topk'], 'solver.n_steps': selected['n_steps'],
                'world.width': 256, 'world.height': 256,
                'world.env_name': 'envs.adroit:swm/AdroitRelocateFragment-v0'}
    for key, value in expected.items():
        actual = cfg
        for part in key.split('.'):
            actual = actual[part]
        if actual != value:
            raise ValueError('mainline protocol %s must equal %s' % (key, value))
    if cfg['plan_config'].get('history_len', 1) != 1:
        raise ValueError('mainline history_len must retain its default one')
    train, test = Path(cfg['dataset']['stats']), Path(cfg['eval']['dataset_name'])
    if not train.is_absolute() or not test.is_absolute() or train.resolve() == test.resolve():
        raise ValueError('train normalization and held-out test paths must be distinct absolute paths')
    if not Path(cfg['policy']).is_absolute():
        raise ValueError('checkpoint must be an explicit absolute path')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--route', choices=('pix', 'pcd'), required=True)
    parser.add_argument('--task', choices=('relocate-v0',), default='relocate-v0')
    parser.add_argument('--mode', choices=('full', 'smoke'), default='full',
                        help='Full retains the paper protocol; smoke is a one-fragment diagnostic')
    parser.add_argument('--initials', type=Path, help='Relocate exact native initial states NPZ')
    parser.add_argument('--rgb-cache', type=Path, help='Relocate sparse exact replay RGB HDF5')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-binding', type=Path,
                        help='Frozen content binding and optional fragments selection; checked before planning')
    parser.add_argument('--gate-trace', action='store_true',
                        help='Smoke only: passively retain full solver candidates/costs/elites/action blocks')
    parser.add_argument('--out', type=Path, required=True)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.gate_trace and args.mode != 'smoke':
        parser.error('--gate-trace requires the bounded smoke mode')
    if args.initials is None or (args.route == 'pix' and args.rgb_cache is None):
        parser.error('Relocate requires --initials, and pix also requires --rgb-cache')
    root = args.code_root.resolve()
    if (root / 'lewm_pcd').is_dir():
        root = root / 'lewm_pcd'
    for path in (args.train, args.test, args.checkpoint):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    input_binding = build_input_binding(args.train, args.test, args.checkpoint,
                                        args.initials, args.route, args.mode)
    expected_binding = (json.loads(args.expected_binding.read_text(encoding='utf-8'))
                        if args.expected_binding is not None else None)
    if expected_binding is not None:
        validate_input_binding(input_binding, expected_binding)
    rgb_binding = validate_rgb_cache(args.rgb_cache, args.test, args.initials, args.mode) if args.route == 'pix' else None
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', qualified=False, route=args.route, task=args.task,
                  mode=args.mode, diagnostic=args.mode == 'smoke', budget=budget(args.mode),
                  scope=('bounded interface diagnostic; not the paper CEM result' if args.mode == 'smoke'
                         else 'paired50 future-object-position fragments; independent from BC'),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  preprocessing_note='original World resize224 and original img_transform224 retained; native raw256',
                  fragment_criterion=dict(distance_lt=.1,
                                      scope='future-object-position fragment; not native >25-step task success'),
                  original_result=str(args.out / 'mainline_results.txt'))
    report['input_binding'] = input_binding
    if args.expected_binding is not None:
        report['expected_binding_sha256'] = file_sha(args.expected_binding)
    if rgb_binding is not None:
        report['rgb_cache_binding'] = rgb_binding
    report_path = args.out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    original_get_dataset = None
    original_evaluate = None
    try:
        sys.path.insert(0, str(root))
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from envs.adroit.fragment import EpisodeAliases
        mainline = importlib.import_module('eval')
        with initialize_config_dir(version_base=None, config_dir=str(root / 'config/eval')):
            cfg = compose(config_name='adroit_%s_%s_fragment' % (args.task[:-3], args.route))
        cfg.policy = str(args.checkpoint.resolve())
        cfg.dataset.stats = str(args.train.resolve())
        cfg.eval.dataset_name = str(args.test.resolve())
        apply_budget(cfg, args.mode)
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        cfg.output.filename = str(args.out / 'mainline_results.txt')
        validate_config(OmegaConf.to_container(cfg, resolve=True), args.mode)
        (args.out / 'effective_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        report['config_sha256'] = file_sha(args.out / 'effective_config.yaml')
        sources = [Path(__file__), root/'eval.py', root/'jepa.py', root/'envs/adroit/fragment_env.py',
                   root/'envs/adroit/fragment.py', root/'envs/adroit/__init__.py',
                   root/'envs/adroit/hdf_binding.py',
                   root/'config/eval/adroit_relocate_pix_fragment.yaml',
                   root/'config/eval/adroit_relocate_pcd_fragment.yaml']
        if args.route == 'pcd':
            sources.append(root/'envs/adroit/pcd.py')
        report['source_hashes'] = {str(path): file_sha(path) for path in sources}
        report['input_hashes'] = {str(path.resolve()): file_sha(path)
                                 for path in (args.train, args.test, args.checkpoint)}
        for path in (args.initials, args.rgb_cache, args.checkpoint.parent/'config.json',
                     args.checkpoint.parent/'config.yaml'):
            if path is not None:
                report['input_hashes'][str(path.resolve())] = file_sha(path)
        for name in ('stable_worldmodel.world', 'stable_worldmodel.policy', 'stable_worldmodel.planning.solver.cem'):
            path = Path(importlib.import_module(name).__file__)
            report['source_hashes'][str(path)] = file_sha(path)
        original_get_dataset = mainline.get_dataset

        def get_dataset(config, name):
            # CURRENT eval uses cfg.dataset.stats; original StandardScaler stays unchanged.
            is_train = Path(name).resolve() == args.train.resolve()
            # EpisodeAliases derives ep_idx; it need not exist in fresh native HDF.
            keys = ['action'] if is_train else ['action', 'seed', 'step_idx',
                        'qpos', 'qvel', 'observation', 'initial_state_sha256']
            if not is_train and args.route == 'pcd':
                keys += ['pcd_xyz', 'pcd_rgb']
            reader = mainline.swm.data.HDF5Dataset(path=str(Path(name).resolve()),
                          keys_to_load=keys, keys_to_cache=config.dataset.keys_to_cache)
            count = budget(args.mode)['train_episodes' if is_train else 'test_episodes']
            allowed = (count,) if args.mode == 'full' else (count, 10000 if is_train else 150)
            if len(reader.lengths) not in allowed:
                raise ValueError('expected episode count in %s for %s' % (allowed, name))
            report.setdefault('data_episode_counts', {})['train' if is_train else 'test'] = len(reader.lengths)
            return reader if is_train else EpisodeAliases(reader, task=args.task,
                              initial_states=args.initials, rgb_frames=args.rgb_cache)

        mainline.get_dataset = get_dataset
        original_evaluate = mainline.swm.World.evaluate
        def evaluate(world, *positional, **kwargs):
            report['source_selection'] = validate_fragment_selection(
                kwargs['episodes_idx'], kwargs['start_steps'], kwargs['goal_offset'],
                expected_binding.get('fragments') if expected_binding is not None else None)
            report['sampled_episodes'] = list(map(int, kwargs['episodes_idx']))
            report['sampled_start_steps'] = list(map(int, kwargs['start_steps']))
            trace = FullCEMTrace(args.out / 'cem_trace') if args.gate_trace else None
            rollout = forward_trace = None
            if trace is not None:
                world.policy.solver.callbacks.append(trace)
            try:
                with (observe_model(world.policy.solver.cost, args.out / 'model_trace')
                      if trace is not None else nullcontext()) as forward_trace, \
                     (observe_rollout(world, args.out / 'rollout_trace')
                      if trace is not None else nullcontext()) as rollout:
                    result = original_evaluate(world, *positional, **kwargs)
            finally:
                if trace is not None:
                    world.policy.solver.callbacks.remove(trace)
                    report['cem_trace'] = trace.records
                    if forward_trace is not None:
                        report['model_trace'] = forward_trace.summary
                    if rollout is not None:
                        report['rollout_trace'] = rollout.records
                        report['actual_control_steps'] = sum(row['kind'] == 'step' for row in rollout.records)
            report['metrics'] = {key: (value.tolist() if hasattr(value, 'tolist') else value)
                                 for key, value in result.items()}
            return result
        mainline.swm.World.evaluate = evaluate
        report['status'] = 'RUNNING'
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        mainline.run.__wrapped__(cfg)
        if len(report.get('sampled_episodes', [])) != budget(args.mode)['num_eval']:
            raise RuntimeError('actual fragment count differs from the selected CEM budget')
        if not Path(report['original_result']).is_file() or not report.get('metrics'):
            raise RuntimeError('mainline CEM did not produce its actual result and metrics')
        # Completion is not a successful controller score; the original metric is in results.txt.
        report['status'] = 'COMPLETED'
        report['exit_code'] = 0
    except BaseException as error:
        report.update(status='FAILED', error=type(error).__name__ + ': ' + str(error),
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
        raise
    finally:
        if original_get_dataset is not None:
            mainline.get_dataset = original_get_dataset
        if original_evaluate is not None:
            mainline.swm.World.evaluate = original_evaluate
        native_module = sys.modules.get('mj_envs.envs.hand_manipulation_suite.' + args.task.replace('-v0', '_v0'))
        if native_module is not None:
            native_path = Path(native_module.__file__)
            report['native_source_path'] = str(native_path)
            report['native_source_sha256'] = file_sha(native_path)
        else:
            report['native_source_sha256'] = 'UNVERIFIED'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        (args.out / 'exit_code').write_text(str(report.get('exit_code', 1)) + '\n')


if __name__ == '__main__':
    main()
