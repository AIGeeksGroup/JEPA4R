"""Task-only facade around the fresh CURRENT mainline eval.run; no CEM copy."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
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
    parser.add_argument('--out', type=Path, required=True)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.initials is None or (args.route == 'pix' and args.rgb_cache is None):
        parser.error('Relocate requires --initials, and pix also requires --rgb-cache')
    root = args.code_root.resolve()
    if (root / 'lewm_pcd').is_dir():
        root = root / 'lewm_pcd'
    for path in (args.train, args.test, args.checkpoint):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
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
            report['sampled_episodes'] = list(map(int, kwargs['episodes_idx']))
            report['sampled_start_steps'] = list(map(int, kwargs['start_steps']))
            result = original_evaluate(world, *positional, **kwargs)
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
