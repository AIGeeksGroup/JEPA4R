"""Task-only facade around the fresh CURRENT mainline eval.run; no CEM copy."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys


def validate_config(cfg):
    expected = {'seed': 42, 'eval.goal_offset_steps': 25, 'eval.eval_budget': 50, 'eval.num_eval': 50,
                'eval.img_size': 224, 'plan_config.horizon': 5, 'plan_config.receding_horizon': 5,
                'plan_config.action_block': 5, 'solver.num_samples': 300, 'solver.topk': 30,
                'solver.n_steps': 10, 'world.width': 256, 'world.height': 256}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--route', choices=('pix', 'pcd'), required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.code_root.resolve()
    if (root / 'lewm_pcd').is_dir():
        root = root / 'lewm_pcd'
    for path in (args.train, args.test, args.checkpoint):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', qualified=False, route=args.route,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  preprocessing_note='original World resize224 and original img_transform224 retained; native raw256',
                  fragment_criterion=dict(distance_lt=.075, axis_dot_gt=.95, drop_is_failure=True),
                  original_result=str(args.out / 'mainline_results.txt'))
    report_path = args.out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    original_get_dataset = None
    try:
        sys.path.insert(0, str(root))
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from envs.adroit.fragment import EpisodeAliases
        mainline = importlib.import_module('eval')
        with initialize_config_dir(version_base=None, config_dir=str(root / 'config/eval')):
            cfg = compose(config_name='adroit_pen_%s_fragment' % args.route)
        cfg.policy = str(args.checkpoint.resolve())
        cfg.dataset.stats = str(args.train.resolve())
        cfg.eval.dataset_name = str(args.test.resolve())
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        cfg.output.filename = str(args.out / 'mainline_results.txt')
        validate_config(OmegaConf.to_container(cfg, resolve=True))
        (args.out / 'effective_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        report['config_sha256'] = hashlib.sha256((args.out / 'effective_config.yaml').read_bytes()).hexdigest()
        sources = [Path(__file__), root/'eval.py', root/'jepa.py', root/'envs/adroit/fragment_env.py',
                   root/'envs/adroit/fragment.py', root/'envs/adroit/__init__.py',
                   root/'config/eval/adroit_pen_pix_fragment.yaml', root/'config/eval/adroit_pen_pcd_fragment.yaml']
        if args.route == 'pcd':
            sources.append(root/'envs/adroit/pcd.py')
        report['source_hashes'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        report['input_hashes'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (args.train, args.test, args.checkpoint)}
        for name in ('stable_worldmodel.world', 'stable_worldmodel.policy', 'stable_worldmodel.planning.solver.cem'):
            path = Path(importlib.import_module(name).__file__)
            report['source_hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        original_get_dataset = mainline.get_dataset

        def get_dataset(config, name):
            # CURRENT eval uses cfg.dataset.stats; original StandardScaler stays unchanged.
            reader = mainline.swm.data.HDF5Dataset(path=str(Path(name).resolve()),
                                                  keys_to_cache=config.dataset.keys_to_cache)
            count = 10000 if Path(name).resolve() == args.train.resolve() else 150
            if len(reader.lengths) != count:
                raise ValueError('expected %d episodes in %s' % (count, name))
            return EpisodeAliases(reader)

        mainline.get_dataset = get_dataset
        report['status'] = 'RUNNING'
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        mainline.run.__wrapped__(cfg)
        # Completion is not a successful controller score; the original metric is in results.txt.
        report['status'] = 'COMPLETED'
        report['exit_code'] = 0
    except BaseException as error:
        report.update(status='FAILED', error=type(error).__name__ + ': ' + str(error), exit_code=1)
        raise
    finally:
        if original_get_dataset is not None:
            mainline.get_dataset = original_get_dataset
        native_module = sys.modules.get('mj_envs.envs.hand_manipulation_suite.pen_v0')
        if native_module is not None:
            native_path = Path(native_module.__file__)
            report['native_source_path'] = str(native_path)
            report['native_source_sha256'] = hashlib.sha256(native_path.read_bytes()).hexdigest()
        else:
            report['native_source_sha256'] = 'UNVERIFIED'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        (args.out / 'exit_code').write_text(str(report.get('exit_code', 1)) + '\n')


if __name__ == '__main__':
    main()
