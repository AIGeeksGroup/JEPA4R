"""CPU-only guards for publishing Pen selectors through the existing BC entry."""
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'bench/cortexbench'
ENTRY = BENCH / 'run_bc.py'
ENTRY_SHA256_LF = '2d91779003dbf8fb283bdd0fbca08665f5c496d63d822817769b783e2e2fdae9'
MANIFEST = ROOT.parent / 'docs/results/adroit_pen_baselines_20261026.json'
MODELS = {'pldm_pen': ('load_pldm', 'checkpoint_dir', 192),
          'lejepa_pen': ('load_lejepa', 'checkpoint_dir', 512),
          'visreg_vitb': ('load_visreg', 'checkpoint', 768),
          'vjepa21_vitb': ('load_vjepa21', 'checkpoint', 768)}


@contextmanager
def isolated_entry(config='Adroit_BC_config', fresh=False, external=None, joined=False):
    """Run the real entry with only unavailable simulation/import boundaries faked."""
    package = ModuleType('mujoco_vc')
    package.__path__ = []
    gw, ml, loop = (ModuleType(name) for name in ('mujoco_vc.gym_wrapper', 'mujoco_vc.model_loading', 'train_loop'))
    calls, launches = [], []
    def original_sample(*args, **kwargs):
        paths = [object()]
        calls.append((args, kwargs, paths))
        return paths
    def original_load(*args, **kwargs):
        return ml.vc_models_dir_path, args, kwargs
    ml.vc_models_dir_path = 'original-vc-models'
    ml.load_pretrained_model = original_load
    loop.sample_paths = original_sample
    package.gym_wrapper, package.model_loading = gw, ml
    numpy = ModuleType('numpy')
    numpy.seterr = lambda **kwargs: None
    runner = ModuleType('runpy')
    runner.run_path = lambda path, **kwargs: launches.append((path, kwargs, list(sys.argv)))
    modules = {'mujoco_vc': package, 'mujoco_vc.gym_wrapper': gw, 'mujoco_vc.model_loading': ml,
               'train_loop': loop, 'numpy': numpy, 'runpy': runner}
    flags = ['--config-name=' + config] if joined else ['--config-name', config]
    argv = [str(ENTRY), *flags, 'env=pen-v0', 'seed=100', 'num_demos=100', 'eval_num_traj=25']
    with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)), patch.object(sys, 'argv', argv), patch.dict(os.environ):
        os.environ.pop('ADROIT_FRESH_EVAL', None)
        os.environ.pop('CORTEXBENCH_EAI_VC_ROOT', None)
        if fresh: os.environ['ADROIT_FRESH_EVAL'] = '1'
        if external is not None: os.environ['CORTEXBENCH_EAI_VC_ROOT'] = str(external)
        namespace = {'__file__': str(ENTRY), '__name__': '_pen_release_contract_entry'}
        exec(compile(ENTRY.read_text(encoding='utf-8'), str(ENTRY), 'exec'), namespace)
        yield namespace, calls, launches, ml, argv


class PenBaselineReleaseTests(unittest.TestCase):
    def release_manifest(self):
        self.assertTrue(MANIFEST.is_file(), 'the published result manifest must be included')
        return json.loads(MANIFEST.read_text(encoding='utf-8'))

    def test_existing_bc_entry_is_unchanged_from_release_base(self):
        source = ENTRY.read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(), ENTRY_SHA256_LF)

    def test_four_selectors_match_existing_loader_arguments_and_features(self):
        loader = BENCH / 'jepa_baselines.py'
        self.assertTrue(loader.is_file(), 'baseline loaders must be included')
        functions = {node.name: node for node in ast.parse(loader.read_text()).body if isinstance(node, ast.FunctionDef)}
        for selector, (name, checkpoint_arg, dimension) in MODELS.items():
            with self.subTest(selector=selector):
                path = BENCH / f'conf/model/{selector}.yaml'
                self.assertTrue(path.is_file(), f'{selector} selector must be included')
                values = dict(line.split(':', 1) for line in path.read_text().splitlines()
                              if line.strip() and not line.lstrip().startswith('#'))
                values = {key.strip(): value.strip() for key, value in values.items()}
                self.assertEqual(values['_target_'], 'jepa_baselines.' + name)
                node = functions[name]
                self.assertIn(checkpoint_arg, [arg.arg for arg in node.args.args])
                self.assertIn(checkpoint_arg, values)
                exported = [n for n in ast.walk(node) if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)]
                self.assertEqual(len(exported), 1)
                self.assertEqual(len(exported[0].value.elts), 4)
                if selector == 'pldm_pen':
                    self.assertEqual(ast.unparse(exported[0].value.elts[1]), 'int(dim)')
                else:
                    self.assertEqual(ast.literal_eval(exported[0].value.elts[1]), dimension)
                if checkpoint_arg == 'checkpoint_dir':
                    self.assertEqual(values['epoch'], '10')
                if selector == 'vjepa21_vitb':
                    self.assertEqual(values['resolution'], '384')

    def test_evaluation_seeds_and_native_launcher_arguments_are_forwarded(self):
        with isolated_entry() as (entry, calls, launches, ml, argv):
            env, policy = object(), object()
            for seed in (100, 200, 300):
                for _ in range(2):
                    result = entry['sample_paths']('original-positional', base_seed=seed,
                        env=env, policy=policy, num_traj=25, eval_mode=True, horizon=100)
                    self.assertIs(result, calls[-1][2])
            self.assertEqual(len(calls), 6)
            self.assertEqual([item[1]['base_seed'] for item in calls], [100, 100, 200, 200, 300, 300])
            self.assertTrue(all(item[0] == ('original-positional',) and item[1]['env'] is env and
                                item[1]['policy'] is policy and item[1]['num_traj'] == 25 for item in calls))
            self.assertEqual(launches[0][1], {'run_name': '__main__'})
            self.assertEqual(launches[0][2][1:], argv[1:])
            self.assertIs(entry['rollout_from_init_states'](), calls[-1][2])

    def test_54321_shortcut_and_explicit_adroit_fresh_guard_are_preserved(self):
        for config, fresh, expected_calls, joined in (
                ('Adroit_BC_config', False, 1, False), ('Adroit_BC_config', True, 2, True),
                ('Metaworld_BC_config', False, 1, True), ('Metaworld_BC_config', True, 1, False)):
            with self.subTest(config=config, fresh=fresh):
                with isolated_entry(config, fresh, joined=joined) as (entry, calls, *_):
                    entry['sample_paths'](base_seed=54321)
                    entry['sample_paths'](base_seed=54321)
                    self.assertEqual(len(calls), expected_calls)

    def test_external_eai_root_and_embedding_lookup_restore_original_paths(self):
        for external in (None, BENCH / 'external-eai-example'):
            with self.subTest(external=external):
                with isolated_entry(external=external) as (entry, calls, launches, ml, argv):
                    wanted = external if external is not None else ROOT / 'third_party/eai-vc'
                    self.assertEqual(Path(entry['VI']).resolve(), (wanted / 'cortexbench/mujoco_vc/visual_imitation').resolve())
                    for selector in MODELS:
                        root, args, kwargs = entry['load_pretrained_model'](selector, 'extra-positional', device='cpu')
                        self.assertEqual(root, str(BENCH))
                        self.assertEqual(args, (selector, 'extra-positional'))
                        self.assertEqual(kwargs, {'device': 'cpu'})
                        self.assertEqual(ml.vc_models_dir_path, 'original-vc-models')
                    root, _, _ = entry['load_pretrained_model']('external_vc_model')
                    self.assertEqual(root, 'original-vc-models')

    def test_published_data_and_bc_budget_keep_expected_and_observed_evidence_separate(self):
        data = self.release_manifest()
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(data['scope']['task_id'], 'pen-v0')
        pretraining = data['protocol']['local_pretraining']
        self.assertEqual(pretraining['training_episodes'], 10000)
        self.assertEqual(pretraining['training_visual_rows'], 914082)
        self.assertEqual(pretraining['independent_test_episodes_not_used_for_pretraining_or_validation'], 150)
        demos = data['external_data']['official_bc_demos']
        self.assertEqual([demos['demos'], demos['frames']], [100, 10000])
        protocol = data['protocol']['bc']
        self.assertEqual([protocol[k] for k in ('num_demos', 'epochs', 'batch_size', 'eval_trajectories')],
                         [100, 100, 256, 25])
        self.assertEqual(protocol['seeds'], [100, 200, 300])
        self.assertEqual(protocol['eval_epoch_indices'], list(range(5, 100, 5)) + [99])
        self.assertEqual(protocol['expected_adam_updates_per_seed'], 4000)
        self.assertIs(protocol['actual_optimizer_step_hooks_observed'], False)
        self.assertIs(protocol['runtime_encoder_before_after_hash_observed'], False)
        self.assertEqual(data['statistics']['ddof'], 1)

    def test_published_four_method_statistics_match_all_completed_seed_curves(self):
        data = self.release_manifest()
        self.assertEqual(set(data['methods']), {'pldm', 'lejepa', 'visreg', 'vjepa21'})
        selectors = {'pldm': 'pldm_pen', 'lejepa': 'lejepa_pen', 'visreg': 'visreg_vitb', 'vjepa21': 'vjepa21_vitb'}
        for name, selector in selectors.items():
            with self.subTest(method=name):
                method = data['methods'][name]
                self.assertEqual(method['embedding_selector'], selector)
                dimension = MODELS[selector][2]
                self.assertEqual(method['embedding_dim'], dimension)
                self.assertEqual(method['policy_input_dim'], 3 * dimension + 24)
                runs = method['runs']
                self.assertEqual([run['seed'] for run in runs], [100, 200, 300])
                for run in runs:
                    curve = run['eval_success_pct']
                    self.assertEqual(len(curve), 20)
                    self.assertTrue(all(math.isfinite(x) and 0 <= x <= 100 for x in curve))
                    self.assertEqual(run['best_of_20_success_pct'], max(curve))
                    self.assertEqual(run['final_success_pct'], curve[-1])
                    evidence = run['evidence']
                    self.assertEqual([evidence[k] for k in ('loss_epochs', 'evaluation_rounds', 'actual_child_exit_code')], [100, 20, 0])
                    self.assertEqual(evidence['saved_policy_CPU_reload_max_abs'], 0)
                summary = method['summary']
                for prefix, key in (('best_of_20', 'best_of_20_success_pct'), ('final', 'final_success_pct')):
                    values = [run[key] for run in runs]
                    self.assertEqual(summary[prefix + '_values_pct'], values)
                    self.assertAlmostEqual(summary[prefix + '_mean_pct'], statistics.mean(values), places=6)
                    self.assertAlmostEqual(summary[prefix + '_sample_sd_pp'], statistics.stdev(values), places=6)


if __name__ == '__main__':
    unittest.main()
