"""Narrow CPU checks; fake model only, no training or neural-network update."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).parent
FORMAL = HERE.parent / 'official_state_formal_v1'
sys.path.insert(0, str(FORMAL))
spec = importlib.util.spec_from_file_location('_last3_test_fixture', FORMAL / 'test_formal_main.py')
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)

from evaluate_last3 import LastThreeBridge, SUBSET_PROOF, STEPS, formal


class LastThreeContract(unittest.TestCase):
    def test_only_expected_node_selection_changes_in_frozen_method(self):
        self.assertEqual(SUBSET_PROOF['changed_AST_sites'], 1)
        self.assertTrue(SUBSET_PROOF['all_other_method_AST_equal'])
        self.assertEqual(SUBSET_PROOF['selected_checkpoint_steps'], [800000, 900000, 1000000])

    def test_three_restores_original_eval_body_no_update_new_csv_only(self):
        events = []
        original_randint = np.random.randint

        def linux_randint(low, high=None, *args, **kwargs):
            if high == 2**32:
                kwargs['dtype'] = np.int64
            return original_randint(low, high, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            model, out = Path(folder) / 'model', Path(folder) / 'new_results'
            model.mkdir()
            out.mkdir()
            (model / 'eval.csv').write_text('old results must remain\n')
            official, _ = fixture.make_official(model, events)
            official.FLAGS.save_dir = str(model)
            official.FLAGS.train_steps = 1000000
            official.FLAGS.eval_interval = 100000
            agent = official.agents['fake'].create(0, np.zeros((1, 46)), np.zeros((1, 5)), {})
            for step in STEPS:
                (model / f'params_{step}.pkl').write_text(json.dumps(dict(count=step)))
            bridge = LastThreeBridge(official, Path('/unused_no_dataset'), out)
            bridge.step = 1000000  # Existing source training budget, never new updates.
            bridge.final_agent = official.restore_agent(agent, model, 1000000)
            code, _ = formal.compile_deferred_main(official,
                (FORMAL / 'source_reference/main.py').read_bytes(), lambda *a: None)
            env = type('Env', (), {'task_infos': [dict(task_name='one'), dict(task_name='two')]})()
            env.unwrapped = env
            np.random.seed(18)
            for step in STEPS:
                before = np.random.get_state()
                for _ in range(2):
                    np.random.randint(0, 2**32, dtype=np.int64)
                bridge.nodes.append(dict(step=step, before=before, after=np.random.get_state(), env=env,
                                         config=official.FLAGS.agent))
            expected_tail = fixture.rng_bytes()
            csv_paths = []
            old_logger = official.CsvLogger

            def separate_logger(path):
                csv_paths.append(out / 'eval.csv')
                return old_logger(str(out / 'eval.csv'))

            official.CsvLogger = separate_logger
            with mock.patch.object(np.random, 'randint', linux_randint):
                bridge.run_deferred_evaluations(code)
            self.assertEqual(fixture.rng_bytes(), expected_tail)
            evaluations = [event for event in events if event[0] == 'eval']
            self.assertEqual([event[1] for event in evaluations], [800000, 800000, 900000, 900000, 1000000, 1000000])
            self.assertEqual([event for event in events if event[0] == 'train'], [])
            self.assertEqual(csv_paths, [out / 'eval.csv'])
            self.assertEqual((model / 'eval.csv').read_text(), 'old results must remain\n')
            status = json.loads((out / 'entry_status.json').read_text())
            self.assertEqual(status['training_updates_in_this_process'], 0)
            self.assertEqual(status['source_training_updates'], 1000000)
            self.assertNotIn('training_batches_returned', status)
            self.assertEqual(len(list(out.glob('deferred_eval_*.json'))), 3)


if __name__ == '__main__':
    unittest.main()
