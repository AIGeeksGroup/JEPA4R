"""No torch import, model, dataset, CUDA call or process launch."""
import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

CASE = Path(__file__).resolve().parent
REFERENCE = CASE.parent / 'double_cube_lejepa_match_3cube_20261026_v1'


class SourceContract(unittest.TestCase):
    def test_parsed_recovery_args_serialize_in_original_native_config(self):
        spec = importlib.util.spec_from_file_location('resume_entry_config_test', CASE / 'resume_entry.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        adapted = adapter.adapt_main((REFERENCE / 'source/train_lejepa_mw.py').read_text())
        main = ast.parse(adapted).body[0]
        parser_nodes = []
        for node in main.body:
            parser_nodes.append(node)
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'args'):
                break
        plan = json.loads((CASE / 'plan.json').read_text())
        native_args = plan['argv'][plan['argv'].index('--') + 1:]
        namespace = {'argparse': argparse, 'Path': Path, 'json': json}
        with patch('sys.argv', ['native'] + native_args):
            exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), '<real-native-argparse>', 'exec'), namespace)
        config_write = next(node for node in main.body if isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == 'write_text')
        native_json_call = config_write.value.args[0]
        serialized = eval(compile(ast.Expression(body=native_json_call), '<original-native-config-json>', 'eval'), namespace)
        config = json.loads(serialized)
        self.assertEqual(config['resume_model'], plan['recovery']['parent_model'])
        self.assertEqual(config['verified_export_root'], plan['verified_export_root'])
        self.assertIsInstance(namespace['args'].resume_model, str)
        self.assertIsInstance(namespace['args'].verified_export_root, str)

    def test_adapter_has_only_declared_main_edits(self):
        path = CASE / 'resume_entry.py'
        self.assertTrue(path.is_file(), 'weight-only recovery adapter is absent')
        spec = importlib.util.spec_from_file_location('resume_entry_test', path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        original = (REFERENCE / 'source/train_lejepa_mw.py').read_text()
        adapted = adapter.adapt_main(original)
        before = ast.parse(original)
        after = ast.parse(adapted)
        self.assertEqual([n.name for n in after.body], ['main'])
        old_main = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        original_main = ast.get_source_segment(original, old_main)
        self.assertEqual(adapter.restore_main(adapted), original_main)
        for node in ast.walk(after):
            if isinstance(node, ast.For) and ast.unparse(node.target) == 'epoch':
                self.assertEqual(ast.unparse(node.iter), 'range(args.start_epoch, args.epochs)')
        self.assertIn('torch.manual_seed(0)', adapted)
        self.assertIn('warmup_steps, total_steps = len(train), len(train) * args.epochs', adapted)
        self.assertIn("ap.add_argument('--start-epoch', type=int, default=0)", adapted)
        self.assertIn("ap.add_argument('--resume-model', type=str, default=None)", adapted)
        self.assertIn('range(self.V)', original)
        self.assertEqual(hashlib.sha256((REFERENCE / 'source/train_lejepa_mw.py').read_bytes()).hexdigest(), adapter.CORE_SHA)

    def test_plan_uses_new_output_and_full_scheduler_budget(self):
        plan = json.loads((CASE / 'plan.json').read_text())
        self.assertEqual(plan['recovery']['start_epoch'], 7)
        self.assertEqual(plan['recovery']['total_epochs'], 10)
        self.assertEqual(plan['budget']['nominal_new_batches'], 58593)
        self.assertEqual(plan['budget']['total_original_scheduler_budget'], 195310)
        self.assertEqual(plan['environment']['STABLEWM_HOME'], plan['output_root'])
        self.assertIn(CASE.name, plan['output_root'])
        self.assertIn(CASE.name, plan['verified_export_root'])
        self.assertFalse(plan['storage']['shared_mirror_enabled'])
        native_args = plan['argv'][plan['argv'].index('--') + 1:]
        flags = dict(zip(native_args[::2], native_args[1::2]))
        self.assertEqual(flags['--start-epoch'], '7')
        self.assertEqual(flags['--epochs'], '10')
        self.assertEqual(flags['--resume-model'], plan['recovery']['parent_model'])
        self.assertEqual(flags['--verified-export-root'], plan['verified_export_root'])


if __name__ == '__main__':
    unittest.main()
