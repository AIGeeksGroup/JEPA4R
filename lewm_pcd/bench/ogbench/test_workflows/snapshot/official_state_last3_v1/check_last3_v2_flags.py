"""Target-only flags prefix check: no output creation, env, model, or update."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=Path, required=True)
parser.add_argument('--ledger-dir', type=Path, required=True)
args = parser.parse_args()
path = HERE / 'evaluate_last3_v2.py'
spec = importlib.util.spec_from_file_location('_last3_flags_check_v2', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
main = next(node for node in ast.parse(path.read_bytes()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'main')
start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'source')
end = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assert)
           and isinstance(node.test, ast.Compare) and isinstance(node.test.comparators[0], ast.Name)
           and node.test.comparators[0].id == 'recorded')
prefix = ast.Module(body=main.body[start:end + 1], type_ignores=[])
for node in ast.walk(prefix):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in {'create', 'update', 'main', 'reset', 'step', 'make_env_and_datasets'}
namespace = dict(module.__dict__, model=args.model_dir.resolve(), ledger_dir=args.ledger_dir.resolve())
exec(compile(prefix, '<v2_original_flags_prefix_only>', 'exec'), namespace)
print(json.dumps(dict(actual_exit_code=0, entry_sha256=module.formal.sha(path),
    alpha=float(namespace['config'].alpha), full_recorded_config_equal=True,
    model_created=False, environment_created=False, agent_update_called=False, output_files_created=False)))
