"""Reuse the existing two narrow checks with the v2 entry module."""
import importlib.util
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('evaluate_last3', HERE / 'evaluate_last3_v2.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runpy.run_path(str(HERE / 'test_evaluate_last3.py'), run_name='__main__')
