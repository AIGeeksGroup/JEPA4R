"""One CPU regression for nonfinite candidate metadata; no Torch/model/GPU."""
import ast
import json
from pathlib import Path
import unittest


class NonfiniteReceipt(unittest.TestCase):
    def test_nonfinite_candidate_serializes_as_negative_cpu_selection(self):
        tree = ast.parse((Path(__file__).parent / 'produce.py').read_bytes())
        keywords = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update':
                for keyword in node.keywords:
                    if keyword.arg in ('pixel_max_abs_diff', 'latent_max_abs_diff', 'selected_device', 'negative_reason'):
                        keywords.setdefault(keyword.arg, keyword.value)

        class NonfiniteTensor:
            def __sub__(self, other): return self
            def abs(self): return self
            def max(self): return float('nan')

        bad = NonfiniteTensor()
        scope = dict(finite=False, numeric=False, faster=False, cpu_pixels=bad, gpu_pixels=bad,
                     cpu_latent=bad, gpu_latent=bad)
        record = {name: eval(compile(ast.Expression(keywords[name]), '<candidate_metadata>', 'eval'), scope)
                  for name in ('pixel_max_abs_diff', 'latent_max_abs_diff')}
        encoded = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(encoded), dict(pixel_max_abs_diff=None, latent_max_abs_diff=None))
        self.assertEqual(eval(compile(ast.Expression(keywords['selected_device']), '<selection>', 'eval'), scope), 'cpu')
        self.assertEqual(eval(compile(ast.Expression(keywords['negative_reason']), '<reason>', 'eval'), scope), 'cuda_nonfinite')


if __name__ == '__main__':
    unittest.main()
