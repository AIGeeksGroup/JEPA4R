"""The code style the repository keeps: the `ruff format` layout and the lint rules of pyproject.toml."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("ruff"), "ruff is not installed (pip install -e '.[dev]')")
class StyleTest(unittest.TestCase):
    def ruff(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "ruff", *args, "--no-cache", "."], cwd=ROOT, capture_output=True, text=True
        )

    def test_layout_is_ruff_format(self):
        result = self.ruff("format", "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lint_rules_pass(self):
        result = self.ruff("check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
