from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryTests(unittest.TestCase):
    def test_generated_python_artifacts_are_ignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("__pycache__/", ignore)
        self.assertIn("*.py[cod]", ignore)
        self.assertIn(".pytest_cache/", ignore)
        self.assertIn("runs/", ignore)
        self.assertIn("*.pt", ignore)

    def test_package_contains_only_upstream_modules(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["tool"]["setuptools"]["packages"], ["models", "losses", "data"])

    def test_first_party_defaults_are_machine_independent(self) -> None:
        machine_path = re.compile(r"/(?:workspace|nix|root)/")
        violations: list[str] = []
        for folder in ("configs", "scripts"):
            for path in (ROOT / folder).rglob("*"):
                if path.suffix not in {".py", ".yaml", ".yml"}:
                    continue
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if machine_path.search(line):
                        violations.append(f"{path.relative_to(ROOT)}:{line_number}")
        self.assertEqual(violations, [])

    def test_readme_matches_the_current_action_conditioning_api(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for term in ("skip_frame", "ordered action tokens", "cross_attention", "ranking loss", "ManiSkill2"):
            self.assertIn(term, readme)

    def test_downstream_mpc_is_not_part_of_the_core_release(self) -> None:
        for relative_path in ("planning", "environments/mpc", "requirements-mpc.txt"):
            self.assertFalse((ROOT / relative_path).exists(), relative_path)

    def test_adapted_code_keeps_third_party_notices(self) -> None:
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for project in ("Pointcept", "HMA", "LeWorldModel"):
            self.assertIn(project, notice)
        for license_file in (
            "LICENSES/Pointcept-MIT.txt",
            "LICENSES/HMA-Apache-2.0.txt",
            "LICENSES/LeWorldModel-MIT.txt",
        ):
            self.assertTrue((ROOT / license_file).is_file(), license_file)


if __name__ == "__main__":
    unittest.main()
