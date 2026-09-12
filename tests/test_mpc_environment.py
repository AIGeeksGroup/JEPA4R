from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _locked_requirements(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("--")
    }


def test_mpc_requirements_are_standalone_and_model_compatible() -> None:
    requirements = ROOT / "requirements-mpc.txt"
    environment_project = ROOT / "environments" / "mpc" / "pyproject.toml"
    assert requirements.is_file(), "requirements-mpc.txt is missing"
    assert environment_project.is_file(), "downstream MPC pyproject.toml is missing"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mpc_project = tomllib.loads(environment_project.read_text(encoding="utf-8"))
    locked = _locked_requirements(requirements)
    assert mpc_project["project"]["name"] == "pointcloud-world-model-mpc-runtime"
    assert mpc_project["project"]["requires-python"] == ">=3.11,<3.12"
    assert set(mpc_project["project"]["dependencies"]) == locked
    assert "mpc" not in project["project"].get("optional-dependencies", {})
    shared_model_runtime = {
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "spconv-cu120==2.3.6",
        "timm==1.0.29",
        "addict==2.4.0",
        "einops==0.8.2",
        "omegaconf==2.3.1",
        "pytest==8.3.5",
    }
    sonata_runtime = {
        "huggingface-hub==1.29.0",
        (
            "torch-scatter @ https://data.pyg.org/whl/torch-2.5.0+cu121/"
            "torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl"
        ),
        (
            "sonata @ git+https://github.com/facebookresearch/sonata.git@"
            "18c09ff8d713494f78a8213792262b910977a65d"
        ),
    }
    assert shared_model_runtime <= locked
    assert sonata_runtime <= locked
    assert sonata_runtime <= set(mpc_project["project"]["dependencies"])
    assert sonata_runtime <= set(project["project"]["dependencies"])
    assert "numpy==1.26.4" not in locked
    assert "numpy==1.23.5" in locked
    assert "planning" in project["tool"]["setuptools"]["packages"]


def test_setup_uses_separate_nix_environment() -> None:
    setup = ROOT / "scripts" / "setup_liftcube_mpc_env.sh"
    assert setup.is_file(), "MPC setup script is missing"
    script = setup.read_text(encoding="utf-8")
    assert "env_dir=/nix/lewm/4d-jepa-mpc" in script
    assert "/nix/lewm/4d-jepa-worldmodel/bin/python" in script
    assert "--system-site-packages" not in script
    assert "-r requirements.txt" not in script
    assert "-r requirements-mpc.txt" in script
    assert "/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment" in script
    assert "pip install" in script
    assert "BASH_SOURCE[0]" in script
    assert "pip install -e ." not in script
    assert "liftcube-multistep-evaluation" not in script
    assert "uv " not in script


def test_checker_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    assert checker_path.is_file(), "MPC environment checker is missing"
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"not the frozen checkpoint")
    expected = "0" * 64
    report = module.check_environment(
        checkpoint,
        require_renderer=False,
        expected_checkpoint_sha256=expected,
    )

    assert report["ok"] is False
    assert any(expected in error for error in report["errors"])


def test_checker_declares_required_renderer_system_packages() -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.EXPECTED_SYSTEM_PACKAGES == {
        "libgl1": "1.4.0-1",
        "libglib2.0-0": "2.72.4-0ubuntu2.9",
        "libvulkan1": "1.3.204.1-2",
    }
    apt_lock = ROOT / "environments" / "mpc" / "apt-packages.txt"
    assert apt_lock.is_file()
    assert _locked_requirements(apt_lock) == {
        f"{package}={version}"
        for package, version in module.EXPECTED_SYSTEM_PACKAGES.items()
    }
    assert module.EXPECTED_VERSIONS["numpy"] == "1.23.5"
    assert module.EXPECTED_VERSIONS["huggingface-hub"] == "1.29.0"
    assert module.EXPECTED_VERSIONS["torch-scatter"] == "2.1.2+pt25cu121"
    assert module.EXPECTED_VERSIONS["sonata"] == "1.0"


def test_checker_can_restore_repo_import_path(monkeypatch) -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT],
    )

    module.ensure_repo_importable()

    assert Path(sys.path[0]).resolve() == ROOT


def test_renderer_setup_selects_nvidia_vulkan_icd(monkeypatch) -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    module.configure_vulkan_icd()

    assert os.environ["VK_ICD_FILENAMES"] == module.DEFAULT_VULKAN_ICD


def test_checker_selects_vulkan_icd_during_module_import(monkeypatch) -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/etc/vulkan/icd.d/nvidia_icd.json",
    )
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert os.environ["VK_ICD_FILENAMES"] == module.DEFAULT_VULKAN_ICD


def test_checker_falls_back_to_exact_rpm_system_package(monkeypatch) -> None:
    checker_path = ROOT / "scripts" / "check_liftcube_mpc_env.py"
    spec = importlib.util.spec_from_file_location("check_liftcube_mpc_env", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda command: "/usr/bin/rpm" if command == "rpm" else None,
    )

    def check_output(command, text):
        calls.append((command, text))
        return "libglvnd-glx-1.3.2-1.el8.x86_64\n"

    monkeypatch.setattr(module.subprocess, "check_output", check_output)

    actual, expected = module._query_system_package("libgl1")

    assert actual == expected == "libglvnd-glx-1.3.2-1.el8.x86_64"
    assert calls == [(["rpm", "-q", "libglvnd-glx"], True)]


def test_lzh24_sonata_proprio_goal_cem_runner_is_recoverable() -> None:
    runner = ROOT / "scripts" / "run_liftcube_sonata_proprio_goal_cem_10seeds_lzh24.sh"
    assert runner.is_file(), "lzh-24 CEM runner is missing"
    script = runner.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python" in script
    assert "/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt" in script
    assert "681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28" in script
    assert "/public/home/lzh_24/project/JEPA-4R/data/compact_rgb_proprio/LiftCube-v0" in script
    assert "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json" in script
    assert "--seeds 4100 4101 4102 4103 4104 4105 4106 4107 4108 4109" in script
    assert "--cem-seed 31000" in script
    assert "--candidates 128" in script
    assert "--model-batch-size 32" in script
    assert "--iterations 5" in script
    assert "--elites 16" in script
    assert "--horizon 2" in script
    assert "--chunk-size 10" in script
    assert "--max-steps 200" in script
    assert 'status_file="${run_dir}.status"' in script
    assert 'pid_file="${run_dir}.pid"' in script
    assert 'log_file="${run_dir}.log"' in script
    assert 'mkdir -p "$(dirname "$run_dir")"' in script
    assert 'mkdir -p "$run_dir"' not in script
    assert "write_status RUNNING" in script
    assert "write_status COMPLETE" in script
    assert "write_status FAILED" in script


def test_latent_cem_runner_is_recoverable_and_uses_formal_settings() -> None:
    runner = ROOT / "scripts" / "run_liftcube_latent_cem_experiment.sh"
    assert runner.is_file(), "latent CEM runner is missing"
    script = runner.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "/nix/lewm/4d-jepa-mpc/bin/python" in script
    assert (
        "/workspace/users/lewm/runs/4d-jepa-architecture/"
        "group1_skip10_action_sequence_batch8_2500/best.pt"
    ) in script
    assert "--seeds 4100 4101 4102 4103 4104" in script
    assert "--candidates 128" in script
    assert "--iterations 5" in script
    assert "--elites 16" in script
    assert "--horizon 2" in script
    assert "--chunk-size 10" in script
    assert "--max-steps 200" in script
    assert 'status_file="${run_dir}.status"' in script
    assert 'pid_file="${run_dir}.pid"' in script
    assert 'log_file="${run_dir}.log"' in script
    assert '[[ -e "$run_dir" || -e "$status_file" || -e "$log_file" ]]' in script
    assert "kill -0" in script
    assert "write_status RUNNING" in script
    assert "write_status COMPLETE" in script
    assert "write_status FAILED" in script


def test_sonata_latent_cem_runner_is_recoverable_and_uses_ten_seeds() -> None:
    runner = ROOT / "scripts" / "run_liftcube_sonata_latent_cem_10seeds.sh"
    assert runner.is_file(), "Sonata latent CEM runner is missing"
    script = runner.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "/nix/lewm/4d-jepa-mpc/bin/python" in script
    assert (
        "/workspace/users/lewm/runs/4d-jepa-architecture/"
        "group1_skip10_sonata_frozen_batch8_2500/best.pt"
    ) in script
    assert "38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32" in script
    assert "liftcube_sonata_latent_cem_best_10seeds" in script
    assert "/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0" in script
    assert "--seeds 4100 4101 4102 4103 4104 4105 4106 4107 4108 4109" in script
    assert "--cem-seed 31000" in script
    assert "--candidates 128" in script
    assert "--model-batch-size 32" in script
    assert "--iterations 5" in script
    assert "--elites 16" in script
    assert "--horizon 2" in script
    assert "--chunk-size 10" in script
    assert "--max-steps 200" in script
    assert "CUDA_VISIBLE_DEVICES=2" in script
    assert "--expected-checkpoint-sha256" in script
    assert "--require-renderer" in script
    assert 'status_file="${run_dir}.status"' in script
    assert 'pid_file="${run_dir}.pid"' in script
    assert 'log_file="${run_dir}.log"' in script
    assert '[[ -e "$run_dir" || -e "$status_file" || -e "$log_file" ]]' in script
    assert "kill -0" in script
    assert "write_status RUNNING" in script
    assert "write_status COMPLETE" in script
    assert "write_status FAILED" in script
