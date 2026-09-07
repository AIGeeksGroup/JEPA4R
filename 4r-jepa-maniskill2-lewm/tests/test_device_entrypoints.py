"""CPU-only CLI routing checks; fake hardware reports never create a renderer."""

import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint(filename):
    spec = importlib.util.spec_from_file_location(
        f"test_loaded_{Path(filename).stem}", ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode,gpu", [
    ("train", True), ("train-cpu", False), ("smoke", False), ("eval-smoke", False),
])
def test_baseline_production_uses_gpu_and_cpu_is_explicit_or_smoke(mode, gpu):
    entry = load_entrypoint("run_ppo.py")
    command = entry.build_command(mode)
    assert Path(command[0]) == ROOT / (".venv-rl-gpu" if gpu else ".venv-rl") / "bin/python"
    if gpu:
        assert command[command.index("--gpu-ids") + 1] == "0"
        assert "--num-gpus" not in command
    else:
        assert command[command.index("--num-gpus") + 1] == "0"
        assert "--gpu-ids" not in command
    assert str(ROOT / "configs/ppo_pointnet_identity.py") in command


def test_baseline_no_argument_default_is_production_gpu():
    entry = load_entrypoint("run_ppo.py")
    assert entry.build_command() == entry.build_command("train")


def test_print_command_is_safe_without_gpu_or_environment(monkeypatch, tmp_path, capsys):
    entry = load_entrypoint("run_ppo.py")
    monkeypatch.setattr(entry, "ROOT", tmp_path)

    def forbidden_runtime(*args, **kwargs):
        pytest.fail("Printing a command must not query/probe/start GPU or training.")

    monkeypatch.setattr(entry, "require_headroom", forbidden_runtime)
    monkeypatch.setattr(entry.subprocess, "run", forbidden_runtime)
    monkeypatch.setattr(entry.subprocess, "call", forbidden_runtime)
    entry.main(["--print-command"])
    command = shlex.split(capsys.readouterr().out)
    assert Path(command[0]) == tmp_path / ".venv-rl-gpu/bin/python"
    assert command == entry.build_command("train")


def test_missing_production_gpu_environment_fails_without_cpu_fallback(monkeypatch, tmp_path):
    entry = load_entrypoint("run_ppo.py")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    # A CPU interpreter is present, but it must never be selected implicitly.
    cpu = tmp_path / ".venv-rl/bin/python"
    cpu.parent.mkdir(parents=True)
    cpu.write_text("THIS IS A TEST DOUBLE, NOT A PYTHON EXECUTABLE")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("A missing CUDA environment must fail before probing or launching.")

    monkeypatch.setattr(entry, "require_headroom", forbidden_launch)
    monkeypatch.setattr(entry.subprocess, "call", forbidden_launch)
    with pytest.raises(SystemExit, match="CPU fallback is not automatic"):
        entry.main([])


def test_failed_cuda_probe_does_not_launch_training_or_fallback(monkeypatch, tmp_path):
    entry = load_entrypoint("run_ppo.py")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    selected = tmp_path / ".venv-rl-gpu/bin/python"
    selected.parent.mkdir(parents=True)
    selected.write_text("TEST EXECUTABLE PLACEHOLDER; NEVER INVOKED")
    monkeypatch.setattr(entry, "require_headroom", lambda command: {"passed": True})

    def failed_probe(command, **kwargs):
        assert command[0] == str(selected)
        assert command[1] == "-c"
        raise subprocess.CalledProcessError(1, command)

    def forbidden_launch(*args, **kwargs):
        pytest.fail("An unsuccessful CUDA probe must not launch any training.")

    monkeypatch.setattr(entry.subprocess, "run", failed_probe)
    monkeypatch.setattr(entry.subprocess, "call", forbidden_launch)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main(["train"])


@pytest.mark.parametrize("free_mib,accepted", [(2048, False), (16384, True)])
def test_production_preflight_includes_model_memory_budget(monkeypatch, free_mib, accepted):
    entry = load_entrypoint("run_ppo.py")

    def fake_query(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert command[1].startswith("--query-gpu=")
        return f"0, {free_mib}\n"

    monkeypatch.setattr(entry.subprocess, "check_output", fake_query)
    if accepted:
        assert entry.require_headroom(entry.build_command())["passed"]
    else:
        with pytest.raises(RuntimeError, match="no training/renderer started"):
            entry.require_headroom(entry.build_command())


@pytest.mark.parametrize("requested,smoke,expected", [
    (None, False, "cuda"), (None, True, "cpu"),
    ("cpu", False, "cpu"), ("cuda", False, "cuda"),
    ("cpu", True, "cpu"), ("cuda", True, "cuda"),
])
def test_frozen_ppo_defaults_to_gpu_except_smoke(requested, smoke, expected):
    entry = load_entrypoint("run_frozen_ppo.py")
    assert entry.select_device(requested, smoke) == expected


def test_frozen_ppo_missing_cuda_environment_cannot_fallback(monkeypatch, tmp_path):
    entry = load_entrypoint("run_frozen_ppo.py")
    selected = tmp_path / "missing-worldmodel-gpu/bin/python"

    def fake_prepare(args):
        assert args.device == "cuda"
        return None, {}, [str(selected)]

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Missing frozen-PPO CUDA environment must not trigger a fallback.")

    monkeypatch.setattr(entry, "prepare", fake_prepare)
    monkeypatch.setattr(entry.subprocess, "call", forbidden_launch)
    monkeypatch.setattr(entry.subprocess, "run", forbidden_launch)
    monkeypatch.setattr(sys, "argv", ["run_frozen_ppo.py", "--wm-checkpoint", "unused.pt"])
    with pytest.raises(RuntimeError, match="no automatic CPU fallback"):
        entry.main()


@pytest.mark.parametrize("free_mib,accepted", [(512, False), (24576, True)])
def test_renderer_and_model_budget_refuses_low_memory_without_touching_jobs(
    monkeypatch, tmp_path, free_mib, accepted
):
    from ms2_lewm import environment

    commands = []

    def fake_query(command, **kwargs):
        commands.append(command)
        assert command[0] == "nvidia-smi"
        assert command[1].startswith("--query-gpu=")
        return SimpleNamespace(stdout=f"0, 80, 8192, {free_mib}\n")

    def forbidden_kill(*args, **kwargs):
        pytest.fail("Resource guards must never kill or signal existing jobs.")

    monkeypatch.setattr(environment, "TASK_ROOT", tmp_path)
    monkeypatch.setattr(environment.subprocess, "run", fake_query)
    monkeypatch.setattr(environment.os, "kill", forbidden_kill)
    monkeypatch.setenv("VK_ICD_FILENAMES", "/test/nvidia_icd.json")
    if accepted:
        report = environment.ensure_renderer_headroom(1, extra_model_mib=8192)
        assert report["passed"]
    else:
        with pytest.raises(RuntimeError, match="protect existing tasks"):
            environment.ensure_renderer_headroom(1, extra_model_mib=8192)
    assert len(commands) == 1
