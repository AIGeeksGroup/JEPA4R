from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_liftcube_sonata_frozen_2500.sh"


def test_runner_allows_overriding_gpu_and_world_model_environment():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"' in script
    assert "export PYTHONDONTWRITEBYTECODE=1" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert 'python="${WORLDMODEL_PYTHON:-/nix/lewm/4d-jepa-worldmodel/bin/python}"' in script
    assert "environments/mpc" not in script
    assert "cem" not in script.lower()


def test_runner_resolves_git_from_path_with_optional_override():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'git_bin="${GIT_BIN:-$(command -v git)}"' in script


def test_runner_refuses_existing_outputs_and_duplicate_live_pid():
    script = RUNNER.read_text(encoding="utf-8")

    assert '[[ -f "$pid_file" ]]' in script
    assert 'kill -0 "$previous_pid"' in script
    assert "refusing to overwrite existing run" in script
    assert "refusing to duplicate live process" in script


def test_runner_persists_atomic_status_pid_log_and_all_diagnostics():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'status="${run_root}/${name}.status"' in script
    assert 'pid_file="${run_root}/${name}.pid"' in script
    assert 'log="${run_root}/${name}.log"' in script
    assert '[[ -e "$run_dir" || -e "$status" || -e "$pid_file" || -e "$log" ]]' in script
    assert 'mkdir -p "$run_dir"' not in script
    for state in ("PREPARING", "RUNNING", "EVALUATING", "COMPLETE", "FAILED"):
        assert f"write_status {state}" in script
    assert 'mv "$temporary" "$status"' in script
    assert "trap on_exit EXIT" in script
    for step in (500, 1000, 1500, 2000, 2500):
        assert f"checkpoint_{step:06d}" in script
    assert "best.pt" in script
    assert '"$python" -m scripts.eval_liftcube_latent_diagnostics' in script
    assert '"$python" scripts/eval_liftcube_latent_diagnostics.py' not in script


def test_runner_uses_the_frozen_sonata_config_and_rgb_data():
    script = RUNNER.read_text(encoding="utf-8")

    assert "configs/maniskill_group1_sonata_frozen.yaml" in script
    assert "/compact_rgb/LiftCube-v0" in script
    assert "group1_skip10_sonata_frozen_batch8_2500" in script
    assert "--horizons 1 2 4" in script
    assert "--batch-size 8" in script
