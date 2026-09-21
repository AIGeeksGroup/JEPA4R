from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_ur5e_task1_expand10k.sh"


def script_text():
    return SCRIPT.read_text(encoding="utf-8")


def test_supervisor_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_supervisor_preserves_exact_data_contract():
    text = script_text()
    assert "COMPLETED_EXTENSION_EPISODES=940" in text
    assert "REMAINING_EXPERT_EPISODES=8060" in text
    assert "NEW_EXPERT_EPISODES=9000" in text
    assert "ORIGINAL_EXPERT_TRAIN_EPISODES=900" in text
    assert "EXPANDED_EXPERT_TRAIN_EPISODES=9900" in text
    assert "RANDOM_FAILURE_EPISODES=100" in text
    assert "HOLDOUT_EXPERT_EPISODES=100" in text
    assert "EXPANDED_TRAIN_EPISODES=10000" in text
    assert "MAX_PARALLEL=16" in text
    assert "collect_fixed_ur5e_task_pcd_sharded.py" in text
    assert "select_fixed_ur5e_episodes.py" in text
    assert "publish_fixed_ur5e_task_dataset.py" in text


def test_smoke_gate_covers_collection_training_and_reload():
    text = script_text()
    assert "run_smoke" in text
    assert "SMOKE_ACCEPTED" in text
    assert "--episodes 4" in text
    assert "--episodes 2" in text
    assert "trainer.max_epochs=1" in text
    assert "+trainer.limit_train_batches=2" in text
    assert "+trainer.limit_val_batches=1" in text
    assert "check_ur5e_holdout_checkpoint.py" in text
    assert 'grep -q \'"status": "PASS"\'' in text


def test_formal_training_is_gated_and_has_distinct_outputs():
    text = script_text()
    assert "FULL_DATA_ACCEPTED" in text
    assert "TRAINING_ACCEPTED" in text
    assert "model/encoder=dexart" in text
    assert "trainer.max_epochs=10" in text
    assert "ur5e_cube_double_task1_single_pnp_pcd_dexart_expert10k_10ep" in text
    assert "weights_epoch_10.pt" in text
    assert text.index("run_smoke") < text.index("run_full_collection")
    assert text.index("run_full_collection") < text.index("run_formal_training")
    assert "cem" not in text.lower()
