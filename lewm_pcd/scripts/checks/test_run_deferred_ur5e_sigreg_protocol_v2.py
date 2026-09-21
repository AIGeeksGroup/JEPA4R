from pathlib import Path


SCRIPT = Path(__file__).with_name("run_deferred_ur5e_sigreg_protocol_v2.sh")


def text():
    return SCRIPT.read_text(encoding="utf-8")


def test_deferred_recheck_uses_pipeline_lock_and_waits_for_idle_gpu():
    value = text()
    assert "ur5e_multitask_pipeline.lock" in value
    assert "flock -n 9" in value
    assert "nvidia-smi --query-compute-apps=pid" in value
    assert "STABLEWM_HOME=$RUNTIME_ROOT" in value
    assert "PYTHONPATH=$ROOT:$ROOT/third_party/stable-worldmodel" in value
    assert "SIGREG_V2_WAITING_FOR_PIPELINE" in value
    assert "SIGREG_V2_WAITING_FOR_GPU" in value


def test_deferred_recheck_preserves_checkpoints_and_old_reports():
    value = text()
    assert value.count("check_ur5e_holdout_checkpoint.py") == 2
    assert "checkpoint_epoch_10_protocol_v2.json" in value
    assert "checkpoint_epoch_10.json" not in value
    assert "sha256sum" in value
    assert "cmp" in value
    assert "train.py" not in value
