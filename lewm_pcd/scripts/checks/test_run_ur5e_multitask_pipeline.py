from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_ur5e_multitask_pipeline.sh"


def text():
    return SCRIPT.read_text(encoding="utf-8")


def function_body(name: str) -> str:
    value = text()
    start = value.index(f"{name}() {{")
    end = value.index("\n}\n", start)
    return value[start:end]


def test_pipeline_declares_all_nine_tasks_and_two_gates():
    value = text()
    for slug in (
        "cube_double_task1_single_pnp",
        "cube_double_task2_double_pnp1",
        "cube_double_task3_double_pnp2",
        "cube_double_task4_swap",
        "cube_double_task5_stack",
        "cube_triple_task1_single_pnp",
        "cube_triple_task2_triple_pnp",
        "cube_triple_task3_pnp_from_stack",
        "cube_triple_task4_cycle",
    ):
        assert slug in value
    assert "SMOKE_ACCEPTED" in value
    assert "FULL_ACCEPTED" in value
    assert "smoke" in value and "full" in value


def test_pipeline_has_fail_stop_and_no_cem():
    value = text()
    assert "set -euo pipefail" in value
    assert "flock" in value
    assert "refusing to overwrite" in value
    assert "CEM" not in value
    assert "FULL_ACCEPTED" in value
    assert "SMOKE_ACCEPTED" in value


def test_pipeline_uses_explicit_holdout_training_and_dexart():
    value = text()
    assert "data=ogb_ur5e_task_pcd" in value
    assert "model/encoder=dexart" in value
    assert '_pcd_dexart_"$epochs"ep' in value
    assert 'checkpoint_dexart_epoch_"$epochs".json' in value
    assert "UR5E_HOLDOUT_DATA" in value
    assert "check_ur5e_holdout_checkpoint.py" in value

def test_full_collection_reuses_one_environment_for_twenty_episode_batches():
    value = text()
    assert "FULL_COLLECT_ENVS=1" in value
    assert "FULL_COLLECT_BATCH=20" in value
    assert '--num-envs "$COLLECT_ENVS"' in value
    assert '--batch-size "$COLLECT_BATCH"' in value


def test_full_training_keeps_original_batch_size():
    value = text()
    assert "FULL_TRAIN_BATCH=128" in value
    assert 'loader.batch_size="$train_batch"' in value


def test_publication_rechunks_and_removes_validated_work_directories():
    value = text()
    assert "rechunk_h5.py" in value
    assert 'mv "$rechunked" "$train"' in value
    assert '"$holdout.work"' in value


def test_full_run_reuses_already_published_train_and_holdout_data():
    value = text()
    assert 'if [ ! -s "$d/train.h5" ] || [ ! -s "$d/holdout.h5" ]; then' in value


def test_parallel_smoke_uses_two_real_shards_and_writes_marker_after_validation():
    value = text()
    assert "parallel-smoke" in value
    assert "PARALLEL_COLLECTION_SMOKE_ACCEPTED" in value
    assert 'collect_parallel "$family" "$id" expert 4' in value
    assert 'collect_parallel "$family" "$id" random 4' in value
    assert "PARALLEL_SMOKE_SHARDS=2" in value
    assert "PARALLEL_SMOKE_MAX_PARALLEL=2" in value


def test_full_collection_uses_eight_way_sharding_after_smoke_gate():
    value = text()
    assert "FULL_COLLECT_SHARDS=8" in value
    assert "FULL_COLLECT_MAX_PARALLEL=8" in value
    assert '--max-parallel "$max_parallel"' in value
    assert 'test -s "$PARALLEL_SMOKE_MARKER"' in value


def test_validated_publication_removes_only_shard_intermediates():
    value = text()
    assert '"$expert.shards"' in value
    assert '"$random.shards"' in value
    assert '"$holdout.shards"' in value


def test_parallel_collection_has_gpu_and_disk_resource_gate():
    value = text()
    assert "MIN_COLLECT_FREE_GB=20" in value
    assert "collection_resource_gate" in value
    assert "nvidia-smi --query-compute-apps=pid" in value
    assert "df -PB1" in value


def test_full_run_collects_and_validates_all_data_before_training():
    value = text()
    full = function_body("run_full")
    assert "FULL_DATA_ACCEPTED" in value
    assert full.index("run_full_collection") < full.index("validate_all_full_data")
    assert full.index("validate_all_full_data") < full.index("run_full_training")
    assert "train " not in function_body("run_full_collection")
    assert "collect_parallel" not in function_body("run_full_training")


def test_full_training_requires_atomic_all_data_marker():
    training = function_body("run_full_training")
    validation = function_body("validate_all_full_data")
    assert 'test -s "$FULL_DATA_MARKER"' in training
    assert 'write_marker "$FULL_DATA_MARKER"' in validation
    assert 'validate "$d/train.h5"' in validation
    assert 'validate "$d/holdout.h5"' in validation


def test_training_cache_is_task_scoped_and_removed_after_holdout_check():
    body = function_body("train")
    assert 'task_cache=$RUNTIME_ROOT/checkpoints/_spt_cache/$run' in body
    assert 'SPT_CACHE_DIR="$task_cache" CUDA_VISIBLE_DEVICES=0' in body
    assert body.index("check_ur5e_holdout_checkpoint.py") < body.index(
        'rm -rf -- "$task_cache"'
    )
