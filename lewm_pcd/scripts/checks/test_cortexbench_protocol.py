from lewm_pcd.bench.cortexbench.protocol import (
    expected_evaluator_records,
    num_demos_for_suite,
    validate_metric_records,
)


def test_adroit_uses_official_100_demo_route_and_metaworld_keeps_25():
    assert num_demos_for_suite("adroit") == 100
    assert num_demos_for_suite("metaworld") == 25


def test_evaluator_schedule_has_twenty_records_for_official_bc():
    assert expected_evaluator_records(epochs=100, eval_frequency=5) == 20
    assert expected_evaluator_records(epochs=1, eval_frequency=1) == 1


def test_formal_run_rejects_incomplete_metrics_and_extracts_best_of_twenty():
    records = [{"epoch_loss": 1.0} for _ in range(100)]
    records.extend(
        {
            "eval/epoch": float(epoch),
            "eval/success": float(index),
            "eval/highest_success": float(index),
            "eval/score_mean": float(index),
            "eval/highest_score": float(index),
        }
        for index, epoch in enumerate([*range(5, 100, 5), 99])
    )

    result = validate_metric_records(records, epochs=100, eval_frequency=5)
    assert result["loss_records"] == 100
    assert result["eval_records"] == 20
    assert result["best_success"] == 19.0
    assert result["final_success"] == 19.0

    try:
        validate_metric_records(records[:-1], epochs=100, eval_frequency=5)
    except ValueError as exc:
        assert "evaluator records" in str(exc)
    else:
        raise AssertionError("an incomplete evaluator trace must be rejected")
