import numpy as np
import pytest

from planning.frozen_probe import (
    binary_metrics,
    continuous_metrics,
    fit_logistic_probe,
    fit_ridge_probe,
)


def test_ridge_probe_recovers_known_multioutput_signal():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(160, 4))
    weights = np.asarray(
        [[1.5, -0.3], [-2.0, 0.2], [0.7, 1.1], [0.4, -1.3]]
    )
    targets = features @ weights + np.asarray([0.25, -0.75])

    probe = fit_ridge_probe(features, targets, alpha=1e-10)

    np.testing.assert_allclose(probe.predict(features), targets, atol=1e-8)
    assert probe.kind == "ridge"


def test_logistic_probe_separates_known_signal_deterministically():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(300, 3))
    labels = (features[:, 0] - 0.7 * features[:, 1] + 0.2 > 0).astype(np.int64)

    first = fit_logistic_probe(features, labels, l2=1e-5, max_iter=500)
    second = fit_logistic_probe(features, labels, l2=1e-5, max_iter=500)
    probabilities = first.predict(features)
    metrics = binary_metrics(labels, probabilities, train_positive_rate=labels.mean())

    np.testing.assert_array_equal(first.feature_mean, second.feature_mean)
    np.testing.assert_array_equal(first.feature_scale, second.feature_scale)
    np.testing.assert_array_equal(first.coef, second.coef)
    np.testing.assert_array_equal(first.intercept, second.intercept)
    assert metrics["auroc"] > 0.995
    assert metrics["balanced_accuracy"] > 0.97


def test_logistic_probe_rejects_one_class_training_labels():
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic_probe(np.ones((8, 2)), np.zeros(8, dtype=np.int64))


def test_continuous_metrics_include_constant_train_mean_baseline():
    target = np.asarray([1.0, 2.0, 4.0, 5.0])
    prediction = np.asarray([1.5, 2.0, 3.5, 5.0])

    metrics = continuous_metrics(target, prediction, train_mean=np.asarray(2.5))

    assert metrics["available"] is True
    assert metrics["n"] == 4
    assert metrics["target_std"] == pytest.approx(np.std(target))
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.125))
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["r2"] == pytest.approx(0.95)
    assert metrics["constant_train_mean_rmse"] == pytest.approx(
        np.sqrt(np.mean((target - 2.5) ** 2))
    )


def test_binary_metrics_include_counts_and_train_majority_baseline():
    target = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probability = np.asarray([0.1, 0.4, 0.7, 0.9])

    metrics = binary_metrics(target, probability, train_positive_rate=0.25)

    assert metrics["available"] is True
    assert metrics["n"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["constant_train_majority_class"] == 0
    assert metrics["constant_train_majority_accuracy"] == pytest.approx(0.5)


def test_binary_metrics_reports_one_class_validation_as_unavailable():
    metrics = binary_metrics(
        np.ones(5, dtype=np.int64),
        np.linspace(0.6, 0.9, 5),
        train_positive_rate=0.5,
    )

    assert metrics["available"] is False
    assert metrics["reason"] == "validation target contains only one class"
    assert metrics["positive_count"] == 5
    assert metrics["negative_count"] == 0
