"""Deterministic linear probes and metrics for frozen representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


def _features(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"features must be a non-empty [N,D] array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    return array


def _standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


@dataclass(frozen=True)
class StandardizedLinearProbe:
    """Linear map operating on train-standardized features."""

    kind: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    output_was_1d: bool

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = _features(features)
        if values.shape[1] != self.feature_mean.shape[0]:
            raise ValueError(
                f"feature dimension {values.shape[1]} does not match fitted "
                f"dimension {self.feature_mean.shape[0]}"
            )
        output = (
            (values - self.feature_mean) / self.feature_scale
        ) @ self.coef + self.intercept
        return output[:, 0] if self.output_was_1d else output

    def predict(self, features: np.ndarray) -> np.ndarray:
        output = self.decision_function(features)
        return expit(output) if self.kind == "logistic" else output

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "output_was_1d": self.output_was_1d,
        }


def fit_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> StandardizedLinearProbe:
    """Fit a deterministic ridge regression with an unpenalized intercept."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x = _features(features)
    y = np.asarray(targets, dtype=np.float64)
    output_was_1d = y.ndim == 1
    if output_was_1d:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] < 1:
        raise ValueError(f"targets must have shape [N] or [N,O], got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("targets must be finite")

    feature_mean, feature_scale = _standardization(x)
    standardized = (x - feature_mean) / feature_scale
    intercept = y.mean(axis=0)
    centered_target = y - intercept
    gram = standardized.T @ standardized
    regularized = gram + float(alpha) * np.eye(gram.shape[0])
    right_hand_side = standardized.T @ centered_target
    try:
        coef = np.linalg.solve(regularized, right_hand_side)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(regularized, right_hand_side, rcond=None)[0]
    return StandardizedLinearProbe(
        kind="ridge",
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coef=coef,
        intercept=intercept,
        output_was_1d=output_was_1d,
    )


def fit_logistic_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
    max_iter: int = 500,
) -> StandardizedLinearProbe:
    """Fit deterministic class-balanced binary logistic regression."""
    if l2 < 0 or max_iter < 1:
        raise ValueError("l2 must be non-negative and max_iter must be positive")
    x = _features(features)
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"labels must have shape [N], got {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    y = y.astype(np.float64, copy=False)
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("training labels must contain both classes")

    feature_mean, feature_scale = _standardization(x)
    standardized = (x - feature_mean) / feature_scale
    sample_weight = np.where(
        y == 1.0,
        x.shape[0] / (2.0 * counts[1]),
        x.shape[0] / (2.0 * counts[0]),
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:-1], parameters[-1]
        logits = standardized @ weights + bias
        losses = np.logaddexp(0.0, logits) - y * logits
        probability = expit(logits)
        residual = sample_weight * (probability - y) / x.shape[0]
        value = float(np.sum(sample_weight * losses) / x.shape[0])
        value += 0.5 * float(l2) * float(weights @ weights)
        gradient = np.empty_like(parameters)
        gradient[:-1] = standardized.T @ residual + float(l2) * weights
        gradient[-1] = residual.sum()
        return value, gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"logistic probe optimization failed: {result.message}")
    return StandardizedLinearProbe(
        kind="logistic",
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coef=result.x[:-1, None],
        intercept=np.asarray([result.x[-1]], dtype=np.float64),
        output_was_1d=True,
    )


def continuous_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    train_mean: np.ndarray,
) -> dict[str, object]:
    """Return aggregate regression metrics and a train-mean baseline."""
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError(
            f"target and prediction must have the same non-empty shape, got "
            f"{truth.shape} and {estimate.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("target and prediction must be finite")
    baseline = np.asarray(train_mean, dtype=np.float64)
    try:
        baseline_error = truth - np.broadcast_to(baseline, truth.shape)
    except ValueError as exc:
        raise ValueError("train_mean is not broadcastable to target") from exc
    error = estimate - truth
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    result: dict[str, object] = {
        "available": True,
        "n": int(truth.shape[0]),
        "target_std": float(np.std(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": None if total_sum <= 0.0 else 1.0 - residual_sum / total_sum,
        "constant_train_mean_rmse": float(np.sqrt(np.mean(baseline_error**2))),
    }
    if truth.ndim == 2 and truth.shape[1] > 1:
        result["per_dimension"] = [
            continuous_metrics(
                truth[:, index],
                estimate[:, index],
                train_mean=np.asarray(baseline).reshape(-1)[index],
            )
            for index in range(truth.shape[1])
        ]
    return result


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    ordered = labels[order]
    ordered_score = probabilities[order]
    cumulative_positive = np.cumsum(ordered)
    threshold_ends = np.r_[
        np.flatnonzero(ordered_score[1:] != ordered_score[:-1]),
        len(ordered_score) - 1,
    ]
    precision = cumulative_positive[threshold_ends] / (threshold_ends + 1)
    recall = cumulative_positive[threshold_ends] / np.sum(ordered)
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gain * precision))


def binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    train_positive_rate: float,
) -> dict[str, object]:
    """Return binary ranking/classification metrics and a train-majority baseline."""
    truth = np.asarray(target)
    score = np.asarray(probability, dtype=np.float64)
    if truth.ndim != 1 or score.shape != truth.shape or truth.size == 0:
        raise ValueError("target and probability must be same-shape non-empty vectors")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("binary target must contain only 0 and 1")
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("probabilities must be finite values in [0,1]")
    if not 0.0 <= float(train_positive_rate) <= 1.0:
        raise ValueError("train_positive_rate must be in [0,1]")
    truth = truth.astype(np.int64, copy=False)
    positive = int(truth.sum())
    negative = int(len(truth) - positive)
    majority = int(float(train_positive_rate) >= 0.5)
    result: dict[str, object] = {
        "available": positive > 0 and negative > 0,
        "n": int(len(truth)),
        "positive_count": positive,
        "negative_count": negative,
        "constant_train_majority_class": majority,
        "constant_train_majority_accuracy": float(np.mean(truth == majority)),
    }
    if not result["available"]:
        result["reason"] = "validation target contains only one class"
        result.update(
            {"auroc": None, "average_precision": None, "balanced_accuracy": None}
        )
        return result

    ranks = rankdata(score, method="average")
    auroc = (
        float(ranks[truth == 1].sum()) - positive * (positive + 1) / 2.0
    ) / (positive * negative)
    predicted = (score >= 0.5).astype(np.int64)
    true_positive_rate = float(np.mean(predicted[truth == 1] == 1))
    true_negative_rate = float(np.mean(predicted[truth == 0] == 0))
    result.update(
        {
            "auroc": float(auroc),
            "average_precision": _average_precision(truth, score),
            "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        }
    )
    return result
