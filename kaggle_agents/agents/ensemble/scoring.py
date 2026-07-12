"""Scoring functions for ensemble evaluation."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


_LOG_LOSS_NAMES = ("logloss", "log_loss", "log loss", "cross_entropy")
_RMSLE_NAMES = (
    "rmsle",
    "root_mean_squared_log",
    "root mean squared logarithmic",
)
_RMSE_NAMES = ("rmse", "root_mean_squared_error", "root mean squared")
_MSE_NAMES = ("mean_squared_error", "mean squared error", "mse")
_MAE_NAMES = ("mean_absolute_error", "mean absolute error", "mae")
_LABEL_METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "kappa",
    "qwk",
    "mcc",
    "matthews",
)


def _contains(metric: str, names: tuple[str, ...]) -> bool:
    return any(name in metric for name in names)


def _encode_class_labels(y_true: np.ndarray) -> np.ndarray:
    """Encode arbitrary class labels to the probability-column convention."""
    y = np.asarray(y_true)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            return np.argmax(y, axis=1).astype(int)
    _, encoded = np.unique(y.ravel(), return_inverse=True)
    return encoded.astype(int)


def _normalize_probabilities(preds: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(preds, dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Probability rows must have a positive sum")
        probabilities = probabilities / row_sums
    return probabilities


def _hard_labels(preds: np.ndarray, y_true: np.ndarray, metric: str) -> np.ndarray:
    """Convert probabilities/continuous outputs to encoded hard labels."""
    predictions = np.asarray(preds, dtype=float)
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        return np.argmax(predictions, axis=1)

    flat = predictions.ravel()
    if "kappa" in metric or "qwk" in metric:
        classes = np.sort(np.unique(y_true))
        nearest = np.abs(flat[:, None] - classes[None, :]).argmin(axis=1)
        return classes[nearest]
    return (flat >= 0.5).astype(int)


def _label_average(metric: str, n_classes: int) -> str:
    """Honor explicit F1/precision/recall averaging variants."""
    for average in ("micro", "macro", "weighted"):
        if average in metric:
            return average
    return "binary" if n_classes <= 2 else "macro"


def _hard_label_score(
    y_true: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> float:
    """Evaluate a maximize-style hard-label classification metric."""
    n_classes = len(np.unique(y_true))
    average = _label_average(metric, n_classes)

    if "accuracy" in metric or metric == "acc":
        return float(accuracy_score(y_true, labels))
    if "f1" in metric:
        return float(f1_score(y_true, labels, average=average, zero_division=0))
    if "precision" in metric:
        return float(precision_score(y_true, labels, average=average, zero_division=0))
    if "recall" in metric:
        return float(recall_score(y_true, labels, average=average, zero_division=0))
    if "mcc" in metric or "matthews" in metric:
        return float(matthews_corrcoef(y_true, labels))
    if "kappa" in metric or "qwk" in metric:
        weights = "quadratic" if ("quadratic" in metric or "qwk" in metric) else None
        return float(cohen_kappa_score(y_true, labels, weights=weights))
    raise ValueError(f"Unsupported classification metric: {metric}")


def _classification_score(preds: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    """Return a lower-is-better score for a classification metric."""
    raw_predictions = np.asarray(preds, dtype=float)
    raw_targets = np.asarray(y_true).ravel()
    is_ordinal_vector = (
        ("kappa" in metric or "qwk" in metric)
        and (raw_predictions.ndim == 1 or raw_predictions.shape[1] == 1)
    )
    ordinal_classes = np.unique(raw_targets)
    binary_probability = (
        is_ordinal_vector
        and len(ordinal_classes) == 2
        and np.all(np.isfinite(raw_predictions))
        and np.nanmin(raw_predictions) >= 0
        and np.nanmax(raw_predictions) <= 1
    )
    probabilities = (
        raw_predictions.ravel()
        if is_ordinal_vector
        else _normalize_probabilities(raw_predictions)
    )
    if (
        is_ordinal_vector
        and not binary_probability
        and np.issubdtype(raw_targets.dtype, np.number)
    ):
        y_encoded = raw_targets
    else:
        y_encoded = _encode_class_labels(raw_targets)

    if _contains(metric, _LOG_LOSS_NAMES) or not metric:
        labels = list(range(probabilities.shape[1])) if probabilities.ndim > 1 else None
        return float(log_loss(y_encoded, probabilities, labels=labels))

    if "brier" in metric:
        if probabilities.ndim == 1:
            return float(brier_score_loss(y_encoded, probabilities))
        one_hot = np.eye(probabilities.shape[1], dtype=float)[y_encoded]
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    if any(name in metric for name in ("auc", "roc", "gini")):
        if probabilities.ndim > 1 and probabilities.shape[1] > 2:
            score = roc_auc_score(
                y_encoded,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )
        elif probabilities.ndim > 1 and probabilities.shape[1] == 2:
            score = roc_auc_score(y_encoded, probabilities[:, 1])
        else:
            score = roc_auc_score(y_encoded, probabilities.ravel())
        return -float(score)

    labels = _hard_labels(probabilities, y_encoded, metric)
    return -_hard_label_score(y_encoded, labels, metric)


def _regression_score(preds: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    """Return a lower-is-better score in the requested regression metric."""
    predictions = np.asarray(preds, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()

    if _contains(metric, _RMSLE_NAMES):
        if np.any(y < 0):
            raise ValueError("RMSLE requires non-negative targets")
        return float(np.sqrt(mean_squared_log_error(y, np.clip(predictions, 0, None))))
    if _contains(metric, _RMSE_NAMES) or not metric:
        return float(np.sqrt(mean_squared_error(y, predictions)))
    if _contains(metric, _MSE_NAMES):
        return float(mean_squared_error(y, predictions))
    if _contains(metric, _MAE_NAMES) or "absolute" in metric:
        return float(mean_absolute_error(y, predictions))
    raise ValueError(f"Unsupported regression metric: {metric}")


def score_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
) -> float:
    """Score predictions where LOWER is better.

    Args:
        preds: Predictions array
        y_true: True labels
        problem_type: 'classification' or 'regression'
        metric_name: Metric name (log_loss, rmse, auc, etc.)

    Returns:
        Score value (negated for maximization metrics)
    """
    if preds is None or y_true is None:
        raise ValueError("preds and y_true cannot be None")

    predictions = np.asarray(preds)
    targets = np.asarray(y_true)
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: preds has {len(preds)} samples, y_true has {len(y_true)}")

    metric = (metric_name or "").lower().strip()
    if problem_type == "classification":
        return _classification_score(predictions, targets, metric or "log_loss")
    return _regression_score(predictions, targets, metric or "rmse")


def compute_oof_score(
    oof_path: Path,
    y_true: np.ndarray,
    metric_name: str = "log_loss",
) -> float:
    """Compute a LOWER-IS-BETTER, non-negative loss from OOF predictions.

    Non-negativity matters: filter_by_score_threshold uses a ratio threshold
    (best * (1 + pct)), which only makes sense for positive losses. Maximize
    metrics bounded by 1 (AUC/accuracy/kappa) are returned as 1 - metric.

    Args:
        oof_path: Path to OOF predictions file
        y_true: True labels
        metric_name: Metric name (log_loss, rmse, auc, accuracy, kappa, ...)

    Returns:
        Loss value (float("inf") when the metric cannot be computed)
    """
    oof = np.asarray(np.load(oof_path), dtype=float)
    metric = (metric_name or "log_loss").lower().strip()
    try:
        regression_metric = _contains(
            metric,
            _RMSLE_NAMES + _RMSE_NAMES + _MSE_NAMES + _MAE_NAMES,
        )
        if regression_metric:
            return _regression_score(oof, np.asarray(y_true), metric)

        classification_metric = (
            _contains(metric, _LOG_LOSS_NAMES)
            or "brier" in metric
            or any(name in metric for name in ("auc", "roc", "gini"))
            or _contains(metric, _LABEL_METRICS)
        )
        if not classification_metric:
            return float("inf")

        internal_score = _classification_score(oof, np.asarray(y_true), metric)
        if _contains(metric, _LOG_LOSS_NAMES) or "brier" in metric:
            return internal_score

        # Maximize metrics are represented internally as -metric. Convert them
        # to a non-negative loss for ratio-based weak-model filtering.
        return float(1.0 + internal_score)
    except Exception as e:
        print(f"   Warning: OOF scoring failed for {metric}: {e}")
        return float("inf")


def filter_by_score_threshold(
    prediction_pairs: dict[str, tuple[Path, Path]],
    y_true: np.ndarray,
    metric_name: str,
    model_scores: dict[str, float] | None = None,
    threshold_pct: float = 0.20,
) -> tuple[dict[str, tuple[Path, Path]], dict[str, float]]:
    """Filter models with score within X% of best.

    Computes scores on-the-fly if needed.

    Args:
        prediction_pairs: Dictionary of prediction pairs
        y_true: True labels
        metric_name: Metric name (log_loss, rmse, etc.)
        model_scores: Pre-computed CV scores (optional)
        threshold_pct: Maximum % worse than best (default 20%)

    Returns:
        Tuple of (filtered_pairs, computed_scores)
    """
    if model_scores is None:
        model_scores = {}

    computed_scores: dict[str, float] = {}
    for name, (oof_path, _) in prediction_pairs.items():
        if name in model_scores:
            computed_scores[name] = model_scores[name]
        else:
            # Compute on-the-fly and cache
            computed_scores[name] = compute_oof_score(oof_path, y_true, metric_name)
            print(f"   Computed OOF score for {name}: {computed_scores[name]:.6f}")

    # Find best score
    best_score = min(computed_scores.values()) if computed_scores else float("inf")

    # Filter by threshold
    filtered: dict[str, tuple[Path, Path]] = {}
    for name, pair in prediction_pairs.items():
        score = computed_scores.get(name, float("inf"))
        threshold = best_score * (1 + threshold_pct)
        if score <= threshold:
            filtered[name] = pair
            print(f"   OK: {name}: score {score:.6f} (within threshold)")
        else:
            print(f"   Warning: {name}: score {score:.6f} > threshold {threshold:.6f}, skipping")

    # Never starve the ensemble: the ratio threshold explodes near zero losses
    # (e.g. 1-AUC of two strong models), so if it leaves <2 models keep the
    # 2 best-scoring ones - ensemble diversity beats the threshold rule
    if len(filtered) < 2 and len(prediction_pairs) >= 2:
        best_two = sorted(computed_scores, key=computed_scores.get)[:2]
        for name in best_two:
            filtered.setdefault(name, prediction_pairs[name])
        print(f"   Filter left <2 models; keeping top-2 by OOF score: {best_two}")

    return filtered, computed_scores
