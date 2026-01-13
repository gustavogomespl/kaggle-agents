"""Scoring functions for ensemble evaluation."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ...core.config import is_metric_minimization


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
    metric = (metric_name or "").lower()
    if not metric:
        metric = "log_loss" if problem_type == "classification" else "rmse"

    # Input validation
    if preds is None or y_true is None:
        raise ValueError("preds and y_true cannot be None")

    # Ensure consistent lengths
    if len(preds) != len(y_true):
        raise ValueError(f"Length mismatch: preds has {len(preds)} samples, y_true has {len(y_true)}")

    # Ensure y_true is integer labels for classification
    if problem_type == "classification":
        # Convert y_true to integer labels if needed
        if y_true.dtype.kind == 'f':  # float type
            # Check if y_true looks like probabilities (values between 0 and 1)
            if np.all((y_true >= 0) & (y_true <= 1)):
                if y_true.ndim > 1 and y_true.shape[1] > 1:
                    # 2D probability array - convert to class labels
                    y_true = np.argmax(y_true, axis=1)
                # 1D - might be probabilities or actual numeric labels
                # If values are close to integers 0,1,2..., treat as labels
                elif np.allclose(y_true, y_true.astype(int)):
                    y_true = y_true.astype(int)
                else:
                    raise ValueError(
                        f"y_true appears to be probabilities, not class labels. "
                        f"Values: {y_true[:5]}"
                    )
            else:
                # Float values outside [0,1] - likely actual labels, convert to int
                y_true = y_true.astype(int)

        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = preds / preds.sum(axis=1, keepdims=True)

        # Determine number of classes for multiclass log_loss
        if preds.ndim > 1 and preds.shape[1] > 1:
            n_classes = preds.shape[1]
            labels = list(range(n_classes))
        else:
            labels = None

        if "auc" in metric:
            if preds.ndim > 1 and preds.shape[1] > 1:
                # Ensure y_true has all classes represented for multiclass AUC
                unique_classes = np.unique(y_true)
                if len(unique_classes) < 2:
                    # Can't compute AUC with single class - fall back to log_loss
                    score = log_loss(y_true, preds, labels=labels)
                else:
                    score = roc_auc_score(y_true, preds, multi_class="ovr", average="weighted")
            else:
                score = roc_auc_score(y_true, preds)
        elif "acc" in metric:
            if preds.ndim > 1 and preds.shape[1] > 1:
                pred_labels = np.argmax(preds, axis=1)
            else:
                pred_labels = (preds >= 0.5).astype(int)
            score = accuracy_score(y_true, pred_labels)
        else:
            # log_loss for classification
            score = log_loss(y_true, preds, labels=labels)
    else:
        if preds.ndim > 1:
            preds = preds.ravel()
        if "mae" in metric:
            score = mean_absolute_error(y_true, preds)
        elif "mse" in metric:
            score = mean_squared_error(y_true, preds)
        else:
            score = np.sqrt(mean_squared_error(y_true, preds))

    if is_metric_minimization(metric):
        return score
    return -score


def compute_oof_score(
    oof_path: Path,
    y_true: np.ndarray,
    metric_name: str = "log_loss",
) -> float:
    """Compute model score via OOF predictions.

    Args:
        oof_path: Path to OOF predictions file
        y_true: True labels
        metric_name: Metric name (log_loss, rmse, etc.)

    Returns:
        Score value
    """
    oof = np.load(oof_path)
    # Clip and normalize for log_loss
    oof = np.clip(oof, 1e-15, 1 - 1e-15)
    if oof.ndim > 1 and oof.shape[1] > 1:
        oof = oof / oof.sum(axis=1, keepdims=True)

    if metric_name == "log_loss":
        return log_loss(y_true, oof)
    if metric_name in ["rmse", "mse", "mean_squared_error"]:
        return np.sqrt(mean_squared_error(y_true, oof.ravel() if oof.ndim > 1 else oof))
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

    return filtered, computed_scores
