"""
Metric-aware postprocessing tuned on OOF predictions.

For metrics scored on hard labels (accuracy, F1, kappa/QWK, MCC), converting
probabilities/continuous predictions with a fixed 0.5 threshold or plain
argmax leaves score on the table. These helpers tune the decision rule on
OOF predictions and apply it to test predictions.

Pure numpy/sklearn/scipy - unit-testable without the workflow.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def metric_label_kind(metric_name: str | None) -> str | None:
    """
    Which label-decision rule a metric needs.

    Returns:
        "qwk" for (quadratic weighted) kappa-style ordinal metrics,
        "threshold" for other hard-label metrics (accuracy/F1/MCC/...),
        None for probability/regression metrics (no postprocessing).
    """
    lower = (metric_name or "").lower()
    if any(k in lower for k in ("kappa", "qwk")):
        return "qwk"
    if any(k in lower for k in ("accuracy", "f1", "precision", "recall", "mcc")):
        return "threshold"
    return None


def _label_metric_fn(metric_name: str):
    """Scorer (higher is better) for hard-label metrics."""
    lower = (metric_name or "").lower()
    def average(y: np.ndarray) -> str:
        for variant in ("micro", "macro", "weighted"):
            if variant in lower:
                return variant
        return "binary" if len(np.unique(y)) <= 2 else "macro"

    if "f1" in lower:
        return lambda y, p: f1_score(y, p, average=average(y), zero_division=0)
    if "precision" in lower:
        return lambda y, p: precision_score(y, p, average=average(y), zero_division=0)
    if "recall" in lower:
        return lambda y, p: recall_score(y, p, average=average(y), zero_division=0)
    if "mcc" in lower or "matthews" in lower:
        return matthews_corrcoef
    if "kappa" in lower or "qwk" in lower:
        weights = "quadratic" if ("quadratic" in lower or "qwk" in lower) else None
        return lambda y, p: cohen_kappa_score(y, p, weights=weights)
    return accuracy_score


def tune_binary_threshold(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    metric_name: str = "accuracy",
) -> tuple[float, float, float]:
    """
    Grid-search the decision threshold on OOF predictions.

    Args:
        oof_preds: 1-D OOF probabilities/scores
        y_true: 1-D binary labels (arbitrary original values are supported)
        metric_name: Hard-label metric to maximize

    Returns:
        (best_threshold, best_score, baseline_score_at_0.5)
    """
    oof = np.asarray(oof_preds, dtype=float).ravel()
    y_original = np.asarray(y_true).ravel()
    classes = np.unique(y_original)
    if len(classes) != 2:
        raise ValueError(f"Binary threshold tuning requires 2 classes, got {len(classes)}")
    y = (y_original == classes[-1]).astype(int)
    metric_fn = _label_metric_fn(metric_name)

    baseline = float(metric_fn(y, (oof >= 0.5).astype(int)))
    best_threshold, best_score = 0.5, baseline

    if np.all((oof >= 0) & (oof <= 1)):
        thresholds = np.linspace(0.05, 0.95, 181)
    else:
        thresholds = np.linspace(float(np.min(oof)), float(np.max(oof)), 181)

    for threshold in thresholds:
        score = float(metric_fn(y, (oof >= threshold).astype(int)))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    return best_threshold, best_score, baseline


def apply_rounding(preds: np.ndarray, boundaries: list[float], classes: np.ndarray) -> np.ndarray:
    """Map continuous ordinal predictions to classes via boundary digitization."""
    preds = np.asarray(preds, dtype=float).ravel()
    idx = np.digitize(preds, boundaries)
    return classes[np.clip(idx, 0, len(classes) - 1)]


def tune_qwk_rounding(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
) -> tuple[list[float], float, float, np.ndarray]:
    """
    Optimize rounding boundaries for quadratic weighted kappa (OptimizedRounder).

    Args:
        oof_preds: 1-D continuous OOF predictions on the label scale
        y_true: 1-D integer ordinal labels

    Returns:
        (boundaries, tuned_qwk, baseline_qwk_midpoint_rule, classes)
    """
    oof = np.asarray(oof_preds, dtype=float).ravel()
    y = np.asarray(y_true).ravel().astype(int)
    classes = np.unique(y)

    if len(classes) < 3:
        # Binary: rounding boundaries degenerate to a single threshold
        threshold, tuned, base = tune_binary_threshold(oof, y, "quadratic_weighted_kappa")
        return [threshold], tuned, base, classes

    def rule_score(boundaries: list[float]) -> float:
        labels = apply_rounding(oof, sorted(boundaries), classes)
        return float(cohen_kappa_score(y, labels, weights="quadratic"))

    # Baseline = midpoint boundaries: the SAME rule family that gets applied,
    # so the reported scores are always the true scores of the returned rule
    # (plain np.rint is a different rule and can even emit invalid classes for
    # non-contiguous label sets)
    midpoint_boundaries = ((classes[:-1] + classes[1:]) / 2.0).tolist()
    baseline = rule_score(midpoint_boundaries)

    result = minimize(
        lambda b: -rule_score(b.tolist()),
        np.asarray(midpoint_boundaries),
        method="Nelder-Mead",
        options={"maxiter": 500},
    )
    boundaries = sorted(result.x.tolist())
    tuned = rule_score(boundaries)

    if tuned < baseline:
        # Never do worse than the default midpoint rule
        boundaries, tuned = midpoint_boundaries, baseline

    return boundaries, tuned, baseline, classes


def labels_from_oof_tuning(
    test_preds: np.ndarray,
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    metric_name: str,
) -> tuple[np.ndarray, dict]:
    """
    Convert 1-D test predictions to hard labels using a decision rule tuned on OOF.

    Args:
        test_preds: 1-D test probabilities/continuous predictions
        oof_preds: 1-D OOF predictions aligned with y_true
        y_true: Training labels
        metric_name: Target hard-label metric

    Returns:
        (labels, info dict with the tuned rule and OOF scores)
    """
    kind = metric_label_kind(metric_name)
    test = np.asarray(test_preds, dtype=float).ravel()
    y = np.asarray(y_true).ravel()
    n_classes = len(np.unique(y))

    if kind == "qwk" and n_classes > 2:
        boundaries, tuned, baseline, classes = tune_qwk_rounding(oof_preds, y)
        labels = apply_rounding(test, boundaries, classes)
        info = {
            "rule": "qwk_rounding",
            "boundaries": [round(b, 4) for b in boundaries],
            "oof_score_tuned": round(tuned, 6),
            "oof_score_baseline": round(baseline, 6),
        }
        return labels, info

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"Threshold postprocessing requires exactly 2 classes, got {len(classes)}"
        )

    threshold, tuned, baseline = tune_binary_threshold(oof_preds, y, metric_name)
    labels = classes[(test >= threshold).astype(int)]
    info = {
        "rule": "binary_threshold",
        "threshold": round(threshold, 4),
        "oof_score_tuned": round(tuned, 6),
        "oof_score_baseline": round(baseline, 6),
    }
    return labels, info
