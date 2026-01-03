"""Utilities for auditing ensemble behavior and stability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from ..core.config import is_metric_minimization
from .calibration import compute_brier_score, isotonic_calibration, platt_scaling


@dataclass
class EnsembleAuditResult:
    """Summary of ensemble audit checks."""

    weights: dict[str, float] = field(default_factory=dict)
    influence: dict[str, float] = field(default_factory=dict)
    dominant_model: str | None = None
    dominance_weight: float | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)


def _compute_metric_score(
    y_true: np.ndarray, preds: np.ndarray, problem_type: str, metric_name: str
) -> float:
    """Compute a score where LOWER is always better."""
    metric = (metric_name or "").lower()
    if not metric:
        metric = "log_loss" if problem_type == "classification" else "rmse"

    if problem_type == "classification":
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = preds / preds.sum(axis=1, keepdims=True)

        if "auc" in metric:
            from sklearn.metrics import roc_auc_score

            if preds.ndim > 1 and preds.shape[1] > 1:
                score = roc_auc_score(y_true, preds, multi_class="ovr", average="weighted")
            else:
                score = roc_auc_score(y_true, preds)
        else:
            score = log_loss(y_true, preds)
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


def check_weight_dominance(
    weights: dict[str, float], threshold: float = 0.8
) -> tuple[str | None, float | None]:
    """Detect if a single model dominates the ensemble weights."""
    if not weights:
        return None, None
    dominant_model = max(weights, key=weights.get)
    dominance_weight = float(weights[dominant_model])
    if dominance_weight >= threshold:
        return dominant_model, dominance_weight
    return None, None


def analyze_model_influence(
    model_names: list[str],
    oof_stack: np.ndarray,
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Estimate each model's impact via leave-one-out."""
    n_models = len(model_names)
    if n_models == 0:
        return {}

    if weights is None:
        weights = np.ones(n_models) / n_models
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_models) / n_models

    baseline_preds = np.average(oof_stack, axis=0, weights=weights)
    baseline_score = _compute_metric_score(y_true, baseline_preds, problem_type, metric_name)

    influence: dict[str, float] = {}
    for idx, name in enumerate(model_names):
        reduced_stack = np.delete(oof_stack, idx, axis=0)
        reduced_weights = np.delete(weights, idx)
        if reduced_weights.sum() <= 0:
            reduced_weights = np.ones_like(reduced_weights) / len(reduced_weights)
        else:
            reduced_weights = reduced_weights / reduced_weights.sum()

        preds = np.average(reduced_stack, axis=0, weights=reduced_weights)
        score = _compute_metric_score(y_true, preds, problem_type, metric_name)
        influence[name] = score - baseline_score

    return influence


def _apply_calibrator(
    preds: np.ndarray, calibrator: Any, method: str
) -> np.ndarray:
    """Apply a fitted calibrator to prediction arrays."""
    if preds.ndim == 1:
        if method == "platt":
            return calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
        return calibrator[0].predict(preds)

    n_samples, n_classes = preds.shape
    calibrated = np.zeros_like(preds)
    for c in range(n_classes):
        if method == "platt":
            calibrated[:, c] = calibrator[c].predict_proba(preds[:, c].reshape(-1, 1))[:, 1]
        else:
            calibrated[:, c] = calibrator[c].predict(preds[:, c])

    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    return calibrated / row_sums


def post_calibrate_ensemble(
    oof_preds: np.ndarray,
    test_preds: np.ndarray,
    y_true: np.ndarray,
    method: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally calibrate final ensemble predictions."""
    info: dict[str, Any] = {
        "method": "none",
        "brier_before": None,
        "brier_after": None,
        "improvement_pct": 0.0,
    }

    brier_before = compute_brier_score(oof_preds, y_true)
    info["brier_before"] = brier_before

    if method not in {"auto", "platt", "isotonic"}:
        return test_preds, info

    best_method = "none"
    best_brier = brier_before
    best_calibrator = None

    if method in {"auto", "platt"}:
        try:
            oof_platt, cal_platt = platt_scaling(oof_preds, y_true)
            brier_platt = compute_brier_score(oof_platt, y_true)
            if brier_platt < best_brier:
                best_brier = brier_platt
                best_method = "platt"
                best_calibrator = cal_platt
        except Exception:
            pass

    if method in {"auto", "isotonic"}:
        try:
            oof_iso, cal_iso = isotonic_calibration(oof_preds, y_true)
            brier_iso = compute_brier_score(oof_iso, y_true)
            if brier_iso < best_brier:
                best_brier = brier_iso
                best_method = "isotonic"
                best_calibrator = cal_iso
        except Exception:
            pass

    info["method"] = best_method
    info["brier_after"] = best_brier
    if brier_before > 0:
        info["improvement_pct"] = 100 * (brier_before - best_brier) / brier_before

    if best_method == "none" or best_calibrator is None:
        return test_preds, info

    calibrated_test = _apply_calibrator(test_preds, best_calibrator, best_method)
    return calibrated_test, info


def full_ensemble_audit(
    model_names: list[str],
    oof_stack: np.ndarray,
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
    weights: np.ndarray | None = None,
    calibration_info: dict[str, Any] | None = None,
    dominance_threshold: float = 0.8,
) -> EnsembleAuditResult:
    """Run a full audit of ensemble behavior."""
    audit = EnsembleAuditResult()

    if weights is not None:
        audit.weights = dict(zip(model_names, weights))
        dominant_model, dominance_weight = check_weight_dominance(
            audit.weights, threshold=dominance_threshold
        )
        audit.dominant_model = dominant_model
        audit.dominance_weight = dominance_weight
        if dominant_model is not None:
            audit.warnings.append(
                f"Dominance detected: {dominant_model} weight={dominance_weight:.2f}"
            )

    audit.influence = analyze_model_influence(
        model_names, oof_stack, y_true, problem_type, metric_name, weights=weights
    )
    for name, delta in audit.influence.items():
        if delta < 0:
            audit.warnings.append(f"Model {name} may add noise (delta={delta:.6f})")

    if calibration_info:
        audit.calibration = calibration_info
        if calibration_info.get("method") != "none" and calibration_info.get(
            "improvement_pct", 0.0
        ) > 0:
            audit.notes.append(
                f"Post-calibration improved Brier by {calibration_info['improvement_pct']:.2f}%"
            )

    return audit
