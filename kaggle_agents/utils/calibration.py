"""Probability calibration utilities for ensemble predictions.

This module provides Platt scaling and isotonic regression calibration
to improve probability estimates before stacking ensemble.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold


@dataclass
class CalibrationResult:
    """Result of probability calibration."""

    model_name: str
    method: Literal["platt", "isotonic", "none"]
    brier_before: float
    brier_after: float
    logloss_before: float
    logloss_after: float
    improvement_pct: float
    calibrator: LogisticRegression | list[IsotonicRegression] | None = None


def compute_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Brier score (multiclass extension).

    Lower is better. Range: [0, 2] for multiclass.

    Args:
        probs: Predicted probabilities (n_samples,) or (n_samples, n_classes)
        y_true: True labels

    Returns:
        Brier score
    """
    if probs.ndim == 1:
        # Binary classification
        return brier_score_loss(y_true, probs)

    # Multiclass: average Brier score across classes
    n_classes = probs.shape[1]
    total_brier = 0.0

    for c in range(n_classes):
        y_binary = (y_true == c).astype(float)
        total_brier += brier_score_loss(y_binary, probs[:, c])

    return total_brier / n_classes


def _safe_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute log loss with proper clipping and normalization."""
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    if probs.ndim > 1 and probs.shape[1] > 1:
        probs = probs / probs.sum(axis=1, keepdims=True)
    return log_loss(y_true, probs)


def platt_scaling(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    cv_folds: np.ndarray | None = None,
    n_cv_splits: int = 5,
) -> tuple[np.ndarray, LogisticRegression | list[LogisticRegression]]:
    """Apply Platt scaling (logistic calibration) to OOF probabilities.

    For multiclass: applies one-vs-rest Platt scaling per class.

    Args:
        oof_probs: OOF probabilities (n_samples, n_classes) or (n_samples,)
        y_true: True labels
        cv_folds: Optional fold assignments for proper OOF calibration
        n_cv_splits: Number of CV splits if cv_folds not provided

    Returns:
        Tuple of (calibrated_probs, fitted_calibrator(s))
    """
    if oof_probs.ndim == 1:
        # Binary classification
        return _platt_scaling_binary(oof_probs, y_true, cv_folds, n_cv_splits)
    # Multiclass: per-class calibration
    return _platt_scaling_multiclass(oof_probs, y_true, cv_folds, n_cv_splits)


def _platt_scaling_binary(
    probs: np.ndarray,
    y_true: np.ndarray,
    cv_folds: np.ndarray | None,
    n_cv_splits: int,
) -> tuple[np.ndarray, LogisticRegression]:
    """Platt scaling for binary classification."""
    calibrated = np.zeros_like(probs)

    if cv_folds is not None:
        # Use provided folds for OOF calibration
        unique_folds = np.unique(cv_folds)
        for fold in unique_folds:
            train_mask = cv_folds != fold
            val_mask = cv_folds == fold

            calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            calibrator.fit(probs[train_mask].reshape(-1, 1), y_true[train_mask])
            calibrated[val_mask] = calibrator.predict_proba(probs[val_mask].reshape(-1, 1))[:, 1]
    else:
        # Use KFold for OOF calibration
        kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(probs):
            calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            calibrator.fit(probs[train_idx].reshape(-1, 1), y_true[train_idx])
            calibrated[val_idx] = calibrator.predict_proba(probs[val_idx].reshape(-1, 1))[:, 1]

    # Fit final calibrator on all data for test predictions
    final_calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    final_calibrator.fit(probs.reshape(-1, 1), y_true)

    return calibrated, final_calibrator


def _platt_scaling_multiclass(
    probs: np.ndarray,
    y_true: np.ndarray,
    cv_folds: np.ndarray | None,
    n_cv_splits: int,
) -> tuple[np.ndarray, list[LogisticRegression]]:
    """Platt scaling for multiclass classification (per-class)."""
    n_samples, n_classes = probs.shape
    calibrated = np.zeros_like(probs)
    calibrators: list[LogisticRegression] = []

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        probs_c = probs[:, c]

        if cv_folds is not None:
            unique_folds = np.unique(cv_folds)
            for fold in unique_folds:
                train_mask = cv_folds != fold
                val_mask = cv_folds == fold

                calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                calibrator.fit(probs_c[train_mask].reshape(-1, 1), y_binary[train_mask])
                calibrated[val_mask, c] = calibrator.predict_proba(probs_c[val_mask].reshape(-1, 1))[:, 1]
        else:
            kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(probs_c):
                calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                calibrator.fit(probs_c[train_idx].reshape(-1, 1), y_binary[train_idx])
                calibrated[val_idx, c] = calibrator.predict_proba(probs_c[val_idx].reshape(-1, 1))[:, 1]

        # Fit final calibrator for test predictions
        final_calibrator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        final_calibrator.fit(probs_c.reshape(-1, 1), y_binary)
        calibrators.append(final_calibrator)

    # Renormalize rows to sum to 1
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)  # Avoid division by zero
    calibrated = calibrated / row_sums

    return calibrated, calibrators


def isotonic_calibration(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    cv_folds: np.ndarray | None = None,
    n_cv_splits: int = 5,
) -> tuple[np.ndarray, list[IsotonicRegression]]:
    """Apply isotonic regression calibration to OOF probabilities.

    Isotonic regression is non-parametric and can capture more complex
    probability mappings than Platt scaling.

    Args:
        oof_probs: OOF probabilities (n_samples, n_classes) or (n_samples,)
        y_true: True labels
        cv_folds: Optional fold assignments for proper OOF calibration
        n_cv_splits: Number of CV splits if cv_folds not provided

    Returns:
        Tuple of (calibrated_probs, fitted_calibrators_per_class)
    """
    if oof_probs.ndim == 1:
        oof_probs = oof_probs.reshape(-1, 1)

    n_samples, n_classes = oof_probs.shape
    calibrated = np.zeros_like(oof_probs)
    calibrators: list[IsotonicRegression] = []

    for c in range(n_classes):
        y_binary = (y_true == c).astype(float) if n_classes > 1 else y_true.astype(float)
        probs_c = oof_probs[:, c]

        if cv_folds is not None:
            unique_folds = np.unique(cv_folds)
            for fold in unique_folds:
                train_mask = cv_folds != fold
                val_mask = cv_folds == fold

                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(probs_c[train_mask], y_binary[train_mask])
                calibrated[val_mask, c] = iso.predict(probs_c[val_mask])
        else:
            kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(probs_c):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(probs_c[train_idx], y_binary[train_idx])
                calibrated[val_idx, c] = iso.predict(probs_c[val_idx])

        # Fit final calibrator for test predictions
        final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        final_iso.fit(probs_c, y_binary)
        calibrators.append(final_iso)

    # Renormalize rows to sum to 1 for multiclass
    if n_classes > 1:
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        calibrated = calibrated / row_sums

    # Squeeze if binary
    if n_classes == 1:
        calibrated = calibrated.squeeze()

    return calibrated, calibrators


def calibrate_oof_predictions(
    oof_path: Path,
    y_true: np.ndarray,
    method: Literal["platt", "isotonic", "auto"] = "auto",
    cv_folds: np.ndarray | None = None,
    save_both: bool = True,
) -> CalibrationResult:
    """Calibrate OOF predictions and optionally save both raw and calibrated versions.

    Args:
        oof_path: Path to oof_{model_name}.npy
        y_true: True labels
        method: Calibration method ("platt", "isotonic", or "auto" to pick best)
        cv_folds: Fold assignments for proper OOF calibration
        save_both: If True, saves oof_raw_{name}.npy and oof_cal_{name}.npy

    Returns:
        CalibrationResult with before/after metrics
    """
    name = oof_path.stem.replace("oof_", "", 1)
    oof_raw = np.load(oof_path)

    # Compute metrics before calibration
    brier_before = compute_brier_score(oof_raw, y_true)
    logloss_before = _safe_log_loss(y_true, oof_raw)

    # Try both methods if auto
    if method == "auto":
        # Try Platt
        try:
            oof_platt, cal_platt = platt_scaling(oof_raw, y_true, cv_folds)
            brier_platt = compute_brier_score(oof_platt, y_true)
        except Exception:
            brier_platt = float("inf")
            oof_platt = None
            cal_platt = None

        # Try Isotonic
        try:
            oof_iso, cal_iso = isotonic_calibration(oof_raw, y_true, cv_folds)
            brier_iso = compute_brier_score(oof_iso, y_true)
        except Exception:
            brier_iso = float("inf")
            oof_iso = None
            cal_iso = None

        # Pick the better one
        if brier_platt <= brier_iso and oof_platt is not None:
            oof_cal, calibrator = oof_platt, cal_platt
            selected_method: Literal["platt", "isotonic", "none"] = "platt"
        elif oof_iso is not None:
            oof_cal, calibrator = oof_iso, cal_iso
            selected_method = "isotonic"
        else:
            # Both failed, use raw
            oof_cal = oof_raw
            calibrator = None
            selected_method = "none"
    elif method == "platt":
        oof_cal, calibrator = platt_scaling(oof_raw, y_true, cv_folds)
        selected_method = "platt"
    else:
        oof_cal, calibrator = isotonic_calibration(oof_raw, y_true, cv_folds)
        selected_method = "isotonic"

    # Compute metrics after calibration
    brier_after = compute_brier_score(oof_cal, y_true)
    logloss_after = _safe_log_loss(y_true, oof_cal)

    # Calculate improvement
    improvement_pct = 100 * (brier_before - brier_after) / brier_before if brier_before > 0 else 0

    # Save both versions if requested
    if save_both:
        models_dir = oof_path.parent
        raw_path = models_dir / f"oof_raw_{name}.npy"
        cal_path = models_dir / f"oof_cal_{name}.npy"

        np.save(raw_path, oof_raw)
        np.save(cal_path, oof_cal)

    return CalibrationResult(
        model_name=name,
        method=selected_method,
        brier_before=brier_before,
        brier_after=brier_after,
        logloss_before=logloss_before,
        logloss_after=logloss_after,
        improvement_pct=improvement_pct,
        calibrator=calibrator,
    )


def calibrate_test_predictions(
    test_path: Path,
    calibrator: LogisticRegression | list[LogisticRegression] | list[IsotonicRegression],
    method: Literal["platt", "isotonic"],
) -> np.ndarray:
    """Apply fitted calibrator to test predictions.

    Args:
        test_path: Path to test_{model_name}.npy
        calibrator: Fitted calibrator from calibrate_oof_predictions
        method: Calibration method used

    Returns:
        Calibrated test predictions
    """
    test_raw = np.load(test_path)

    if test_raw.ndim == 1:
        # Binary classification
        if method == "platt":
            cal = calibrator  # type: ignore
            calibrated = cal.predict_proba(test_raw.reshape(-1, 1))[:, 1]
        else:
            cal = calibrator[0]  # type: ignore
            calibrated = cal.predict(test_raw)
    else:
        # Multiclass
        n_samples, n_classes = test_raw.shape
        calibrated = np.zeros_like(test_raw)

        for c in range(n_classes):
            if method == "platt":
                cal = calibrator[c]  # type: ignore
                calibrated[:, c] = cal.predict_proba(test_raw[:, c].reshape(-1, 1))[:, 1]
            else:
                cal = calibrator[c]  # type: ignore
                calibrated[:, c] = cal.predict(test_raw[:, c])

        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        calibrated = calibrated / row_sums

    return calibrated


def evaluate_calibration_quality(
    oof_raw: np.ndarray,
    oof_cal: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Evaluate calibration quality improvement.

    Args:
        oof_raw: Raw OOF predictions
        oof_cal: Calibrated OOF predictions
        y_true: True labels

    Returns:
        Dictionary with brier_raw, brier_cal, logloss_raw, logloss_cal, improvement_pct
    """
    brier_raw = compute_brier_score(oof_raw, y_true)
    brier_cal = compute_brier_score(oof_cal, y_true)
    logloss_raw = _safe_log_loss(y_true, oof_raw)
    logloss_cal = _safe_log_loss(y_true, oof_cal)

    improvement_pct = 100 * (brier_raw - brier_cal) / brier_raw if brier_raw > 0 else 0

    return {
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "logloss_raw": logloss_raw,
        "logloss_cal": logloss_cal,
        "improvement_pct": improvement_pct,
    }
