"""Stacking ensemble logic and related functions."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from ...utils.calibration import calibrate_oof_predictions, calibrate_test_predictions
from ...utils.ensemble_audit import full_ensemble_audit, post_calibrate_ensemble
from ...utils.oof_validation import print_oof_summary, validate_oof_stack
from .meta_model import constrained_meta_learner, diagnose_stacking_issues, tune_meta_model
from .scoring import filter_by_score_threshold, score_predictions
from .utils import encode_labels


def load_cv_folds(
    name: str,
    models_dir: Path,
    folds_path: Path | None,
    n_samples: int,
) -> np.ndarray | None:
    """Load per-model or global fold assignments when available.

    Args:
        name: Model name
        models_dir: Directory containing model artifacts
        folds_path: Path to folds.csv (if exists)
        n_samples: Expected number of samples

    Returns:
        Fold assignments array or None
    """
    fold_assignment_path = models_dir / f"fold_assignment_{name}.npy"
    if fold_assignment_path.exists():
        try:
            folds = np.load(fold_assignment_path)
            if len(folds) == n_samples:
                return folds
            print(f"   Warning: Fold assignment length mismatch for {name}")
        except Exception as e:
            print(f"   Warning: Failed to load fold_assignment for {name}: {e}")

    if folds_path is not None and folds_path.exists():
        try:
            folds_df = pd.read_csv(folds_path)
            if "fold" in folds_df.columns and len(folds_df) == n_samples:
                return folds_df["fold"].to_numpy()
            print("   Warning: folds.csv missing 'fold' column or length mismatch")
        except Exception as e:
            print(f"   Warning: Failed to read folds.csv: {e}")

    return None


def stack_from_prediction_pairs(
    prediction_pairs: dict[str, tuple[Path, Path]],
    y: pd.Series,
    problem_type: str,
    metric_name: str,
    models_dir: Path,
    expected_class_order: list[str] | None,
    train_ids: np.ndarray | None,
    folds_path: Path | None,
    enable_calibration: bool,
    enable_post_calibration: bool,
    n_targets: int | None,
    calibration_method: str = "auto",
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """Build stacking ensemble directly from saved OOF/Test predictions.

    Args:
        prediction_pairs: Dict of model_name -> (oof_path, test_path)
        y: Target values
        problem_type: 'classification' or 'regression'
        metric_name: Metric name for scoring
        models_dir: Directory containing model artifacts
        expected_class_order: Expected class order for classification
        train_ids: Training sample IDs
        folds_path: Path to folds.csv
        enable_calibration: Whether to calibrate base model predictions
        enable_post_calibration: Whether to calibrate ensemble output
        n_targets: Number of targets (1 for binary)
        calibration_method: Calibration method ('auto', 'isotonic', 'sigmoid')

    Returns:
        Tuple of (ensemble_dict, final_predictions) or (None, None)
    """
    if len(prediction_pairs) < 2:
        return None, None

    valid_pairs, results = validate_oof_stack(
        prediction_pairs,
        models_dir,
        train_ids=train_ids,
        expected_class_order=expected_class_order,
        folds_path=folds_path,
        strict_mode=False,
        problem_type=problem_type,
    )
    print_oof_summary(results)
    if len(valid_pairs) < 2:
        return None, None

    # Filter out weak models (more than 50% worse than best)
    print("   Filtering weak models by OOF score...")
    y_true_np = np.asarray(y)
    valid_pairs, computed_scores = filter_by_score_threshold(
        valid_pairs,
        y_true_np,
        metric_name,
        threshold_pct=0.50,  # Allow models up to 50% worse than best
    )
    if len(valid_pairs) < 2:
        print("   Warning: Not enough models after filtering weak performers")
        return None, None

    model_names = list(valid_pairs.keys())

    if problem_type == "classification":
        y_encoded, class_order = encode_labels(y, expected_class_order)
    else:
        y_encoded = np.asarray(y)
        class_order = expected_class_order

    oof_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
    calibration_summaries: list[dict[str, Any]] = []

    for name, (oof_path, test_path) in valid_pairs.items():
        try:
            oof_raw = np.load(oof_path)
            test_raw = np.load(test_path)
        except Exception as e:
            print(f"   Warning: Failed to load predictions for {name}: {e}")
            continue

        oof_raw = np.asarray(oof_raw, dtype=float)
        test_raw = np.asarray(test_raw, dtype=float)

        # Exclude models with NaN/Inf/zero-variance (CRITICAL for ensemble health)
        if np.isnan(oof_raw).any():
            nan_pct = np.isnan(oof_raw).mean() * 100
            print(f"   Warning: Excluding {name}: {nan_pct:.1f}% NaN values in OOF")
            continue
        if np.isinf(oof_raw).any():
            print(f"   Warning: Excluding {name}: Contains Inf values in OOF")
            continue
        if oof_raw.std() < 1e-10:
            print(f"   Warning: Excluding {name}: Zero variance (constant predictions)")
            continue
        if np.isnan(test_raw).any():
            nan_pct = np.isnan(test_raw).mean() * 100
            print(f"   Warning: Excluding {name}: {nan_pct:.1f}% NaN values in test predictions")
            continue
        if np.isinf(test_raw).any():
            print(f"   Warning: Excluding {name}: Contains Inf values in test predictions")
            continue

        if enable_calibration and problem_type == "classification":
            try:
                cv_folds = load_cv_folds(
                    name, models_dir, folds_path, n_samples=len(y_encoded)
                )
                result = calibrate_oof_predictions(
                    oof_path,
                    y_encoded,
                    method=calibration_method,
                    cv_folds=cv_folds,
                    save_both=True,
                )
                use_cal = (
                    result.method != "none"
                    and result.brier_after < result.brier_before
                    and result.calibrator is not None
                )

                if use_cal:
                    cal_path = models_dir / f"oof_cal_{name}.npy"
                    oof_preds = np.load(cal_path)
                    test_preds = calibrate_test_predictions(
                        test_path, result.calibrator, result.method
                    )
                else:
                    oof_preds = oof_raw
                    test_preds = test_raw

                calibration_summaries.append(
                    {
                        "model": name,
                        "method": result.method if use_cal else "none",
                        "brier_before": result.brier_before,
                        "brier_after": result.brier_after,
                        "improvement_pct": result.improvement_pct if use_cal else 0.0,
                    }
                )
            except Exception as e:
                print(f"   Warning: Calibration failed for {name}: {e}")
                oof_preds = oof_raw
                test_preds = test_raw
        else:
            oof_preds = oof_raw
            test_preds = test_raw

        if n_targets == 1 and oof_preds.ndim == 2 and oof_preds.shape[1] == 2:
            oof_preds = oof_preds[:, 1]
        if n_targets == 1 and test_preds.ndim == 2 and test_preds.shape[1] == 2:
            test_preds = test_preds[:, 1]

        if oof_preds.ndim == 2 and oof_preds.shape[1] == 1:
            oof_preds = oof_preds.squeeze()
        if test_preds.ndim == 2 and test_preds.shape[1] == 1:
            test_preds = test_preds.squeeze()

        oof_list.append(oof_preds)
        test_list.append(test_preds)

    # Validate shapes after normalization (CRITICAL: prevents inhomogeneous array errors)
    if oof_list:
        oof_shapes = {tuple(o.shape) for o in oof_list}
        test_shapes = {tuple(t.shape) for t in test_list}
        model_names = list(valid_pairs.keys())[: len(oof_list)]

        if len(oof_shapes) > 1 or len(test_shapes) > 1:
            print("   Warning: Shape mismatch detected after normalization:")
            print(f"      OOF shapes: {oof_shapes}")
            print(f"      Test shapes: {test_shapes}")

            # Find the most common shape and keep only compatible models
            shape_counts = Counter(tuple(o.shape) for o in oof_list)
            target_shape = shape_counts.most_common(1)[0][0]
            print(f"      Keeping models with shape: {target_shape}")

            valid_idx = [i for i, o in enumerate(oof_list) if tuple(o.shape) == target_shape]
            oof_list = [oof_list[i] for i in valid_idx]
            test_list = [test_list[i] for i in valid_idx]
            kept_names = [model_names[i] for i in valid_idx] if len(model_names) > max(valid_idx) else []
            print(f"      Kept {len(oof_list)} compatible models: {kept_names}")

    if len(oof_list) < 2:
        return None, None

    oof_stack = np.stack(oof_list, axis=0)
    test_stack = np.stack(test_list, axis=0)

    if oof_list[0].ndim == 1:
        meta_X = np.column_stack(oof_list)
        meta_X_test = np.column_stack(test_list)
        binary_single_col = True
        n_features_per_model = 1
    else:
        meta_X = np.concatenate(oof_list, axis=1)
        meta_X_test = np.concatenate(test_list, axis=1)
        binary_single_col = False
        n_features_per_model = oof_list[0].shape[1]

    avg_oof = np.average(oof_stack, axis=0)
    avg_score = score_predictions(avg_oof, y_encoded, problem_type, metric_name)
    print(f"   [META] Simple average: {avg_score:.6f}")

    weights_constrained, constrained_score = constrained_meta_learner(
        oof_stack, y_encoded, problem_type, metric_name
    )
    print(f"   [META] Constrained: {constrained_score:.6f}")

    meta_score = float("inf")
    meta_oof_preds = None
    meta_model = None
    meta_metric_name = metric_name  # Default to passed metric
    try:
        n_classes = len(np.unique(y_encoded)) if problem_type == "classification" else None
        meta_model, meta_metric_name = tune_meta_model(meta_X, y_encoded, problem_type, n_classes)
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            meta_oof_preds = cross_val_predict(
                meta_model, meta_X, y_encoded, cv=cv, method="predict_proba"
            )
            if binary_single_col and meta_oof_preds.ndim > 1:
                meta_oof_preds = meta_oof_preds[:, 1]
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            meta_oof_preds = cross_val_predict(meta_model, meta_X, y_encoded, cv=cv)

        meta_score = score_predictions(
            meta_oof_preds, y_encoded, problem_type, metric_name
        )
        print(f"   [META] Meta-model: {meta_score:.6f}")

        # Run stacking diagnostics to detect potential issues
        stacking_diagnostics = diagnose_stacking_issues(
            meta_model, model_names, meta_X, y_encoded
        )
        if not stacking_diagnostics["is_healthy"]:
            print("   Warning: Stacking issues detected - consider checking data alignment")
    except Exception as e:
        print(f"   Warning: Meta-model evaluation failed: {e}")

    scores = {
        "average": avg_score,
        "constrained": constrained_score,
        "meta": meta_score,
    }
    best_method = min(scores, key=scores.get)

    # Explicit fallback warning when meta-model is significantly worse
    if meta_score != float("inf") and avg_score != float("inf"):
        if meta_score > avg_score * 1.05:
            pct_worse = (meta_score / avg_score - 1) * 100
            print(f"   Warning: Meta-model is {pct_worse:.1f}% worse than simple average!")
            print(f"      Meta: {meta_score:.6f} vs Avg: {avg_score:.6f}")
            print("      -> Forcing fallback to simple average")
            best_method = "average"
        elif meta_score > avg_score:
            print("   Info: Meta-model slightly worse than average, using average")
            best_method = "average"

    print(f"   [META] Best method: {best_method}")

    if best_method == "meta" and meta_model is not None:
        meta_model.fit(meta_X, y_encoded)
        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            final_test_preds = meta_model.predict_proba(meta_X_test)
            if binary_single_col and final_test_preds.ndim > 1:
                final_test_preds = final_test_preds[:, 1]
        else:
            final_test_preds = meta_model.predict(meta_X_test)
        selected_oof = meta_oof_preds if meta_oof_preds is not None else avg_oof
        selected_weights = None
    elif best_method == "constrained":
        final_test_preds = np.average(test_stack, axis=0, weights=weights_constrained)
        selected_oof = np.average(oof_stack, axis=0, weights=weights_constrained)
        selected_weights = weights_constrained
    else:
        final_test_preds = np.average(test_stack, axis=0, weights=np.ones(len(oof_list)))
        selected_oof = avg_oof
        selected_weights = np.ones(len(oof_list)) / len(oof_list)

    selected_score = score_predictions(
        selected_oof, y_encoded, problem_type, metric_name
    )

    if problem_type == "classification":
        final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
        if final_test_preds.ndim > 1 and final_test_preds.shape[1] > 1:
            final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

    calibration_info = {}
    if problem_type == "classification" and enable_post_calibration:
        final_test_preds, calibration_info = post_calibrate_ensemble(
            selected_oof, final_test_preds, y_encoded, method=calibration_method
        )

    audit_weights = selected_weights
    if audit_weights is None and meta_model is not None and hasattr(meta_model, "coef_"):
        coefs = meta_model.coef_
        if coefs.ndim == 2:
            coefs = np.mean(np.abs(coefs), axis=0)
        weights = []
        for i in range(len(oof_list)):
            start = i * n_features_per_model
            end = start + n_features_per_model
            weights.append(float(np.mean(np.abs(coefs[start:end]))))
        audit_weights = np.array(weights)
        if audit_weights.sum() > 0:
            audit_weights = audit_weights / audit_weights.sum()

    audit = full_ensemble_audit(
        model_names,
        oof_stack,
        y_encoded,
        problem_type,
        metric_name,
        weights=audit_weights,
        calibration_info=calibration_info,
    )

    if calibration_summaries:
        print("   [CAL] Base model calibration summary:")
        for summary in calibration_summaries:
            if summary["method"] == "none":
                print(f"      {summary['model']}: no improvement")
            else:
                print(
                    f"      {summary['model']}: "
                    f"{summary['brier_before']:.4f} -> {summary['brier_after']:.4f} "
                    f"({summary['improvement_pct']:.2f}%)"
                )

    if audit.warnings:
        print("   [AUDIT] Warnings:")
        for warning in audit.warnings:
            print(f"      - {warning}")

    ensemble = {
        "meta_model": meta_model if best_method == "meta" else None,
        "base_model_names": model_names,
        "stacking_method": best_method,
        "weights": audit_weights.tolist() if audit_weights is not None else None,
        "oof_score": selected_score,
        "calibration": calibration_summaries,
        "audit": {
            "dominant_model": audit.dominant_model,
            "dominance_weight": audit.dominance_weight,
            "warnings": audit.warnings,
            "notes": audit.notes,
            "calibration": audit.calibration,
        },
        "class_order": class_order,
    }

    return ensemble, final_test_preds
