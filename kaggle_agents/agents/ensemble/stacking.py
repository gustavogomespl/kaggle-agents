"""Stacking ensemble logic and related functions."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...utils.calibration import calibrate_oof_predictions, calibrate_test_predictions
from ...utils.ensemble_audit import full_ensemble_audit, post_calibrate_ensemble
from ...utils.oof_validation import print_oof_summary, validate_oof_stack
from .meta_model import (
    cross_fitted_meta_predictions,
    cross_validated_constrained_blend,
    diagnose_stacking_issues,
)
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
            folds = np.load(fold_assignment_path, allow_pickle=False)
            if len(folds) == n_samples:
                return folds
            print(f"   Warning: Fold assignment length mismatch for {name}")
        except Exception as e:
            print(f"   Warning: Failed to load fold_assignment for {name}: {e}")

    if folds_path is not None and folds_path.exists():
        try:
            if folds_path.suffix.lower() == ".npy":
                folds = np.asarray(
                    np.load(folds_path, allow_pickle=False)
                ).ravel()
                if len(folds) == n_samples:
                    return folds
                print("   Warning: canonical folds.npy length mismatch")
            else:
                folds_df = pd.read_csv(folds_path)
                if "fold" in folds_df.columns and len(folds_df) == n_samples:
                    return folds_df["fold"].to_numpy()
                print("   Warning: folds.csv missing 'fold' column or length mismatch")
        except Exception as e:
            print(f"   Warning: Failed to read folds from {folds_path}: {e}")

    return None


def stack_from_prediction_pairs(
    prediction_pairs: dict[str, tuple[Path, Path]],
    y: np.ndarray | pd.Series,
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
    require_identity_artifacts: bool = False,
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
    target_array = np.asarray(y)
    multi_output = (
        target_array.ndim == 2 and target_array.shape[1] > 1
    )
    expected_multioutput_shape = target_array.shape if multi_output else None
    validation_problem_type = (
        "multi_label"
        if multi_output and problem_type == "classification"
        else problem_type
    )
    if multi_output:
        canonical_width = int(target_array.shape[1])
        if n_targets is not None and int(n_targets) != canonical_width:
            raise ValueError(
                "Canonical target width and submission output width differ: "
                f"{canonical_width} != {n_targets}"
            )
        if (
            expected_class_order is not None
            and len(expected_class_order) != canonical_width
        ):
            raise ValueError(
                "Canonical target width and ordered submission columns differ: "
                f"{canonical_width} != {len(expected_class_order)}"
            )

    valid_pairs, results = validate_oof_stack(
        prediction_pairs,
        models_dir,
        train_ids=train_ids,
        expected_class_order=expected_class_order,
        expected_shape=expected_multioutput_shape,
        folds_path=folds_path,
        strict_mode=False,
        problem_type=validation_problem_type,
        require_identity_artifacts=require_identity_artifacts,
    )
    print_oof_summary(results)
    if len(valid_pairs) < 2:
        return None, None

    # Filter out weak models (more than 50% worse than best)
    print("   Filtering weak models by OOF score...")
    y_true_np = np.asarray(y)
    valid_pairs, _computed_scores = filter_by_score_threshold(
        valid_pairs,
        y_true_np,
        metric_name,
        threshold_pct=0.50,  # Allow models up to 50% worse than best
    )
    if len(valid_pairs) < 2:
        print("   Warning: Not enough models after filtering weak performers")
        return None, None

    model_names = list(valid_pairs.keys())

    if problem_type == "classification" and not multi_output:
        y_encoded, class_order = encode_labels(y, expected_class_order)
    else:
        y_encoded = target_array
        class_order = expected_class_order

    oof_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
    loaded_model_names: list[str] = []
    calibration_summaries: list[dict[str, Any]] = []
    reference_test_ids: np.ndarray | None = None

    for name, (oof_path, test_path) in valid_pairs.items():
        try:
            oof_raw = np.load(oof_path, allow_pickle=False)
            test_raw = np.load(test_path, allow_pickle=False)
        except Exception as e:
            print(f"   Warning: Failed to load predictions for {name}: {e}")
            continue

        oof_raw = np.asarray(oof_raw, dtype=float)
        test_raw = np.asarray(test_raw, dtype=float)

        if multi_output:
            expected_width = int(target_array.shape[1])
            if oof_raw.shape != target_array.shape:
                print(
                    f"   Warning: Excluding {name}: OOF shape "
                    f"{oof_raw.shape} != canonical target shape "
                    f"{target_array.shape}"
                )
                continue
            if test_raw.ndim != 2 or test_raw.shape[1] != expected_width:
                print(
                    f"   Warning: Excluding {name}: test output width/shape "
                    f"{test_raw.shape} != (*, {expected_width})"
                )
                continue

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

        test_ids_path = models_dir / f"test_ids_{name}.npy"
        if not test_ids_path.exists():
            print(f"   Warning: Excluding {name}: missing test_ids_{name}.npy")
            continue
        try:
            model_test_ids = np.asarray(
                np.load(test_ids_path, allow_pickle=False)
            ).reshape(-1)
        except Exception as e:
            print(f"   Warning: Excluding {name}: invalid test IDs ({e})")
            continue
        if len(model_test_ids) != len(test_raw):
            print(
                f"   Warning: Excluding {name}: test ID/prediction mismatch "
                f"({len(model_test_ids)} vs {len(test_raw)})"
            )
            continue
        model_test_ids_str = pd.Series(model_test_ids).astype(str)
        if model_test_ids_str.duplicated().any():
            print(f"   Warning: Excluding {name}: duplicate test IDs")
            continue

        test_reorder: np.ndarray | None = None
        if reference_test_ids is None:
            reference_test_ids = model_test_ids_str.to_numpy()
        else:
            reference_ids_str = np.asarray(reference_test_ids).astype(str)
            if set(model_test_ids_str) != set(reference_ids_str):
                print(f"   Warning: Excluding {name}: incomplete test ID coverage")
                continue
            position_by_id = {
                test_id: idx
                for idx, test_id in enumerate(model_test_ids_str.tolist())
            }
            test_reorder = np.fromiter(
                (position_by_id[test_id] for test_id in reference_ids_str),
                dtype=np.int64,
                count=len(reference_ids_str),
            )

        if (
            enable_calibration
            and problem_type == "classification"
            and not multi_output
        ):
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
                    oof_preds = np.load(cal_path, allow_pickle=False)
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

        if test_reorder is not None:
            test_preds = np.asarray(test_preds)[test_reorder]

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
        loaded_model_names.append(name)

    # Validate shapes after normalization (CRITICAL: prevents inhomogeneous array errors)
    if oof_list:
        oof_shapes = {tuple(o.shape) for o in oof_list}
        test_shapes = {tuple(t.shape) for t in test_list}
        model_names = loaded_model_names

        if len(oof_shapes) > 1 or len(test_shapes) > 1:
            print("   Warning: Shape mismatch detected after normalization:")
            print(f"      OOF shapes: {oof_shapes}")
            print(f"      Test shapes: {test_shapes}")

            # Keep the most common OOF/test shape pair. Looking only at OOF
            # shape can retain incompatible test arrays and fail at np.stack.
            shape_counts = Counter(
                (tuple(o.shape), tuple(t.shape)) for o, t in zip(oof_list, test_list, strict=True)
            )
            target_shapes = shape_counts.most_common(1)[0][0]
            print(f"      Keeping models with OOF/test shapes: {target_shapes}")

            valid_idx = [
                i
                for i, (o, t) in enumerate(zip(oof_list, test_list, strict=True))
                if (tuple(o.shape), tuple(t.shape)) == target_shapes
            ]
            oof_list = [oof_list[i] for i in valid_idx]
            test_list = [test_list[i] for i in valid_idx]
            model_names = [model_names[i] for i in valid_idx]
            print(f"      Kept {len(oof_list)} compatible models: {model_names}")

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
    metric_lower = (metric_name or "").lower()
    ordinal_metric = "kappa" in metric_lower or "qwk" in metric_lower
    score_targets = np.asarray(y) if ordinal_metric else y_encoded

    avg_score = score_predictions(avg_oof, score_targets, problem_type, metric_name)
    print(f"   [META] Simple average: {avg_score:.6f}")

    (
        weights_constrained,
        constrained_oof,
        constrained_score,
    ) = cross_validated_constrained_blend(
        oof_stack,
        score_targets,
        problem_type,
        metric_name,
        split_targets=y_encoded,
    )
    print(f"   [META] Constrained (cross-fitted): {constrained_score:.6f}")

    meta_score = float("inf")
    meta_oof_preds = None
    meta_model = None
    try:
        if multi_output:
            raise ValueError(
                "Per-output meta-model fitting is not enabled; using "
                "cross-fitted constrained or average blending"
            )
        n_classes = len(np.unique(y_encoded)) if problem_type == "classification" else None
        meta_model, meta_oof_preds = cross_fitted_meta_predictions(
            meta_X,
            y_encoded,
            problem_type,
            n_classes,
            binary_single_col=binary_single_col,
        )

        meta_score = score_predictions(
            meta_oof_preds, score_targets, problem_type, metric_name
        )
        print(f"   [META] Meta-model (nested CV): {meta_score:.6f}")

        # Run stacking diagnostics to detect potential issues
        meta_model.fit(meta_X, y_encoded)
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
        if constrained_oof is None:
            raise ValueError("Selected constrained blend has no cross-fitted predictions")
        selected_oof = constrained_oof
        selected_weights = weights_constrained
    else:
        final_test_preds = np.average(test_stack, axis=0, weights=np.ones(len(oof_list)))
        selected_oof = avg_oof
        selected_weights = np.ones(len(oof_list)) / len(oof_list)

    selected_score = float(scores[best_method])

    selected_array = np.asarray(selected_oof)
    ordinal_continuous = (
        ordinal_metric
        and selected_array.ndim == 1
        and (np.nanmin(selected_array) < 0 or np.nanmax(selected_array) > 1)
    )

    if (
        problem_type == "classification"
        and not ordinal_continuous
    ):
        final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
        if (
            not multi_output
            and final_test_preds.ndim > 1
            and final_test_preds.shape[1] > 1
        ):
            final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

    calibration_info = {}
    # Post-calibration only helps probability metrics. For hard-label metrics
    # the decision rule (threshold/rounding) is tuned downstream on selected_oof
    # and applied to final_test_preds - calibrating only the test side would put
    # the two on different scales and invalidate the tuned rule.
    from .postprocessing import metric_label_kind

    label_metric = metric_label_kind(metric_name) is not None
    if (
        problem_type == "classification"
        and enable_post_calibration
        and not ordinal_continuous
        and not label_metric
        and not multi_output
    ):
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
        score_targets,
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
        "selected_oof": selected_oof,  # for metric-aware postprocessing downstream
        "calibration": calibration_summaries,
        "audit": {
            "dominant_model": audit.dominant_model,
            "dominance_weight": audit.dominance_weight,
            "warnings": audit.warnings,
            "notes": audit.notes,
            "calibration": audit.calibration,
        },
        "class_order": class_order,
        "test_ids": reference_test_ids,
    }

    return ensemble, final_test_preds
