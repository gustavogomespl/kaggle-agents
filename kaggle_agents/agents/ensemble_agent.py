"""Ensemble agent for model stacking and blending."""

import os
import re
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict

from ..core.config import get_config, is_metric_minimization
from ..core.state import KaggleState
from ..utils.calibration import calibrate_oof_predictions, calibrate_test_predictions
from ..utils.ensemble_audit import full_ensemble_audit, post_calibrate_ensemble
from ..utils.llm_utils import get_text_content
from ..utils.oof_validation import print_oof_summary, validate_oof_stack


class EnsembleAgent:
    """Agent responsible for creating model ensembles."""

    def __init__(self):
        """Initialize ensemble agent."""
        pass

    def _find_prediction_pairs(self, models_dir: Path) -> dict[str, tuple[Path, Path]]:
        """Find matching OOF/Test prediction pairs under models/."""
        oof_files = sorted(models_dir.glob("oof_*.npy"))
        pairs: dict[str, tuple[Path, Path]] = {}
        for oof_path in oof_files:
            name = oof_path.stem.replace("oof_", "", 1)
            test_path = models_dir / f"test_{name}.npy"
            if test_path.exists():
                pairs[name] = (oof_path, test_path)
        return pairs

    def _validate_oof_alignment(
        self,
        models_dir: Path,
        train_ids: np.ndarray,
        expected_class_order: list[str],
    ) -> dict[str, tuple[Path, Path]]:
        """Validate and filter OOFs by row and class alignment.

        Args:
            models_dir: Directory containing model artifacts
            train_ids: IDs from original train.csv (expected row order)
            expected_class_order: Class order from sample_submission

        Returns:
            Dictionary of valid prediction pairs (name -> (oof_path, test_path))
        """
        valid_pairs: dict[str, tuple[Path, Path]] = {}

        for oof_path in models_dir.glob("oof_*.npy"):
            name = oof_path.stem.replace("oof_", "", 1)
            test_path = models_dir / f"test_{name}.npy"

            if not test_path.exists():
                continue

            # 1. Verify class_order
            class_order_path = models_dir / f"class_order_{name}.npy"
            if class_order_path.exists():
                saved_order = np.load(class_order_path, allow_pickle=True).tolist()
                if saved_order != expected_class_order:
                    print(f"   ⚠️ {name}: class order mismatch, skipping")
                    continue
            # If class_order file doesn't exist, try global class_order.npy
            else:
                global_class_order = models_dir / "class_order.npy"
                if global_class_order.exists():
                    saved_order = np.load(global_class_order, allow_pickle=True).tolist()
                    if saved_order != expected_class_order:
                        print(f"   ⚠️ {name}: global class order mismatch, skipping")
                        continue

            # 2. Verify train_ids (row order)
            train_ids_path = models_dir / f"train_ids_{name}.npy"
            if train_ids_path.exists():
                saved_ids = np.load(train_ids_path, allow_pickle=True)
                if not np.array_equal(saved_ids, train_ids):
                    print(f"   ⚠️ {name}: train IDs mismatch, skipping")
                    continue
            else:
                # Metadata missing: enter with caution
                print(f"   ⚠️ {name}: alignment metadata missing, entering with caution")

            valid_pairs[name] = (oof_path, test_path)

        return valid_pairs

    def _encode_labels(
        self, y: np.ndarray | pd.Series, class_order: list[str] | None
    ) -> tuple[np.ndarray, list[str]]:
        """Encode labels to integer indices with optional class order."""
        y_array = np.asarray(y)

        if class_order:
            cat = pd.Categorical(y_array, categories=class_order)
            if (cat.codes < 0).any():
                print("   ⚠️ Label encoding mismatch with class_order, using sorted uniques")
            else:
                return cat.codes.astype(int), class_order

        classes, y_encoded = np.unique(y_array, return_inverse=True)
        return y_encoded.astype(int), classes.tolist()

    def _load_cv_folds(
        self,
        name: str,
        models_dir: Path,
        folds_path: Path | None,
        n_samples: int,
    ) -> np.ndarray | None:
        """Load per-model or global fold assignments when available."""
        fold_assignment_path = models_dir / f"fold_assignment_{name}.npy"
        if fold_assignment_path.exists():
            try:
                folds = np.load(fold_assignment_path)
                if len(folds) == n_samples:
                    return folds
                print(f"   ⚠️ Fold assignment length mismatch for {name}")
            except Exception as e:
                print(f"   ⚠️ Failed to load fold_assignment for {name}: {e}")

        if folds_path is not None and folds_path.exists():
            try:
                folds_df = pd.read_csv(folds_path)
                if "fold" in folds_df.columns and len(folds_df) == n_samples:
                    return folds_df["fold"].to_numpy()
                print("   ⚠️ folds.csv missing 'fold' column or length mismatch")
            except Exception as e:
                print(f"   ⚠️ Failed to read folds.csv: {e}")

        return None

    def _score_predictions(
        self,
        preds: np.ndarray,
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
    ) -> float:
        """Return a score where LOWER is better."""
        from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error

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
            elif "acc" in metric:
                if preds.ndim > 1 and preds.shape[1] > 1:
                    pred_labels = np.argmax(preds, axis=1)
                else:
                    pred_labels = (preds >= 0.5).astype(int)
                score = accuracy_score(y_true, pred_labels)
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

    def _compute_oof_score(
        self,
        oof_path: Path,
        y_true: np.ndarray,
        metric_name: str = "log_loss",
    ) -> float:
        """Compute model score via OOF predictions (on-the-fly if needed)."""
        from sklearn.metrics import log_loss, mean_squared_error

        oof = np.load(oof_path)
        # Clip and normalize for log_loss
        oof = np.clip(oof, 1e-15, 1 - 1e-15)
        if oof.ndim > 1 and oof.shape[1] > 1:
            oof = oof / oof.sum(axis=1, keepdims=True)

        if metric_name == "log_loss":
            return log_loss(y_true, oof)
        elif metric_name in ["rmse", "mse", "mean_squared_error"]:
            return np.sqrt(mean_squared_error(y_true, oof.ravel() if oof.ndim > 1 else oof))
        return float("inf")

    def _filter_by_score_threshold(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        y_true: np.ndarray,
        metric_name: str,
        model_scores: dict[str, float] | None = None,
        threshold_pct: float = 0.20,
    ) -> tuple[dict[str, tuple[Path, Path]], dict[str, float]]:
        """Filter models with score within X% of best. Computes scores on-the-fly if needed.

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
                computed_scores[name] = self._compute_oof_score(oof_path, y_true, metric_name)
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
                print(f"   ✅ {name}: score {score:.6f} (within threshold)")
            else:
                print(f"   ⚠️ {name}: score {score:.6f} > threshold {threshold:.6f}, skipping")

        return filtered, computed_scores

    def _tune_meta_model(
        self,
        meta_X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
    ) -> LogisticRegression | Ridge:
        """Tune meta-model regularization using cross-validation.

        Uses grid search over C (classification) or alpha (regression) values
        with explicit StratifiedKFold/KFold and proper clip/normalize before log_loss.
        """
        from sklearn.metrics import log_loss, mean_squared_error
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

        if problem_type == "classification":
            best_score = float("inf")
            best_C = 0.01
            # Explicit StratifiedKFold with shuffle
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for C in [0.001, 0.01, 0.1, 1.0]:
                try:
                    # Don't pass multi_class (deprecated)
                    model = LogisticRegression(
                        C=C, random_state=42, max_iter=1000, solver="lbfgs"
                    )
                    # cross_val_predict to get OOF predictions
                    oof_preds = cross_val_predict(model, meta_X, y, cv=cv, method="predict_proba")
                    # CLIP and NORMALIZE before log_loss
                    oof_preds = np.clip(oof_preds, 1e-15, 1 - 1e-15)
                    oof_preds = oof_preds / oof_preds.sum(axis=1, keepdims=True)
                    mean_score = log_loss(y, oof_preds)
                    if mean_score < best_score:
                        best_score = mean_score
                        best_C = C
                except Exception as e:
                    print(f"      ⚠️ C={C} failed: {e}")
                    continue

            print(f"   Meta-model tuning: best C={best_C} (OOF log_loss={best_score:.4f})")
            return LogisticRegression(C=best_C, random_state=42, max_iter=1000, solver="lbfgs")
        else:
            best_score = float("inf")
            best_alpha = 1.0
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            for alpha in [0.1, 1.0, 10.0, 100.0]:
                try:
                    model = Ridge(alpha=alpha, random_state=42)
                    oof_preds = cross_val_predict(model, meta_X, y, cv=cv)
                    mean_score = np.sqrt(mean_squared_error(y, oof_preds))
                    if mean_score < best_score:
                        best_score = mean_score
                        best_alpha = alpha
                except Exception as e:
                    print(f"      ⚠️ alpha={alpha} failed: {e}")
                    continue

            print(f"   Meta-model tuning: best alpha={best_alpha} (OOF RMSE={best_score:.4f})")
            return Ridge(alpha=best_alpha, random_state=42)

    def _constrained_meta_learner(
        self,
        oof_stack: np.ndarray,
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
    ) -> tuple[np.ndarray, float]:
        """Learn non-negative weights that sum to 1."""
        n_models = oof_stack.shape[0]

        def objective(weights: np.ndarray) -> float:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_models) / n_models
            blended = np.average(oof_stack, axis=0, weights=weights)
            return self._score_predictions(blended, y_true, problem_type, metric_name)

        try:
            from scipy.optimize import minimize

            init_weights = np.ones(n_models) / n_models
            result = minimize(
                objective,
                init_weights,
                method="SLSQP",
                bounds=[(0, 1)] * n_models,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            )
            opt_weights = result.x / result.x.sum()
            best_score = objective(opt_weights)
            print(f"   Constrained weights: score {best_score:.6f}")
            return opt_weights, best_score
        except Exception as e:
            print(f"   ⚠️ Constrained optimization failed: {e}")
            opt_weights = self._dirichlet_weight_search(
                oof_stack, y_true, problem_type, metric_name, n_samples=300
            )
            best_score = objective(opt_weights)
            return opt_weights, best_score

    def _dirichlet_weight_search(
        self,
        oof_stack: np.ndarray,
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
        n_samples: int = 300,
    ) -> np.ndarray:
        """Fallback: weight search via Dirichlet sampling on the simplex.

        Args:
            oof_stack: Stacked OOF predictions (n_models, n_samples, n_classes)
            y_true: True labels
            problem_type: 'classification' or 'regression'
            metric_name: Metric name for scoring
            n_samples: Number of Dirichlet samples

        Returns:
            Optimal weights array
        """
        from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

        n_models = oof_stack.shape[0]
        best_weights = np.ones(n_models) / n_models
        best_score = float("inf")

        # Sample weights from simplex via Dirichlet(1,1,...,1)
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            weights = rng.dirichlet(np.ones(n_models))
            blended = np.average(oof_stack, axis=0, weights=weights)

            try:
                if problem_type == "classification":
                    blended = np.clip(blended, 1e-15, 1 - 1e-15)
                    if blended.ndim > 1 and blended.shape[1] > 1:
                        blended = blended / blended.sum(axis=1, keepdims=True)
                    # Classification metrics
                    if metric_name in ["auc", "roc_auc"]:
                        from sklearn.metrics import roc_auc_score

                        # Negate because we minimize, but AUC should be maximized
                        score = -roc_auc_score(
                            y_true, blended, multi_class="ovr", average="weighted"
                        )
                    else:
                        # Default: log_loss for classification
                        score = log_loss(y_true, blended)
                else:
                    # Regression
                    if blended.ndim > 1:
                        blended = blended.ravel()
                    if metric_name in ["mae", "mean_absolute_error"]:
                        score = mean_absolute_error(y_true, blended)
                    elif metric_name in ["mse", "mean_squared_error"]:
                        score = mean_squared_error(y_true, blended)
                    else:
                        # Default: RMSE for regression
                        score = np.sqrt(mean_squared_error(y_true, blended))

                if score < best_score:
                    best_score = score
                    best_weights = weights
            except Exception:
                continue

        print(f"   Dirichlet search ({n_samples} samples): best score {best_score:.6f}")
        return best_weights

    def _stack_from_prediction_pairs(
        self,
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
        """Build stacking ensemble directly from saved OOF/Test predictions."""
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

        model_names = list(valid_pairs.keys())

        if problem_type == "classification":
            y_encoded, class_order = self._encode_labels(y, expected_class_order)
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
                print(f"   ⚠️ Failed to load predictions for {name}: {e}")
                continue

            oof_raw = np.asarray(oof_raw, dtype=float)
            test_raw = np.asarray(test_raw, dtype=float)

            if enable_calibration and problem_type == "classification":
                try:
                    cv_folds = self._load_cv_folds(
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
                    print(f"   ⚠️ Calibration failed for {name}: {e}")
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
        avg_score = self._score_predictions(avg_oof, y_encoded, problem_type, metric_name)
        print(f"   [META] Simple average: {avg_score:.6f}")

        weights_constrained, constrained_score = self._constrained_meta_learner(
            oof_stack, y_encoded, problem_type, metric_name
        )
        print(f"   [META] Constrained: {constrained_score:.6f}")

        meta_score = float("inf")
        meta_oof_preds = None
        meta_model = None
        try:
            meta_model = self._tune_meta_model(meta_X, y_encoded, problem_type)
            if problem_type == "classification":
                from sklearn.model_selection import StratifiedKFold, cross_val_predict

                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                meta_oof_preds = cross_val_predict(
                    meta_model, meta_X, y_encoded, cv=cv, method="predict_proba"
                )
                if binary_single_col and meta_oof_preds.ndim > 1:
                    meta_oof_preds = meta_oof_preds[:, 1]
            else:
                from sklearn.model_selection import KFold, cross_val_predict

                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                meta_oof_preds = cross_val_predict(meta_model, meta_X, y_encoded, cv=cv)

            meta_score = self._score_predictions(
                meta_oof_preds, y_encoded, problem_type, metric_name
            )
            print(f"   [META] Meta-model: {meta_score:.6f}")
        except Exception as e:
            print(f"   ⚠️ Meta-model evaluation failed: {e}")

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
            selected_oof = np.average(oof_stack, axis=0, weights=weights_constrained)
            selected_weights = weights_constrained
        else:
            final_test_preds = np.average(test_stack, axis=0, weights=np.ones(len(oof_list)))
            selected_oof = avg_oof
            selected_weights = np.ones(len(oof_list)) / len(oof_list)

        selected_score = self._score_predictions(
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

    def create_oof_weighted_blend(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """Weighted blend using saved OOF predictions.

        Args:
            prediction_pairs: Dictionary of (oof_path, test_path) pairs
            y_true: True labels
            problem_type: 'classification' or 'regression'
            metric_name: Metric name for optimization

        Returns:
            Tuple of (test_predictions, oof_score, weights_dict)
        """
        from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

        names = list(prediction_pairs.keys())
        n_models = len(names)

        print(f"   Creating OOF weighted blend with {n_models} models...")
        print(f"   Problem type: {problem_type}, Metric: {metric_name}")

        # Load OOF and test predictions
        oof_list = [np.load(oof) for oof, _ in prediction_pairs.values()]
        test_list = [np.load(test) for _, test in prediction_pairs.values()]

        # Stack: (n_models, n_samples, n_classes)
        oof_stack = np.stack(oof_list, axis=0)
        test_stack = np.stack(test_list, axis=0)

        # Helper function to compute score based on problem_type and metric_name
        def compute_score(blended: np.ndarray) -> float:
            if problem_type == "classification":
                blended = np.clip(blended, 1e-15, 1 - 1e-15)
                if blended.ndim > 1 and blended.shape[1] > 1:
                    blended = blended / blended.sum(axis=1, keepdims=True)
                # Classification metrics
                if metric_name in ["auc", "roc_auc"]:
                    from sklearn.metrics import roc_auc_score

                    # Negate because we minimize, but AUC should be maximized
                    return -roc_auc_score(
                        y_true, blended, multi_class="ovr", average="weighted"
                    )
                else:
                    # Default: log_loss for classification
                    return log_loss(y_true, blended)
            else:
                # Regression
                if blended.ndim > 1:
                    blended = blended.ravel()
                if metric_name in ["mae", "mean_absolute_error"]:
                    return mean_absolute_error(y_true, blended)
                elif metric_name in ["mse", "mean_squared_error"]:
                    return mean_squared_error(y_true, blended)
                else:
                    # Default: RMSE for regression
                    return np.sqrt(mean_squared_error(y_true, blended))

        # Objective for optimization
        def objective(weights: np.ndarray) -> float:
            weights = np.array(weights) / np.sum(weights)
            blended = np.average(oof_stack, axis=0, weights=weights)
            return compute_score(blended)

        # Try scipy.optimize.minimize, else use Dirichlet fallback
        try:
            from scipy.optimize import minimize

            init_weights = np.ones(n_models) / n_models
            result = minimize(
                objective,
                init_weights,
                method="SLSQP",
                bounds=[(0, 1)] * n_models,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            )
            opt_weights = result.x / result.x.sum()
            print(f"   SciPy optimization: best score {result.fun:.6f}")
        except ImportError:
            print("   SciPy not available, using Dirichlet fallback")
            opt_weights = self._dirichlet_weight_search(
                oof_stack, y_true, problem_type, metric_name, n_samples=300
            )

        # Calculate final score using the same metric
        blended_oof = np.average(oof_stack, axis=0, weights=opt_weights)
        oof_score = compute_score(blended_oof)

        # For display, negate back if AUC (since we negated for minimization)
        display_score = -oof_score if metric_name in ["auc", "roc_auc"] else oof_score

        # Generate test predictions
        blended_test = np.average(test_stack, axis=0, weights=opt_weights)
        if problem_type == "classification":
            blended_test = np.clip(blended_test, 1e-15, 1 - 1e-15)
            if blended_test.ndim > 1 and blended_test.shape[1] > 1:
                blended_test = blended_test / blended_test.sum(axis=1, keepdims=True)

        weights_dict = dict(zip(names, opt_weights))
        print(f"   Weighted blend: OOF score={display_score:.6f}")
        for name, weight in weights_dict.items():
            print(f"      {name}: {weight:.4f}")

        return blended_test, oof_score, weights_dict

    def _validate_class_order(
        self, models_dir: Path, sample_submission_path: Path
    ) -> tuple[bool, str]:
        """Validate that saved predictions use canonical class order.

        Checks if class_order.npy exists and matches sample_submission columns.
        This prevents ensemble degradation from misaligned class predictions.

        Args:
            models_dir: Directory containing model artifacts
            sample_submission_path: Path to sample_submission.csv

        Returns:
            Tuple of (is_valid, message)
        """
        class_order_path = models_dir / "class_order.npy"

        if not class_order_path.exists():
            return False, (
                "class_order.npy not found - predictions may have misaligned class orders. "
                "Ensemble averaging may produce degraded results."
            )

        if not sample_submission_path.exists():
            return False, "sample_submission.csv not found for class order validation"

        try:
            # Load saved class order
            saved_order = np.load(class_order_path, allow_pickle=True).tolist()

            # Load expected class order from sample submission
            sample_sub = pd.read_csv(sample_submission_path)
            expected_order = sample_sub.columns[1:].tolist()

            if saved_order != expected_order:
                return False, (
                    f"Class order mismatch! "
                    f"Saved: {saved_order[:5]}... "
                    f"Expected: {expected_order[:5]}..."
                )

            return True, f"Class order validated ({len(saved_order)} classes)"

        except Exception as e:
            return False, f"Class order validation error: {e}"

    def _ensemble_from_predictions(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        sample_submission_path: Path,
        output_path: Path,
    ) -> bool:
        """Create a simple average ensemble directly from saved predictions."""
        if not sample_submission_path.exists():
            print("   ❌ Sample submission not found, cannot build prediction ensemble")
            return False

        # Validate class order alignment before ensembling
        if prediction_pairs:
            models_dir = list(prediction_pairs.values())[0][0].parent
            is_valid, msg = self._validate_class_order(models_dir, sample_submission_path)
            if is_valid:
                print(f"   ✅ {msg}")
            else:
                print(f"   ⚠️  Class order warning: {msg}")
                # Continue but warn - older models may not have class_order.npy

        sample_sub = pd.read_csv(sample_submission_path)
        preds_list = []
        names = []

        for name, (_, test_path) in prediction_pairs.items():
            preds = np.load(test_path)
            preds = np.asarray(preds, dtype=np.float32)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds_list.append(preds)
            names.append(name)

        if len(preds_list) < 2:
            print("   ⚠️  Not enough prediction pairs for ensemble")
            return False

        # Ensure consistent shapes
        shapes = {p.shape for p in preds_list}
        if len(shapes) != 1:
            print(f"   ❌ Prediction shapes mismatch: {shapes}")
            return False

        stacked = np.stack(preds_list, axis=0)  # (n_models, n_samples, n_cols)
        ensemble_preds = stacked.mean(axis=0)

        if ensemble_preds.shape[0] != len(sample_sub):
            print(
                f"   ❌ Prediction length mismatch: preds={ensemble_preds.shape[0]}, sample={len(sample_sub)}"
            )
            return False

        if ensemble_preds.shape[1] == 1:
            sample_sub.iloc[:, 1] = ensemble_preds[:, 0]
        else:
            if ensemble_preds.shape[1] != (len(sample_sub.columns) - 1):
                print(
                    "   ❌ Prediction column mismatch: "
                    f"preds={ensemble_preds.shape[1]}, sample_cols={len(sample_sub.columns) - 1}"
                )
                return False
            sample_sub.iloc[:, 1:] = ensemble_preds

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        print(f"   ✅ Saved prediction-only ensemble to {output_path.name}")
        print(f"   ✅ Models used: {', '.join(names)}")
        return True

    def create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        metric_name: str = "",
        sample_submission_path: Path | None = None,
        train_ids: np.ndarray | None = None,
        expected_class_order: list[str] | None = None,
        n_targets: int | None = None,
        folds_path: Path | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        """Create stacking ensemble from best models using saved OOF predictions.

        Args:
            models: List of trained models
            model_names: List of model names
            working_dir: Working directory
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Tuple of (ensemble_dict, final_test_predictions or None)
        """
        # Generate out-of-fold predictions from base models
        print(f"  Creating stacking ensemble with {len(models)} base models...")

        models_dir = working_dir / "models"
        prediction_pairs = {
            name: (models_dir / f"oof_{name}.npy", models_dir / f"test_{name}.npy")
            for name in model_names
            if (models_dir / f"oof_{name}.npy").exists()
            and (models_dir / f"test_{name}.npy").exists()
        }

        enable_calibration = os.getenv("KAGGLE_AGENTS_STACKING_CALIBRATION", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        enable_post_calibration = os.getenv(
            "KAGGLE_AGENTS_STACKING_POST_CALIBRATION", "1"
        ).lower() not in {"0", "false", "no"}
        calibration_method = os.getenv(
            "KAGGLE_AGENTS_STACKING_CALIBRATION_METHOD", "auto"
        ).lower()
        if n_targets is None and sample_submission_path and sample_submission_path.exists():
            try:
                sample_head = pd.read_csv(sample_submission_path, nrows=1)
                n_targets = sample_head.shape[1] - 1
                if expected_class_order is None and sample_head.shape[1] > 2:
                    expected_class_order = sample_head.columns[1:].tolist()
            except Exception as e:
                print(f"   ⚠️ Failed to read sample submission for targets: {e}")

        if prediction_pairs:
            print(f"  Found {len(prediction_pairs)} prediction pairs for stacking")
            ensemble, final_test_preds = self._stack_from_prediction_pairs(
                prediction_pairs=prediction_pairs,
                y=y,
                problem_type=problem_type,
                metric_name=metric_name,
                models_dir=models_dir,
                expected_class_order=expected_class_order,
                train_ids=train_ids,
                folds_path=folds_path,
                enable_calibration=enable_calibration,
                enable_post_calibration=enable_post_calibration,
                n_targets=n_targets,
                calibration_method=calibration_method,
            )
            if ensemble is not None and final_test_preds is not None:
                name_to_model = dict(zip(model_names, models, strict=False))
                ensemble["base_models"] = [
                    name_to_model[name]
                    for name in ensemble.get("base_model_names", [])
                    if name in name_to_model
                ]
                return ensemble, final_test_preds

        meta_features = []
        valid_models = []
        valid_names = []

        for i, (model, name) in enumerate(zip(models, model_names, strict=False)):
            print(f"    Processing model {i + 1}/{len(models)}: {name}")

            # Try to load OOF predictions
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                print(f"      ✅ Loaded OOF from {oof_path.name}")
                oof_preds = np.load(oof_path)
                meta_features.append(oof_preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                print("      ⚠️  OOF file not found, falling back to cross_val_predict (slow)")
                if problem_type == "classification":
                    oof_preds = cross_val_predict(
                        model, X, y, cv=5, method="predict_proba", n_jobs=-1
                    )
                    # Take probabilities for positive class
                    if oof_preds.ndim > 1:
                        meta_features.append(oof_preds[:, 1])
                    else:
                        meta_features.append(oof_preds)
                else:
                    oof_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                    meta_features.append(oof_preds)

                valid_models.append(model)
                valid_names.append(name)

        if not meta_features:
            raise ValueError("No meta-features could be generated")

        # Create meta-feature matrix
        meta_X = np.column_stack(meta_features)

        # Train meta-model with tuned regularization
        print("    Tuning and training meta-model...")
        meta_model = self._tune_meta_model(meta_X, y.values if hasattr(y, "values") else y, problem_type)
        meta_model.fit(meta_X, y)

        # We don't need to retrain base models if we use the saved test preds!
        # But we keep them in the return dict for completeness

        return (
            {
                "meta_model": meta_model,
                "base_models": valid_models,
                "base_model_names": valid_names,
                "stacking_method": "meta",
                "weights": None,
                "class_order": None,
            },
            None,
        )

    def create_blending_ensemble(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> dict[str, Any]:
        """Create blending ensemble using simple averaging.

        Args:
            models: List of trained models
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Dictionary with base models and weights
        """
        print(f"  Creating blending ensemble with {len(models)} models...")

        # Optimize weights
        weights = self.optimize_blending_weights(models, X, y, problem_type)

        return {"base_models": models, "weights": weights}

    def optimize_blending_weights(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> list[float]:
        """Optimize blending weights using scipy.minimize."""
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss, mean_squared_error

        print("    Optimizing blending weights...")

        # Generate OOF predictions
        oof_preds = []
        for model in models:
            if problem_type == "classification":
                preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
                if preds.ndim > 1:
                    oof_preds.append(preds[:, 1])
                else:
                    oof_preds.append(preds)
            else:
                preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                oof_preds.append(preds)

        oof_preds = np.column_stack(oof_preds)

        # Define loss function
        def loss_func(weights):
            # Normalize weights
            weights = np.array(weights)
            weights /= weights.sum()

            # Weighted average
            final_preds = np.average(oof_preds, axis=1, weights=weights)

            if problem_type == "classification":
                # Clip to avoid log(0)
                final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)
                return log_loss(y, final_preds)
            return np.sqrt(mean_squared_error(y, final_preds))

        # Initial weights (equal)
        init_weights = [1.0 / len(models)] * len(models)

        # Constraints: weights sum to 1, 0 <= weight <= 1
        constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}
        bounds = [(0, 1)] * len(models)

        result = minimize(
            loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        opt_weights = result.x / result.x.sum()
        print(f"    Optimal weights: {opt_weights}")
        opt_weights = result.x / result.x.sum()
        print(f"    Optimal weights: {opt_weights}")
        return opt_weights.tolist()

    def create_caruana_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        metric_name: str = "",
        n_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Create ensemble using Caruana's Hill Climbing (Forward Selection).

        Iteratively adds the model that maximizes the ensemble's CV score.
        Allows repetition of models (weighted ensemble by count).
        """
        from sklearn.metrics import log_loss, mean_squared_error

        print(f"  Creating Caruana Ensemble (Hill Climbing) with {len(models)} models...")

        # Load OOFs
        oof_preds = []
        valid_models = []
        valid_names = []

        for i, (model, name) in enumerate(zip(models, model_names, strict=False)):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                preds = np.load(oof_path)
                oof_preds.append(preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                print(f"    ⚠️ Skipping {name} (no OOF found)")

        if not oof_preds:
            raise ValueError("No OOF predictions found for Caruana ensemble")

        oof_preds = np.column_stack(oof_preds)
        n_models = oof_preds.shape[1]

        # Metric function
        def get_score(y_true, y_pred):
            if problem_type == "classification":
                # Assuming AUC for classification if not specified, or LogLoss
                # Let's use LogLoss for optimization as it's differentiable/smooth
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -log_loss(y_true, y_pred)  # Maximize negative log loss
            return -np.sqrt(mean_squared_error(y_true, y_pred))  # Maximize negative RMSE

        # Hill Climbing
        current_ensemble_preds = np.zeros_like(oof_preds[:, 0])
        ensemble_counts = np.zeros(n_models, dtype=int)
        best_score = -float("inf")

        # Initial step: pick best single model
        for i in range(n_models):
            score = get_score(y, oof_preds[:, i])
            if score > best_score:
                best_score = score
                best_init_idx = i

        current_ensemble_preds = oof_preds[:, best_init_idx]
        ensemble_counts[best_init_idx] = 1

        print(f"    Init Best Score: {best_score:.4f} (Model: {valid_names[best_init_idx]})")

        # Iterations
        for it in range(n_iterations):
            best_iter_score = -float("inf")
            best_iter_idx = -1

            # Try adding each model
            current_size = it + 2  # +1 for init, +1 for current iter (1-based)

            for i in range(n_models):
                # Candidate: current sum + new model prediction
                # Average = (current_sum + new_pred) / current_size
                # But we maintain sum for efficiency?
                # Actually, let's maintain the running sum to avoid re-summing everything
                # current_ensemble_preds is currently the AVERAGE.
                # So convert back to sum: current_avg * (current_size - 1)

                current_sum = current_ensemble_preds * (current_size - 1)
                candidate_avg = (current_sum + oof_preds[:, i]) / current_size

                score = get_score(y, candidate_avg)
                if score > best_iter_score:
                    best_iter_score = score
                    best_iter_idx = i

            # Update best
            if best_iter_score > best_score:
                best_score = best_iter_score
                ensemble_counts[best_iter_idx] += 1
                current_ensemble_preds = (
                    current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
                ) / current_size
                # print(f"    Iter {it+1}: Added {valid_names[best_iter_idx]} -> Score: {best_score:.4f}")
            else:
                # If no improvement, should we stop? Caruana usually continues to smooth out
                # But for simplicity, we can continue or stop. Let's continue.
                ensemble_counts[best_iter_idx] += 1
                current_ensemble_preds = (
                    current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
                ) / current_size

        # Calculate final weights
        weights = ensemble_counts / ensemble_counts.sum()
        print(f"    Final Caruana Weights: {weights}")

        oof_score = self._score_predictions(
            current_ensemble_preds,
            y.values if hasattr(y, "values") else y,
            problem_type,
            metric_name,
        )

        return {
            "base_models": valid_models,
            "base_model_names": valid_names,
            "weights": weights.tolist(),
            "oof_score": oof_score,
        }

    def create_temporal_ensemble(
        self,
        working_dir: Path,
        submissions: list[Any],  # List[SubmissionResult]
        current_iteration: int,
        metric_name: str,
    ) -> bool:
        """
        Create Temporal Ensemble (Success Memory) by blending past best submissions.
        Strategies:
        1. Rank Averaging (Robust) - Primary
        2. Weighted Blending (Fallback)

        Args:
            working_dir: Path to working directory
            submissions: List of SubmissionResult objects
            current_iteration: Current iteration number
            metric_name: Name of the evaluation metric (to determine sort direction)

        Returns:
            True if ensemble created and saved as submission.csv
        """
        print(f"\n  ⏳ Temporal Ensemble (Iteration {current_iteration})")

        # Determine strict direction
        minimize = is_metric_minimization(metric_name)
        print(f"      Metric: {metric_name} (Minimize: {minimize})")

        # 1. Gather candidate files
        candidates = []

        # From state history
        valid_history = [
            s
            for s in submissions
            if s.file_path and Path(s.file_path).exists() and s.public_score is not None
        ]

        # Also scan directory for manual matches (recovered state)
        for f in working_dir.glob("submission_iter_*_score_*.csv"):
            if f.name not in [Path(s.file_path).name for s in valid_history]:
                try:
                    # Parse score from filename: submission_iter_X_score_0.1234.csv
                    parts = f.stem.split("_")
                    if "score" in parts:
                        score_idx = parts.index("score") + 1
                        score = float(parts[score_idx])
                        candidates.append({"path": f, "score": score})
                except Exception:
                    continue

        # Convert history to uniform dict
        for sub in valid_history:
            candidates.append({"path": Path(sub.file_path), "score": sub.public_score})

        # Deduplicate by path
        unique_candidates = {str(c["path"]): c for c in candidates}.values()
        candidates = list(unique_candidates)

        if len(candidates) < 2:
            print(
                f"      Running single model (History: {len(candidates)}), needs 2+ for ensemble."
            )
            return False

        # Sort by score (Assume HIGHER is better for selection logic, we will check metric later)
        # Actually simplest heuristic: take top 3 distinct files
        # We don't know metric direction here easily, but usually MLE-bench scores are "higher=better" implies internal conversion?
        # Let's assume standard kaggle logic: we need to know metric.
        # SAFE FALLBACK: Just take the *last* 3 iterations as they should be improving?
        # BETTER: Sort by score descending (assuming AUC/Acc) or ascending (RMSE/LogLoss)??
        # CRITICAL: We need metric direction. But Rank Averaging is robust to scale, not direction if sorted wrong.
        # Let's use the explicit 'best_score' tracking in state to know which submissions were "improvements".
        # Filter candidates to only those that were considered "best" at their time?
        # SIMPLIFICATION: Just take the top 3 available files. Assuming 'score' in filename is meaningful.

        # Sort logic based on metric direction
        # If minimize (RMSE, LogLoss): asc=True (lower score is better)
        # If maximize (AUC, Accuracy): asc=False (higher score is better)
        # We want the BEST files first.
        # So for minimize: sort by score ASC.
        # For maximize: sort by score DESC (reverse=True).
        reverse_sort = not minimize

        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=reverse_sort)
        top_k = sorted_candidates[:3]

        print(f"      Blending Top {len(top_k)} past submissions:")
        dfs = []
        for c in top_k:
            print(f"      - {c['path'].name} (Score: {c['score']:.4f})")
            try:
                df = pd.read_csv(c["path"])
                # Sort by ID to ensure alignment
                if "id" in df.columns:
                    df = df.sort_values("id")
                dfs.append(df)
            except Exception as e:
                print(f"        ⚠️ Failed to read: {e}")

        if not dfs:
            return False

        # Rank Averaging
        # 1. Convert predictions to Ranks (0..1)
        # 2. Average Ranks
        # 3. (Optional) Map back to distribution? Or just use Scaled Rank as prob?
        # For submission, we need actual values.
        # If Regression: Average Values.
        # If Classification (Probs): Average Probs.
        # Rank Averaging is mostly for ROC-AUC / Ranking metrics.
        # Let's stick to SIMPLE WEIGHTED BLENDING based on rank (1st gets 50%, 2nd 30%, 3rd 20%)

        try:
            sample = dfs[0]
            if len(sample.columns) < 2:
                return False
            pred_col = sample.columns[1]

            # Weighted Average
            # Weights: 1st=3, 2nd=2, 3rd=1 (normalized)
            weights = np.array([3.0, 2.0, 1.0])[: len(dfs)]
            weights /= weights.sum()

            print(f"      Weights: {weights}")

            final_preds = np.zeros_like(sample[pred_col], dtype=float)

            for i, df in enumerate(dfs):
                vals = df[pred_col].values
                # Sanity fill NaNs
                vals = np.nan_to_num(vals)
                final_preds += vals * weights[i]

            # Save
            submission_path = working_dir / "submission.csv"
            out_df = sample.copy()
            out_df[pred_col] = final_preds
            out_df.to_csv(submission_path, index=False)
            print(f"      ✅ Saved Temporal Ensemble to {submission_path}")
            return True

        except Exception as e:
            print(f"      ❌ Temporal Ensemble Failed: {e}")
            return False

    def predict_stacking(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using stacking ensemble.

        Args:
            ensemble: Ensemble dictionary with meta_model and base_models
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble.get("base_models", [])
        meta_model = ensemble.get("meta_model")
        stacking_method = ensemble.get("stacking_method", "meta")
        weights = ensemble.get("weights")

        if stacking_method in {"average", "constrained"} and weights is not None:
            weights_array = np.array(weights, dtype=float)
            if weights_array.sum() <= 0:
                weights_array = np.ones(len(base_models)) / len(base_models)
            else:
                weights_array = weights_array / weights_array.sum()

            predictions = []
            multi_class = False
            for model in base_models:
                if problem_type == "classification" and hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1 and preds.shape[1] > 2:
                        multi_class = True
                        predictions.append(preds)
                    elif preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))

            blended = np.average(predictions, axis=0, weights=weights_array)
            if problem_type == "classification" and multi_class:
                blended = np.clip(blended, 1e-15, 1 - 1e-15)
                blended = blended / blended.sum(axis=1, keepdims=True)
            return blended

        # Generate predictions from base models
        meta_features = []
        binary_single_col = False
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1 and preds.shape[1] > 2:
                        meta_features.append(preds)
                    elif preds.ndim > 1:
                        meta_features.append(preds[:, 1])
                        binary_single_col = True
                    else:
                        meta_features.append(preds)
                        binary_single_col = True
                else:
                    meta_features.append(model.predict(X))
            else:
                meta_features.append(model.predict(X))

        # Create meta-features
        if meta_features and isinstance(meta_features[0], np.ndarray) and meta_features[0].ndim > 1:
            meta_X = np.concatenate(meta_features, axis=1)
        else:
            meta_X = np.column_stack(meta_features)

        # Predict with meta-model
        if meta_model is None:
            raise ValueError("Stacking meta_model is missing")

        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            preds = meta_model.predict_proba(meta_X)
            if binary_single_col and preds.ndim > 1:
                return preds[:, 1]
            return preds
        return meta_model.predict(meta_X)

    def predict_blending(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using blending ensemble.

        Args:
            ensemble: Ensemble dictionary with base_models and weights
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble["base_models"]
        weights = ensemble["weights"]

        predictions = []
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))
            else:
                predictions.append(model.predict(X))

        # Weighted average
        return np.average(predictions, axis=0, weights=weights)

    def plan_ensemble_strategy(
        self, models: list[Any], problem_type: str, eda_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Plan ensemble strategy using LLM."""
        import json

        from langchain_core.messages import HumanMessage

        from ..core.config import get_llm

        llm = get_llm()

        model_descriptions = []
        for i, m in enumerate(models):
            model_descriptions.append(f"Model {i + 1}: {type(m).__name__}")

        prompt = f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We have {len(models)} trained models: {", ".join(model_descriptions)}.
- Problem Type: {problem_type}
- EDA Insights: {str(eda_summary)[:500]}...

# Your task
- Suggest a plan to ensemble these solutions.
- The suggested plan should be novel, effective, and easy to implement.
        - Consider:
            1. "caruana_ensemble": Hill Climbing / Forward Selection (State of the Art).
            2. "stacking": Train a meta-model (LogisticRegression) on OOF predictions.
            3. "weighted_blending": Simple optimized weights.

# Response format
Return a JSON object:
{{
    "strategy_name": "caruana_ensemble" or "stacking_xgboost_meta" or "weighted_blending",
    "description": "Brief description of strategy",
    "meta_learner_config": {{ ... }} (if applicable)
}}
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "strategy_name": "weighted_blending",
                "description": "Fallback to weighted blending",
            }

    def __call__(self, state: KaggleState) -> KaggleState:
        """Create ensemble from trained models.

        Args:
            state: Current workflow state

        Returns:
            Updated state with ensemble model
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE AGENT: Creating Model Ensemble")
        print("=" * 60)

        errors = []
        if isinstance(state, dict):
            errors = list(state.get("errors", []) or [])

        # check for temporal ensemble opportunity first if in later iterations
        current_iteration = state.get("current_iteration", 0)
        submissions = state.get("submissions", []) if isinstance(state, dict) else state.submissions
        working_dir_value = (
            state.get("working_directory", "")
            if isinstance(state, dict)
            else state.working_directory
        )
        working_dir = Path(working_dir_value) if working_dir_value else Path()

        # If we have history, try temporal ensemble first (it's cheap and robust)
        # We do this at the END of the standard ensemble logic?
        # Actually, the user requirement is: "Create a final/iterative ensemble step"
        # Since EnsembleAgent runs *before* Submission, we replace the model's output with the blend.
        # BUT: The current iteration's model hasn't been submitted yet!
        # SO: We should blend [Current Model Predictions] + [Past Best Submissions].
        # The current code generates "submission.csv" from the current ensemble/model structure.
        # We can wrap that.

        try:
            # Handle both dict and dataclass state access
            models_trained = (
                state.get("models_trained", []) if isinstance(state, dict) else state.models_trained
            )
            train_data_path = (
                state.get("train_data_path", "")
                if isinstance(state, dict)
                else state.train_data_path
            )
            current_train_path = (
                state.get("current_train_path", "")
                if isinstance(state, dict)
                else getattr(state, "current_train_path", "")
            )
            current_test_path = (
                state.get("current_test_path", "")
                if isinstance(state, dict)
                else getattr(state, "current_test_path", "")
            )
            competition_name = (
                state.get("competition_name", "")
                if isinstance(state, dict)
                else state.competition_name
            )
            eda_summary = (
                state.get("eda_summary", {}) if isinstance(state, dict) else state.eda_summary
            )
            best_model = (
                state.get("best_model", {}) if isinstance(state, dict) else state.best_model
            )
            working_dir_value = (
                state.get("working_directory", "")
                if isinstance(state, dict)
                else state.working_directory
            )
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            test_data_path = (
                state.get("test_data_path", "") if isinstance(state, dict) else state.test_data_path
            )
            sample_submission_path = (
                state.get("sample_submission_path", "")
                if isinstance(state, dict)
                else state.sample_submission_path
            )
            models_dir = working_dir / "models"

            # Access metric name safely from competition_info
            comp_info = (
                state.get("competition_info")
                if isinstance(state, dict)
                else getattr(state, "competition_info", None)
            )
            metric_name = getattr(comp_info, "evaluation_metric", "") if comp_info else ""
            if isinstance(state, dict):
                baseline_score = state.get("best_single_model_score") or state.get(
                    "baseline_cv_score"
                )
            else:
                baseline_score = getattr(state, "best_single_model_score", None) or getattr(
                    state, "baseline_cv_score", None
                )
            baseline_score_val = (
                float(baseline_score)
                if isinstance(baseline_score, (int, float)) and np.isfinite(baseline_score)
                else None
            )

            def _should_revert_ensemble(ensemble_score: float | None) -> tuple[bool, str]:
                if baseline_score_val is None:
                    return False, "no_baseline"
                if ensemble_score is None:
                    return True, "missing_oof_score"
                try:
                    score_val = float(ensemble_score)
                except (TypeError, ValueError):
                    return True, "invalid_oof_score"
                if not np.isfinite(score_val):
                    return True, "non_finite_oof_score"
                is_minimize = is_metric_minimization(metric_name)
                if is_minimize:
                    return score_val >= baseline_score_val, "not_improved"
                return score_val <= baseline_score_val, "not_improved"

            def _load_oof_target() -> tuple[np.ndarray | None, str | None]:
                candidate_paths = [
                    current_train_path,
                    train_data_path,
                    str(working_dir / "train.csv"),
                ]
                train_path = next(
                    (Path(p) for p in candidate_paths if p and Path(p).exists()), None
                )
                if not train_path:
                    return None, None
                try:
                    train_df = pd.read_csv(train_path)
                except Exception as e:
                    print(f"   ⚠️ Failed to load train data for OOF gating: {e}")
                    return None, None
                sample_path = (
                    Path(sample_submission_path)
                    if sample_submission_path
                    else working_dir / "sample_submission.csv"
                )
                class_order: list[str] | None = None
                if sample_path.exists():
                    try:
                        sample_sub = pd.read_csv(sample_path)
                        class_order = sample_sub.columns[1:].tolist()
                        if class_order and all(col in train_df.columns for col in class_order):
                            # Ambiguous multi-label layout; skip gating to avoid wrong metric.
                            return None, None
                    except Exception:
                        class_order = None

                target_col = None
                for col in ("target", "label", train_df.columns[-1]):
                    if col in train_df.columns:
                        target_col = col
                        break
                if not target_col:
                    return None, None

                y_series = train_df[target_col]
                n_unique = y_series.nunique(dropna=True)
                unique_ratio = n_unique / max(len(y_series), 1)

                if class_order and len(class_order) > 1:
                    problem_type = "classification"
                elif y_series.dtype.kind in {"O", "U", "S"}:
                    problem_type = "classification"
                elif unique_ratio <= 0.2 and n_unique <= 200:
                    problem_type = "classification"
                else:
                    problem_type = "regression"

                if problem_type == "classification":
                    if y_series.dtype.kind in {"O", "U", "S"}:
                        if class_order:
                            mapping = {label: idx for idx, label in enumerate(class_order)}
                            mapped = y_series.map(mapping)
                            if mapped.isnull().any():
                                return None, None
                            y_vals = mapped.to_numpy()
                        else:
                            y_vals, _ = pd.factorize(y_series)
                    else:
                        y_vals = y_series.to_numpy()
                else:
                    y_vals = y_series.to_numpy()

                return y_vals, problem_type

            # DEBUG: Detailed information about available models
            dev_results = state.get("development_results", [])
            successful_results = [r for r in dev_results if r.success] if dev_results else []

            print("\n   📊 Ensemble Prerequisites Check:")
            print(f"      Total development results: {len(dev_results)}")
            print(f"      Successful results: {len(successful_results)}")
            print(f"      Models trained count: {len(models_trained)}")

            if successful_results:
                print("\n   ✅ Successful components:")
                for i, result in enumerate(successful_results[-5:], 1):  # Last 5
                    artifacts_str = (
                        ", ".join(result.artifacts_created[:3])
                        if result.artifacts_created
                        else "none"
                    )
                    print(f"      {i}. {artifacts_str}")

            accepted_component_names: set[str] = set()
            if isinstance(state, dict):
                for key in state.keys():
                    if isinstance(key, str) and key.startswith("component_result_"):
                        accepted_component_names.add(key.replace("component_result_", "", 1))

            accepted_oof_names: set[str] = set()
            if accepted_component_names:
                accepted_oof_names = {
                    name
                    for name in accepted_component_names
                    if (models_dir / f"oof_{name}.npy").exists()
                }

            if accepted_component_names:
                print(
                    "   ✅ Accepted components: " + ", ".join(sorted(accepted_component_names))
                )
            else:
                print("   ⚠️ No accepted component keys found")

            if accepted_oof_names:
                print(
                    "   ✅ Accepted model OOFs: " + ", ".join(sorted(accepted_oof_names))
                )
            else:
                print("   ⚠️ No accepted model OOF files found")

            accepted_names = accepted_oof_names
            has_acceptance_metadata = bool(accepted_component_names)
            accepted_filter_active = bool(accepted_names)
            if has_acceptance_metadata and len(accepted_names) < 2:
                print(
                    "\n   ⚠️  Not enough accepted model components for ensemble "
                    f"(have {len(accepted_names)}, need 2+)"
                )
                best_submission = working_dir / "submission_best.csv"
                if best_submission.exists():
                    shutil.copy(best_submission, working_dir / "submission.csv")
                    print("      ✅ Restored submission.csv from submission_best.csv")
                return {
                    "ensemble_skipped": True,
                    "skip_reason": (
                        f"insufficient_accepted_models (have {len(accepted_names)}, need 2+)"
                    ),
                }

            # Check if we have multiple models
            if len(models_trained) < 2:
                prediction_pairs = self._find_prediction_pairs(models_dir)
                best_single_model_name = (
                    state.get("best_single_model_name")
                    if isinstance(state, dict)
                    else getattr(state, "best_single_model_name", None)
                )

                if prediction_pairs and accepted_filter_active:
                    # Case 1: Filter by accepted models only
                    filtered = {
                        name: pair
                        for name, pair in prediction_pairs.items()
                        if name in accepted_names
                    }
                    dropped = set(prediction_pairs) - set(filtered)
                    if dropped:
                        print(
                            "   ⚠️ Excluding prediction pairs from non-accepted components: "
                            + ", ".join(sorted(dropped))
                        )
                    prediction_pairs = filtered
                elif prediction_pairs and not accepted_filter_active:
                    # Case 2: No accepted components - use ONLY best single model to avoid degradation
                    if best_single_model_name and best_single_model_name in prediction_pairs:
                        print(
                            f"\n   ⚠️ No accepted components found. Using best single model: {best_single_model_name}"
                        )
                        print("      Reason: Ensembling unvalidated models causes score degradation")

                        # Use only the best single model's predictions
                        _, test_path = prediction_pairs[best_single_model_name]
                        preds = np.load(test_path)
                        preds = np.asarray(preds, dtype=np.float32)

                        sample_path = (
                            Path(sample_submission_path)
                            if sample_submission_path
                            else working_dir / "sample_submission.csv"
                        )
                        if sample_path.exists():
                            sample_sub = pd.read_csv(sample_path)
                            if preds.ndim == 1:
                                sample_sub.iloc[:, 1] = preds
                            else:
                                sample_sub.iloc[:, 1:] = preds
                            sample_sub.to_csv(working_dir / "submission.csv", index=False)
                            print(f"   ✅ Using best single model predictions: {best_single_model_name}")
                            return {
                                "ensemble_skipped": True,
                                "skip_reason": "using_best_single_model",
                                "best_model_used": best_single_model_name,
                            }
                    else:
                        # No accepted components and no valid best_single_model - fall back to submission_best.csv
                        print("\n   ⚠️ No accepted components and no best_single_model_name tracked.")
                        print("      Falling back to submission_best.csv to avoid ensemble degradation.")
                        best_submission = working_dir / "submission_best.csv"
                        if best_submission.exists():
                            shutil.copy(best_submission, working_dir / "submission.csv")
                            print("      ✅ Restored submission.csv from submission_best.csv")
                        return {
                            "ensemble_skipped": True,
                            "skip_reason": "no_accepted_components_no_best_model",
                        }

                # Check for missing test files
                if prediction_pairs:
                    missing_tests = [
                        p.stem.replace("oof_", "", 1)
                        for p in models_dir.glob("oof_*.npy")
                        if not (models_dir / f"test_{p.stem.replace('oof_', '', 1)}.npy").exists()
                    ]
                    if missing_tests:
                        print(f"   ⚠️ Missing test_* for: {', '.join(missing_tests[:5])}")

                    # Only ensemble if we have 2+ ACCEPTED models
                    if len(prediction_pairs) >= 2:
                        print("\n   ✅ Using prediction-only ensemble from OOF/Test pairs")
                        output_path = working_dir / "submission.csv"
                        sample_path = (
                            Path(sample_submission_path)
                            if sample_submission_path
                            else working_dir / "sample_submission.csv"
                        )
                        if self._ensemble_from_predictions(
                            prediction_pairs, sample_path, output_path
                        ):
                            ensemble_oof_score = None
                            if baseline_score_val is not None:
                                y_vals, inferred_problem_type = _load_oof_target()
                                if y_vals is not None and inferred_problem_type:
                                    try:
                                        oof_list = [
                                            np.load(oof_path)
                                            for oof_path, _ in prediction_pairs.values()
                                        ]
                                        avg_oof = np.mean(oof_list, axis=0)
                                        ensemble_oof_score = self._score_predictions(
                                            avg_oof, y_vals, inferred_problem_type, metric_name
                                        )
                                        print(
                                            f"   📊 Prediction-only ensemble OOF score: {ensemble_oof_score:.6f}"
                                        )
                                    except Exception as e:
                                        print(
                                            f"   ⚠️ Failed to compute OOF score for prediction-only ensemble: {e}"
                                        )

                            revert, reason = _should_revert_ensemble(ensemble_oof_score)
                            if revert:
                                best_submission = working_dir / "submission_best.csv"
                                if best_submission.exists():
                                    shutil.copy(best_submission, working_dir / "submission.csv")
                                    print("      ✅ Restored submission.csv from submission_best.csv")
                                return {
                                    "ensemble_skipped": True,
                                    "skip_reason": reason,
                                    "ensemble_oof_score": ensemble_oof_score,
                                    "baseline_score": baseline_score_val,
                                }

                            return {
                                "ensemble_created": True,
                                "ensemble_method": "prediction_average",
                                "ensemble_oof_score": ensemble_oof_score,
                            }

                print(
                    f"\n   ⚠️  Not enough models for ensemble (need 2+, have {len(models_trained)})"
                )
                print(
                    "      Reason: Ensemble requires at least 2 trained models or 2 OOF/Test pairs"
                )
                best_submission = working_dir / "submission_best.csv"
                if best_submission.exists():
                    shutil.copy(best_submission, working_dir / "submission.csv")
                    print("      ✅ Restored submission.csv from submission_best.csv")
                print("      Skipping ensemble step")
                return {
                    "ensemble_skipped": True,
                    "skip_reason": f"insufficient_models (have {len(models_trained)}, need 2+)",
                }

            # Resolve train/test paths (prefer engineered data if available)
            resolved_train_path = (
                Path(current_train_path)
                if current_train_path
                else Path(train_data_path)
                if train_data_path
                else working_dir / "train.csv"
            )
            resolved_test_path = (
                Path(current_test_path)
                if current_test_path
                else Path(test_data_path)
                if test_data_path
                else working_dir / "test.csv"
            )

            print("\n   📂 Data Paths:")
            print(f"      Train: {resolved_train_path.name}")
            print(f"      Test:  {resolved_test_path.name}")
            if current_train_path:
                print("      ✅ Using engineered features (from feature_engineering component)")
            else:
                print("      📊 Using original raw features")

            if not resolved_train_path.exists():
                print(f"  ❌ Train data not found at {resolved_train_path}, skipping ensemble")
                return state

            # Load processed data
            train_df = pd.read_csv(resolved_train_path)

            # Identify target
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns:
                    target_col = col
                    break

            if not target_col:
                print("  Could not identify target column, skipping ensemble")
                return state

            # Separate features and target
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]

            # Determine problem type
            problem_type = "classification" if y.nunique() < 20 else "regression"

            # Prepare test features for submission generation
            test_features = None
            test_path = resolved_test_path
            if test_path.exists():
                try:
                    test_df = pd.read_csv(test_path)
                    missing_cols = [col for col in X.columns if col not in test_df.columns]
                    if missing_cols:
                        print(f"   ⚠️ Test data missing columns: {missing_cols} (filled with 0)")
                        for col in missing_cols:
                            test_df[col] = 0
                    test_features = test_df[X.columns]
                except Exception as e:
                    print(f"   ⚠️ Failed to load test data for ensemble predictions: {e}")
            else:
                print(f"   ⚠️ Test data not found at {test_path}, skipping submission generation")

            sample_sub_path = (
                Path(sample_submission_path)
                if sample_submission_path
                else working_dir / "sample_submission.csv"
            )
            train_ids = (
                train_df["id"].to_numpy() if "id" in train_df.columns else train_df.index.to_numpy()
            )
            expected_class_order = None
            n_targets = None
            if sample_sub_path.exists():
                try:
                    sample_head = pd.read_csv(sample_sub_path, nrows=1)
                    n_targets = sample_head.shape[1] - 1
                    if sample_head.shape[1] > 2:
                        expected_class_order = sample_head.columns[1:].tolist()
                except Exception as e:
                    print(f"   ⚠️ Failed to read sample submission for class order: {e}")

            folds_path = working_dir / "folds.csv"
            if not folds_path.exists():
                folds_path = None

            # Load top models (top 3 by CV score)
            print("\n   🔍 Loading top models for ensemble...")
            sorted_models = sorted(models_trained, key=lambda x: x["mean_cv_score"], reverse=True)[
                :3
            ]

            top_models = []
            top_model_names = []
            for i, model_info in enumerate(sorted_models, 1):
                model_path = f"{get_config().paths.models_dir}/{model_info['name']}_{competition_name}.joblib"
                print(
                    f"      Model {i}: {model_info['name']} (CV: {model_info['mean_cv_score']:.4f})"
                )

                if Path(model_path).exists():
                    model = joblib.load(model_path)
                    top_models.append(model)
                    top_model_names.append(model_info["name"])
                    print(f"         ✅ Loaded from {model_path}")
                else:
                    print(f"         ❌ Model file not found: {model_path}")

            if len(top_models) < 2:
                print("\n   ❌ Not enough trained models loaded for ensemble")
                print(f"      Required: 2+, Found: {len(top_models)}")
                return {
                    "ensemble_skipped": True,
                    "skip_reason": f"models_not_found (loaded {len(top_models)}, need 2+)",
                }

            # PLAN ENSEMBLE STRATEGY
            print("\n   🎯 Planning ensemble strategy...")
            plan = self.plan_ensemble_strategy(top_models, problem_type, eda_summary)
            ensemble_strategy = plan.get("strategy_name", "weighted_blending")
            print(f"      Strategy: {ensemble_strategy}")
            print(f"      Description: {plan.get('description', '')}")
            print(f"      Combining {len(top_models)} models using {ensemble_strategy}")

            # Create ensemble
            if "stack" in ensemble_strategy.lower():
                ensemble, final_preds = self.create_stacking_ensemble(
                    top_models,
                    top_model_names,
                    working_dir,
                    X,
                    y,
                    problem_type,
                    metric_name=metric_name,
                    sample_submission_path=sample_sub_path,
                    train_ids=train_ids,
                    expected_class_order=expected_class_order,
                    n_targets=n_targets,
                    folds_path=folds_path,
                )
                if final_preds is None and test_features is None:
                    print("  ⚠️ Skipping stacking submission because test features are unavailable")
                else:
                    if final_preds is None:
                        final_preds = self.predict_stacking(ensemble, test_features, problem_type)
                    if sample_sub_path.exists():
                        sub_df = pd.read_csv(sample_sub_path)
                        save_ok = True
                        if final_preds.ndim == 1:
                            sub_df.iloc[:, 1] = final_preds
                        else:
                            if final_preds.shape[1] != (len(sub_df.columns) - 1):
                                print(
                                    "  ⚠️ Prediction column mismatch: "
                                    f"preds={final_preds.shape[1]}, sample_cols={len(sub_df.columns) - 1}"
                                )
                                save_ok = False
                            else:
                                sub_df.iloc[:, 1:] = final_preds

                        if save_ok:
                            submission_path = working_dir / "submission.csv"
                            sub_df.to_csv(submission_path, index=False)
                            print(f"  ✅ Saved ensemble submission to {submission_path}")
                            ensemble_oof_score = (
                                ensemble.get("oof_score") if isinstance(ensemble, dict) else None
                            )
                            revert, reason = _should_revert_ensemble(ensemble_oof_score)
                            if revert:
                                best_submission = working_dir / "submission_best.csv"
                                if best_submission.exists():
                                    shutil.copy(best_submission, submission_path)
                                    print("      ✅ Restored submission.csv from submission_best.csv")
                                return {
                                    "ensemble_skipped": True,
                                    "skip_reason": reason,
                                    "ensemble_oof_score": ensemble_oof_score,
                                    "baseline_score": baseline_score_val,
                                }
                    else:
                        print(
                            f"  ⚠️ Sample submission not found at {sample_sub_path}, skipping submission save"
                        )

            elif "caruana" in ensemble_strategy.lower():
                ensemble = self.create_caruana_ensemble(
                    top_models, top_model_names, working_dir, X, y, problem_type, metric_name
                )

                # Generate Final Submission using Test Preds (Weighted Average)
                print("  Generating final submission from Caruana Ensemble...")
                valid_names = ensemble["base_model_names"]
                weights = np.array(ensemble["weights"], dtype=float)
                base_models = ensemble.get("base_models", [])
                test_meta_features = []
                used_weights = []
                missing_test_models = []

                for idx, (name, weight) in enumerate(zip(valid_names, weights, strict=False)):
                    preds = None
                    test_pred_path = working_dir / "models" / f"test_{name}.npy"
                    if test_pred_path.exists():
                        preds = np.load(test_pred_path)
                    elif test_features is not None:
                        model = base_models[idx] if idx < len(base_models) else None
                        if model is not None:
                            try:
                                if problem_type == "classification" and hasattr(
                                    model, "predict_proba"
                                ):
                                    model_preds = model.predict_proba(test_features)
                                    preds = (
                                        model_preds[:, 1] if model_preds.ndim > 1 else model_preds
                                    )
                                else:
                                    preds = model.predict(test_features)
                            except Exception as e:
                                print(f"  ⚠️ Failed to generate test preds for {name}: {e}")
                    if preds is not None:
                        test_meta_features.append(preds)
                        used_weights.append(weight)
                    else:
                        missing_test_models.append(name)

                if missing_test_models:
                    print(f"  ⚠️ Missing test predictions for: {', '.join(missing_test_models)}")

                if test_meta_features:
                    weight_array = np.array(used_weights, dtype=float)
                    weight_sum = weight_array.sum()
                    if weight_sum <= 0:
                        weight_array = np.ones_like(weight_array) / len(weight_array)
                    else:
                        weight_array = weight_array / weight_sum
                    final_preds = np.average(test_meta_features, axis=0, weights=weight_array)

                    # Save submission
                    if sample_sub_path.exists():
                        sub_df = pd.read_csv(sample_sub_path)
                        sub_df.iloc[:, 1] = final_preds
                        submission_path = working_dir / "submission.csv"
                        sub_df.to_csv(submission_path, index=False)
                        print(f"  ✅ Saved ensemble submission to {submission_path}")
                        ensemble_oof_score = (
                            ensemble.get("oof_score") if isinstance(ensemble, dict) else None
                        )
                        revert, reason = _should_revert_ensemble(ensemble_oof_score)
                        if revert:
                            best_submission = working_dir / "submission_best.csv"
                            if best_submission.exists():
                                shutil.copy(best_submission, submission_path)
                                print("      ✅ Restored submission.csv from submission_best.csv")
                            return {
                                "ensemble_skipped": True,
                                "skip_reason": reason,
                                "ensemble_oof_score": ensemble_oof_score,
                                "baseline_score": baseline_score_val,
                            }
                    else:
                        print(
                            f"  ⚠️ Sample submission not found at {sample_sub_path}, skipping submission save"
                        )
                else:
                    print(
                        "  ❌ No test predictions available for Caruana ensemble, skipping submission."
                    )

            else:
                ensemble = self.create_blending_ensemble(top_models, X, y, problem_type)

            # Evaluate ensemble
            print("  Evaluating ensemble performance...")
            if "stack" in ensemble_strategy.lower():
                # Optimistic estimate for stacking
                best_model.get("mean_cv_score", 0.0) * 1.01
            else:
                # For blending, we already calculated OOF loss during optimization
                pass

            # Save ensemble
            ensemble_path = f"{get_config().paths.models_dir}/ensemble_{competition_name}.joblib"
            joblib.dump(
                {
                    "ensemble": ensemble,
                    "problem_type": problem_type,
                    "strategy": ensemble_strategy,
                    "plan": plan,
                },
                ensemble_path,
            )

            print(
                f"Ensemble Agent: Created {ensemble_strategy} ensemble with {len(top_models)} models"
            )

            return {
                "best_model": {
                    "name": f"ensemble_{ensemble_strategy}",
                    "path": ensemble_path,
                    "mean_cv_score": best_model.get("mean_cv_score", 0.0),
                    "is_ensemble": True,
                }
            }

        except Exception as e:
            error_msg = f"Ensemble creation failed: {e!s}"
            print(f"Ensemble Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            # TEMPORAL ENSEMBLE STEP (Final Boost)
            # After generating the current iteration's "submission.csv" (either from stacking, caruana, or just best model)
            # We explicitly check if we can improve it by blending with history.
            if current_iteration > 1:
                # Ensure current submission is considered as a candidate
                # We temporarily save it as "current_candidate.csv" to be picked up?
                # Or we just pass the path.
                # Actually, create_temporal_ensemble scans for submission_iter_*.
                # The current one is just "submission.csv".
                # Let's simple copy current submission.csv to a temp name so it's included in the blend logic
                # as a "candidate" (maybe with assumed high score?).
                # Actually, simpler: Just run temporal ensemble. If it finds enough history, it overwrites submission.csv.
                # The newly generated submission.csv effectively becomes valid for this iteration.
                current_sub = working_dir / "submission.csv"
                if current_sub.exists():
                    # Give it a temp name to be picked up by the scanner?
                    # Scanner looks for "submission_iter_*.csv".
                    # We create a fake one representing "current"
                    temp_current = working_dir / f"submission_iter_{current_iteration}_current.csv"
                    shutil.copy2(current_sub, temp_current)

                self.create_temporal_ensemble(
                    working_dir, submissions, current_iteration, metric_name
                )

            errors.append(error_msg)
            return {
                "errors": errors,
                "ensemble_skipped": True,
                "skip_reason": "exception",
            }


def ensemble_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for ensemble agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = EnsembleAgent()
    return agent(state)
