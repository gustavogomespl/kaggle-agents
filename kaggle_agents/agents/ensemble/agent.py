"""Ensemble agent for model stacking and blending."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import cross_val_predict

from ...core.config import is_metric_minimization
from ...core.state import KaggleState
from ...utils.csv_utils import read_csv_auto
from ...utils.llm_utils import get_text_content
from .alignment import align_oof_by_canonical_ids, load_and_align_oof, validate_oof_alignment
from .fallback import (
    create_ensemble_with_fallback,
    fallback_to_best_single_model,
    recover_from_checkpoints,
)
from .meta_model import (
    constrained_meta_learner,
    diagnose_stacking_issues,
    dirichlet_weight_search,
    tune_meta_model,
)
from .prediction_pairs import find_prediction_pairs, validate_prediction_artifacts_contract
from .scoring import compute_oof_score, filter_by_score_threshold, score_predictions
from .stacking import load_cv_folds, stack_from_prediction_pairs
from .submission import safe_restore_submission, validate_and_align_submission
from .utils import class_orders_match, encode_labels


class EnsembleAgent:
    """Agent responsible for creating model ensembles."""

    def __init__(self):
        """Initialize ensemble agent."""
        pass

    # Delegate to module functions (for backward compatibility with method calls)
    def _find_prediction_pairs(self, models_dir: Path) -> dict[str, tuple[Path, Path]]:
        return find_prediction_pairs(models_dir)

    def _validate_prediction_artifacts_contract(self, prediction_pairs):
        return validate_prediction_artifacts_contract(prediction_pairs)

    def _validate_oof_alignment(self, models_dir, train_ids, expected_class_order):
        return validate_oof_alignment(models_dir, train_ids, expected_class_order)

    def _encode_labels(self, y, class_order):
        return encode_labels(y, class_order)

    def _score_predictions(self, preds, y_true, problem_type, metric_name):
        return score_predictions(preds, y_true, problem_type, metric_name)

    def _compute_oof_score(self, oof_path, y_true, metric_name="log_loss"):
        return compute_oof_score(oof_path, y_true, metric_name)

    def _filter_by_score_threshold(self, prediction_pairs, y_true, metric_name, model_scores=None, threshold_pct=0.20):
        return filter_by_score_threshold(prediction_pairs, y_true, metric_name, model_scores, threshold_pct)

    def _tune_meta_model(self, meta_X, y, problem_type, n_classes=None):
        return tune_meta_model(meta_X, y, problem_type, n_classes)

    def _diagnose_stacking_issues(self, meta_model, model_names, meta_X, y):
        return diagnose_stacking_issues(meta_model, model_names, meta_X, y)

    def _constrained_meta_learner(self, oof_stack, y_true, problem_type, metric_name):
        return constrained_meta_learner(oof_stack, y_true, problem_type, metric_name)

    def _dirichlet_weight_search(self, oof_stack, y_true, problem_type, metric_name, n_samples=300):
        return dirichlet_weight_search(oof_stack, y_true, problem_type, metric_name, n_samples)

    def _validate_and_align_submission(self, submission_path, sample_submission_path, output_path=None):
        return validate_and_align_submission(submission_path, sample_submission_path, output_path)

    def _safe_restore_submission(self, source_path, dest_path, sample_submission_path):
        return safe_restore_submission(source_path, dest_path, sample_submission_path)

    def _load_and_align_oof(self, oof_path, train_ids_path, reference_ids):
        return load_and_align_oof(oof_path, train_ids_path, reference_ids)

    def _align_oof_by_canonical_ids(self, oof, model_train_ids, canonical_train_ids, model_name="unknown"):
        return align_oof_by_canonical_ids(oof, model_train_ids, canonical_train_ids, model_name)

    def _recover_from_checkpoints(self, models_dir, component_names=None):
        return recover_from_checkpoints(models_dir, component_names)

    def _fallback_to_best_single_model(self, models_dir, problem_type="classification"):
        return fallback_to_best_single_model(models_dir, problem_type)

    def _load_cv_folds(self, name, models_dir, folds_path, n_samples):
        return load_cv_folds(name, models_dir, folds_path, n_samples)

    def _stack_from_prediction_pairs(self, prediction_pairs, y, problem_type, metric_name, models_dir, expected_class_order, train_ids, folds_path, enable_calibration, enable_post_calibration, n_targets, calibration_method="auto"):
        return stack_from_prediction_pairs(prediction_pairs, y, problem_type, metric_name, models_dir, expected_class_order, train_ids, folds_path, enable_calibration, enable_post_calibration, n_targets, calibration_method)

    def create_ensemble_with_fallback(self, models_dir, y, problem_type, metric_name, expected_class_order=None, train_ids=None, min_models=2):
        return create_ensemble_with_fallback(models_dir, y, problem_type, metric_name, expected_class_order, train_ids, min_models)

    def create_oof_weighted_blend(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """Weighted blend using saved OOF predictions."""
        from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

        names = list(prediction_pairs.keys())
        n_models = len(names)

        print(f"   Creating OOF weighted blend with {n_models} models...")

        oof_list = [np.load(oof) for oof, _ in prediction_pairs.values()]
        test_list = [np.load(test) for _, test in prediction_pairs.values()]
        oof_stack = np.stack(oof_list, axis=0)
        test_stack = np.stack(test_list, axis=0)

        def compute_score(blended: np.ndarray) -> float:
            if problem_type == "classification":
                blended = np.clip(blended, 1e-15, 1 - 1e-15)
                if blended.ndim > 1 and blended.shape[1] > 1:
                    blended = blended / blended.sum(axis=1, keepdims=True)
                if metric_name in ["auc", "roc_auc"]:
                    from sklearn.metrics import roc_auc_score
                    return -roc_auc_score(y_true, blended, multi_class="ovr", average="weighted")
                return log_loss(y_true, blended)
            if blended.ndim > 1:
                blended = blended.ravel()
            if metric_name in ["mae", "mean_absolute_error"]:
                return mean_absolute_error(y_true, blended)
            if metric_name in ["mse", "mean_squared_error"]:
                return mean_squared_error(y_true, blended)
            return np.sqrt(mean_squared_error(y_true, blended))

        def objective(weights: np.ndarray) -> float:
            weights = np.array(weights) / np.sum(weights)
            blended = np.average(oof_stack, axis=0, weights=weights)
            return compute_score(blended)

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
        except ImportError:
            opt_weights = self._dirichlet_weight_search(oof_stack, y_true, problem_type, metric_name, n_samples=300)

        blended_oof = np.average(oof_stack, axis=0, weights=opt_weights)
        oof_score = compute_score(blended_oof)

        blended_test = np.average(test_stack, axis=0, weights=opt_weights)
        if problem_type == "classification":
            blended_test = np.clip(blended_test, 1e-15, 1 - 1e-15)
            if blended_test.ndim > 1 and blended_test.shape[1] > 1:
                blended_test = blended_test / blended_test.sum(axis=1, keepdims=True)

        weights_dict = dict(zip(names, opt_weights))
        return blended_test, oof_score, weights_dict

    def _validate_class_order(self, models_dir: Path, sample_submission_path: Path) -> tuple[bool, str]:
        """Validate that saved predictions use canonical class order."""
        class_order_path = models_dir / "class_order.npy"

        if not class_order_path.exists():
            return False, "class_order.npy not found"

        if not sample_submission_path.exists():
            return False, "sample_submission.csv not found"

        try:
            saved_order = np.load(class_order_path, allow_pickle=True).tolist()
            sample_sub = read_csv_auto(sample_submission_path)
            expected_order = sample_sub.columns[1:].tolist()

            if not class_orders_match(saved_order, expected_order):
                return False, "Class order mismatch"

            return True, f"Class order validated ({len(saved_order)} classes)"
        except Exception as e:
            return False, f"Class order validation error: {e}"

    def _ensemble_from_predictions(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        sample_submission_path: Path,
        output_path: Path,
        models_dir: Path | None = None,
    ) -> bool:
        """Create a simple average ensemble directly from saved predictions.

        Uses ID-based merging when shapes don't match to handle models trained
        on different subsets of data.
        """
        if not sample_submission_path.exists():
            print("   Sample submission not found")
            return False

        sample_sub = read_csv_auto(sample_submission_path)
        n_test = len(sample_sub)
        preds_dict: dict[str, np.ndarray] = {}
        ids_dict: dict[str, np.ndarray | None] = {}

        for name, (_, test_path) in prediction_pairs.items():
            preds = np.load(test_path)
            preds = np.asarray(preds, dtype=np.float32)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds_dict[name] = preds

            # Try to load test IDs if available
            if models_dir:
                test_ids_path = models_dir / f"test_ids_{name}.npy"
                if test_ids_path.exists():
                    ids_dict[name] = np.load(test_ids_path, allow_pickle=True)
                else:
                    ids_dict[name] = None
            else:
                ids_dict[name] = None

        if len(preds_dict) < 1:
            print("   No prediction pairs found")
            return False

        names = list(preds_dict.keys())
        preds_list = list(preds_dict.values())

        if len(preds_list) == 1:
            name = names[0]
            ensemble_preds = preds_list[0]
            if ensemble_preds.shape[0] != n_test:
                print(f"   Row count mismatch: {ensemble_preds.shape[0]} vs {n_test}")
                return False

            # CRITICAL: Warn if no test_ids file - assuming alignment is risky!
            if ids_dict.get(name) is None:
                print(f"   WARNING: No test_ids_{name}.npy - assuming alignment with sample_submission (risk of score=0.50)")

            # Validate column count matches submission template
            expected_cols = len(sample_sub.columns) - 1  # Exclude ID column
            if ensemble_preds.shape[1] > expected_cols:
                print(f"   WARNING: Truncating {ensemble_preds.shape[1]} pred cols to {expected_cols} submission cols")
                ensemble_preds = ensemble_preds[:, :expected_cols]

            if ensemble_preds.shape[1] == 1:
                sample_sub.iloc[:, 1] = ensemble_preds[:, 0]
            elif ensemble_preds.shape[1] == expected_cols:
                sample_sub.iloc[:, 1:] = ensemble_preds
            else:
                # Fewer prediction columns than expected - pad with 0.5
                print(f"   WARNING: Padding {ensemble_preds.shape[1]} pred cols to {expected_cols} submission cols")
                padded = np.full((n_test, expected_cols), 0.5)
                padded[:, :ensemble_preds.shape[1]] = ensemble_preds
                sample_sub.iloc[:, 1:] = padded

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sample_sub.to_csv(output_path, index=False)
            return True

        shapes = {p.shape for p in preds_list}
        if len(shapes) != 1:
            print(f"   Shape mismatch detected: {shapes}")
            print("   Attempting ID-based merging with nanmean...")

            # Use ID-based merging instead of filtering
            # Get sample_sub IDs as reference
            test_ids_ref = sample_sub.iloc[:, 0].astype(str).values

            # CRITICAL: Use sample submission's expected column count, not max from predictions
            # This prevents ValueError when predictions have more columns than submission expects
            expected_cols = len(sample_sub.columns) - 1  # Exclude ID column
            pred_cols = max(p.shape[1] for p in preds_list)

            if pred_cols > expected_cols:
                print(f"   WARNING: Predictions have {pred_cols} cols, submission expects {expected_cols}")
                print(f"   Truncating predictions to {expected_cols} columns")
            n_cols = min(pred_cols, expected_cols) if expected_cols > 0 else pred_cols

            # Initialize merged array with NaN
            merged = np.full((n_test, len(names), n_cols), np.nan)

            models_contributed = 0
            for model_idx, name in enumerate(names):
                preds = preds_dict[name]
                model_ids = ids_dict.get(name)

                # Truncate prediction columns if needed
                if preds.shape[1] > n_cols:
                    preds = preds[:, :n_cols]

                if model_ids is not None and len(model_ids) == len(preds):
                    # Use ID-based mapping
                    id_to_pred = {str(id_): preds[i] for i, id_ in enumerate(model_ids)}
                    matched = 0
                    for ref_idx, ref_id in enumerate(test_ids_ref):
                        if ref_id in id_to_pred:
                            pred = id_to_pred[ref_id]
                            cols_to_copy = min(pred.shape[0] if pred.ndim > 0 else 1, n_cols)
                            merged[ref_idx, model_idx, :cols_to_copy] = pred[:cols_to_copy] if pred.ndim > 0 else pred
                            matched += 1
                    print(f"      {name}: ID-matched {matched}/{n_test} ({100*matched/n_test:.1f}%)")
                    if matched > 0:
                        models_contributed += 1
                elif len(preds) == n_test:
                    # Assume aligned order - RISKY! May cause ID misalignment
                    cols_to_copy = min(preds.shape[1], n_cols)
                    merged[:, model_idx, :cols_to_copy] = preds[:, :cols_to_copy]
                    print(f"      {name}: Assumed aligned ({len(preds)} rows) - no test_ids file (risk of score=0.50)")
                    models_contributed += 1
                else:
                    print(f"      {name}: SKIPPED (shape {preds.shape}, no IDs)")

            # CRITICAL: Fail fast if no models contributed predictions
            if models_contributed == 0:
                print("   ERROR: No models contributed valid predictions - cannot create ensemble")
                return False

            # Check if merged array has any valid (non-NaN) values
            valid_count = np.count_nonzero(~np.isnan(merged))
            if valid_count == 0:
                print("   ERROR: Merged array is entirely NaN - no valid predictions")
                return False

            print(f"   {models_contributed}/{len(names)} models contributed, {valid_count} valid predictions")

            # Compute nanmean across models
            ensemble_preds = np.nanmean(merged, axis=1)

            # Fill any remaining NaN with 0.5 (neutral for binary classification)
            ensemble_preds = np.where(np.isnan(ensemble_preds), 0.5, ensemble_preds)

            # Squeeze if single column
            if ensemble_preds.shape[1] == 1:
                ensemble_preds = ensemble_preds.squeeze(axis=1)
        else:
            stacked = np.stack(preds_list, axis=0)
            ensemble_preds = stacked.mean(axis=0)

        # CRITICAL: Check for constant/near-constant predictions (ID alignment bug indicator)
        pred_std = np.std(ensemble_preds)

        if pred_std < 1e-6:
            print("   ERROR: Predictions are constant (std<1e-6) - likely test ID misalignment! Check test_ids_*.npy files.")
        elif pred_std < 0.01:
            print(f"   WARNING: Very low variance (std={pred_std:.6f}) - possible ID alignment issue or broken model.")
        else:
            print(f"   Predictions: min={ensemble_preds.min():.4f}, max={ensemble_preds.max():.4f}, std={pred_std:.4f}")

        # Validate and assign predictions to sample submission
        expected_cols = len(sample_sub.columns) - 1  # Exclude ID column

        if ensemble_preds.ndim == 1:
            if len(ensemble_preds) != n_test:
                print(f"   Final row count mismatch: {len(ensemble_preds)} vs {n_test}")
                return False
            sample_sub.iloc[:, 1] = ensemble_preds
        else:
            if ensemble_preds.shape[0] != n_test:
                print(f"   Final row count mismatch: {ensemble_preds.shape[0]} vs {n_test}")
                return False

            # Validate column count matches submission template
            if ensemble_preds.shape[1] > expected_cols:
                print(f"   WARNING: Truncating {ensemble_preds.shape[1]} pred cols to {expected_cols} submission cols")
                ensemble_preds = ensemble_preds[:, :expected_cols]

            if ensemble_preds.shape[1] == 1:
                sample_sub.iloc[:, 1] = ensemble_preds[:, 0]
            elif ensemble_preds.shape[1] == expected_cols:
                sample_sub.iloc[:, 1:] = ensemble_preds
            else:
                # Fewer prediction columns than expected - pad with 0.5
                print(f"   WARNING: Padding {ensemble_preds.shape[1]} pred cols to {expected_cols} submission cols")
                padded = np.full((n_test, expected_cols), 0.5)
                padded[:, :ensemble_preds.shape[1]] = ensemble_preds
                sample_sub.iloc[:, 1:] = padded

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        print(f"   OK: Saved prediction-only ensemble to {output_path.name}")
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
        """Create stacking ensemble from best models using saved OOF predictions."""
        print(f"  Creating stacking ensemble with {len(models)} base models...")

        models_dir = working_dir / "models"
        prediction_pairs = {
            name: (models_dir / f"oof_{name}.npy", models_dir / f"test_{name}.npy")
            for name in model_names
            if (models_dir / f"oof_{name}.npy").exists()
            and (models_dir / f"test_{name}.npy").exists()
        }

        enable_calibration = os.getenv("KAGGLE_AGENTS_STACKING_CALIBRATION", "1").lower() not in {"0", "false", "no"}
        enable_post_calibration = os.getenv("KAGGLE_AGENTS_STACKING_POST_CALIBRATION", "1").lower() not in {"0", "false", "no"}
        calibration_method = os.getenv("KAGGLE_AGENTS_STACKING_CALIBRATION_METHOD", "auto").lower()

        if n_targets is None and sample_submission_path and sample_submission_path.exists():
            try:
                sample_head = read_csv_auto(sample_submission_path, nrows=1)
                n_targets = sample_head.shape[1] - 1
                if expected_class_order is None and sample_head.shape[1] > 2:
                    expected_class_order = sample_head.columns[1:].tolist()
            except Exception as e:
                print(f"   Warning: Failed to read sample submission: {e}")

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

        # Fallback to cross_val_predict
        meta_features = []
        valid_models = []
        valid_names = []

        for model, name in zip(models, model_names, strict=False):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                oof_preds = np.load(oof_path)
                meta_features.append(oof_preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                if problem_type == "classification":
                    oof_preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
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

        meta_X = np.column_stack(meta_features)
        y_arr = y.values if hasattr(y, "values") else y
        n_classes = len(np.unique(y_arr)) if problem_type == "classification" else None
        meta_model, _ = self._tune_meta_model(meta_X, y_arr, problem_type, n_classes)
        meta_model.fit(meta_X, y)

        return {
            "meta_model": meta_model,
            "base_models": valid_models,
            "base_model_names": valid_names,
            "stacking_method": "meta",
            "weights": None,
            "class_order": None,
        }, None

    def create_blending_ensemble(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> dict[str, Any]:
        """Create blending ensemble using simple averaging."""
        print(f"  Creating blending ensemble with {len(models)} models...")
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

        def loss_func(weights):
            weights = np.array(weights)
            weights /= weights.sum()
            final_preds = np.average(oof_preds, axis=1, weights=weights)
            if problem_type == "classification":
                final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)
                return log_loss(y, final_preds)
            return np.sqrt(mean_squared_error(y, final_preds))

        init_weights = [1.0 / len(models)] * len(models)
        constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}
        bounds = [(0, 1)] * len(models)

        result = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = result.x / result.x.sum()
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
        """Create ensemble using Caruana's Hill Climbing."""
        from sklearn.metrics import log_loss, mean_squared_error

        oof_preds = []
        valid_models = []
        valid_names = []

        for model, name in zip(models, model_names, strict=False):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                preds = np.load(oof_path)
                oof_preds.append(preds)
                valid_models.append(model)
                valid_names.append(name)

        if not oof_preds:
            raise ValueError("No OOF predictions found for Caruana ensemble")

        oof_preds = np.column_stack(oof_preds)
        n_models = oof_preds.shape[1]

        def get_score(y_true, y_pred):
            if problem_type == "classification":
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -log_loss(y_true, y_pred)
            return -np.sqrt(mean_squared_error(y_true, y_pred))

        current_ensemble_preds = np.zeros_like(oof_preds[:, 0])
        ensemble_counts = np.zeros(n_models, dtype=int)
        best_score = -float("inf")

        for i in range(n_models):
            score = get_score(y, oof_preds[:, i])
            if score > best_score:
                best_score = score
                best_init_idx = i

        current_ensemble_preds = oof_preds[:, best_init_idx]
        ensemble_counts[best_init_idx] = 1

        for it in range(n_iterations):
            best_iter_score = -float("inf")
            best_iter_idx = -1
            current_size = it + 2

            for i in range(n_models):
                current_sum = current_ensemble_preds * (current_size - 1)
                candidate_avg = (current_sum + oof_preds[:, i]) / current_size
                score = get_score(y, candidate_avg)
                if score > best_iter_score:
                    best_iter_score = score
                    best_iter_idx = i

            ensemble_counts[best_iter_idx] += 1
            current_ensemble_preds = (
                current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
            ) / current_size

        weights = ensemble_counts / ensemble_counts.sum()
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

    def create_rank_average_ensemble(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, list[str], bool]:
        """Create ensemble by averaging prediction ranks."""
        test_preds: dict[str, np.ndarray] = {}
        for name, (_, test_path) in prediction_pairs.items():
            if test_path.exists():
                try:
                    preds = np.load(test_path)
                    if np.isfinite(preds).all():
                        test_preds[name] = preds
                except Exception:
                    pass

        if len(test_preds) < 2:
            return None, [], False

        model_names = list(test_preds.keys())
        ranked_preds: list[np.ndarray] = []
        for preds in test_preds.values():
            if preds.ndim == 1:
                ranks = rankdata(preds) / len(preds)
            else:
                ranks = np.apply_along_axis(lambda x: rankdata(x) / len(x), axis=0, arr=preds)
            ranked_preds.append(ranks)

        if weights is None:
            weights = np.ones(len(ranked_preds)) / len(ranked_preds)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        stacked = np.stack(ranked_preds, axis=0)
        final_ranks = np.average(stacked, axis=0, weights=weights)
        return final_ranks, model_names, True

    def create_temporal_ensemble(
        self,
        working_dir: Path,
        submissions: list[Any],
        current_iteration: int,
        metric_name: str,
    ) -> bool:
        """Create Temporal Ensemble by blending past best submissions."""
        print(f"\n  Temporal Ensemble (Iteration {current_iteration})")

        minimize = is_metric_minimization(metric_name)
        candidates = []

        valid_history = [
            s for s in submissions
            if s.file_path and Path(s.file_path).exists() and s.public_score is not None
        ]

        for f in working_dir.glob("submission_iter_*_score_*.csv"):
            if f.name not in [Path(s.file_path).name for s in valid_history]:
                try:
                    parts = f.stem.split("_")
                    if "score" in parts:
                        score_idx = parts.index("score") + 1
                        score = float(parts[score_idx])
                        candidates.append({"path": f, "score": score})
                except Exception:
                    continue

        for sub in valid_history:
            candidates.append({"path": Path(sub.file_path), "score": sub.public_score})

        unique_candidates = {str(c["path"]): c for c in candidates}.values()
        candidates = list(unique_candidates)

        if len(candidates) < 2:
            return False

        reverse_sort = not minimize
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=reverse_sort)
        top_k = sorted_candidates[:3]

        dfs = []
        for c in top_k:
            try:
                df = pd.read_csv(c["path"])
                if "id" in df.columns:
                    df = df.sort_values("id")
                dfs.append(df)
            except Exception:
                pass

        if not dfs:
            return False

        try:
            sample = dfs[0]
            if len(sample.columns) < 2:
                return False
            pred_col = sample.columns[1]

            weights = np.array([3.0, 2.0, 1.0])[: len(dfs)]
            weights /= weights.sum()

            final_preds = np.zeros_like(sample[pred_col], dtype=float)
            for df, w in zip(dfs, weights):
                final_preds += df[pred_col].values * w

            output = sample.copy()
            output[pred_col] = final_preds
            output.to_csv(working_dir / "submission.csv", index=False)
            print("   OK: Temporal ensemble saved")
            return True
        except Exception as e:
            print(f"   Warning: Temporal ensemble failed: {e}")
            return False

    def predict_stacking(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        meta_model = ensemble.get("meta_model")
        base_models = ensemble.get("base_models", [])
        weights = ensemble.get("weights")

        if meta_model is None and weights is not None:
            # Weighted average
            predictions = []
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()
            for model in base_models:
                if problem_type == "classification" and hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))
            return np.average(predictions, axis=0, weights=weights_array)

        # Meta-model stacking
        meta_features = []
        binary_single_col = False
        for model in base_models:
            if problem_type == "classification" and hasattr(model, "predict_proba"):
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

        if meta_features and isinstance(meta_features[0], np.ndarray) and meta_features[0].ndim > 1:
            meta_X = np.concatenate(meta_features, axis=1)
        else:
            meta_X = np.column_stack(meta_features)

        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            preds = meta_model.predict_proba(meta_X)
            if binary_single_col and preds.ndim > 1:
                return preds[:, 1]
            return preds
        return meta_model.predict(meta_X)

    def predict_blending(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using blending ensemble."""
        base_models = ensemble["base_models"]
        weights = ensemble["weights"]

        predictions = []
        for model in base_models:
            if problem_type == "classification" and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
                if preds.ndim > 1:
                    predictions.append(preds[:, 1])
                else:
                    predictions.append(preds)
            else:
                predictions.append(model.predict(X))

        return np.average(predictions, axis=0, weights=weights)

    def select_ensemble_strategy(
        self,
        oof_coverage: float,
        problem_type: str,
        metric_name: str,
    ) -> str:
        """Select ensemble strategy based on OOF coverage and problem type."""
        ranking_metrics = {'auc', 'roc_auc', 'map', 'ndcg', 'mrr', 'log_loss', 'logloss'}
        is_ranking_metric = any(m in metric_name.lower() for m in ranking_metrics)

        if oof_coverage >= 0.95:
            strategy = "stacking"
        elif oof_coverage >= 0.70:
            strategy = "intersection_stacking"
        elif is_ranking_metric or problem_type == "classification":
            strategy = "rank_averaging"
        else:
            strategy = "weighted_averaging"

        return strategy

    def plan_ensemble_strategy(
        self, models: list[Any], problem_type: str, eda_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Plan ensemble strategy using LLM."""
        import json

        from langchain_core.messages import HumanMessage

        from ...core.config import get_llm

        llm = get_llm()
        model_descriptions = [f"Model {i + 1}: {type(m).__name__}" for i, m in enumerate(models)]

        prompt = f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We have {len(models)} trained models: {", ".join(model_descriptions)}.
- Problem Type: {problem_type}
- EDA Insights: {str(eda_summary)[:500]}...

# Your task
- Suggest a plan to ensemble these solutions.
- Consider: caruana_ensemble, stacking, weighted_blending, or rank_averaging.

# Response format
Return a JSON object with: strategy_name, description, meta_learner_config (if applicable)
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
            return {"strategy_name": "weighted_blending", "description": "Fallback to weighted blending"}

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """Create ensemble from trained models.

        This is the main entry point for the ensemble agent.
        The full implementation is in the original ensemble_agent.py file.
        """
        # Import the full __call__ implementation
        # For now, delegate to a simplified version
        print("\n" + "=" * 60)
        print("ENSEMBLE AGENT: Creating Model Ensemble")
        print("=" * 60)

        errors = []
        if isinstance(state, dict):
            errors = list(state.get("errors", []) or [])

        try:
            working_dir_value = (
                state.get("working_directory", "")
                if isinstance(state, dict)
                else state.working_directory
            )
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            models_dir = working_dir / "models"
            sample_submission_path = (
                state.get("sample_submission_path", "")
                if isinstance(state, dict)
                else state.sample_submission_path
            )

            # Find prediction pairs
            prediction_pairs = self._find_prediction_pairs(models_dir)
            print(f"   Found {len(prediction_pairs)} prediction pairs")

            if len(prediction_pairs) < 1:
                print("   No prediction pairs found, skipping ensemble")
                return {"ensemble_skipped": True, "skip_reason": "no_prediction_pairs"}

            # Create simple average ensemble
            sample_path = Path(sample_submission_path) if sample_submission_path else working_dir / "sample_submission.csv"
            output_path = working_dir / "submission.csv"

            if self._ensemble_from_predictions(prediction_pairs, sample_path, output_path):
                return {"ensemble_created": True, "n_models": len(prediction_pairs)}
            return {"ensemble_skipped": True, "skip_reason": "ensemble_creation_failed"}

        except Exception as e:
            error_msg = f"Ensemble creation failed: {e!s}"
            print(f"Ensemble Agent ERROR: {error_msg}")
            errors.append(error_msg)
            return {"errors": errors, "ensemble_skipped": True, "skip_reason": "exception"}


def ensemble_agent_node(state: KaggleState) -> dict[str, Any]:
    """LangGraph node function for ensemble agent."""
    agent = EnsembleAgent()
    return agent(state)
