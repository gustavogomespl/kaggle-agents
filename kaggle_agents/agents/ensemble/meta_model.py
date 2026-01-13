"""Meta-model tuning and diagnostics for stacking ensembles."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from .scoring import score_predictions


def tune_meta_model(
    meta_X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    n_classes: int | None = None,
) -> tuple[LogisticRegression | Ridge, str]:
    """Tune meta-model regularization using cross-validation.

    Uses grid search over C (classification) or alpha (regression) values
    with explicit StratifiedKFold/KFold and proper clip/normalize before log_loss.

    Args:
        meta_X: Meta-feature matrix (stacked OOF predictions)
        y: Target values (encoded for classification)
        problem_type: "classification" or "regression"
        n_classes: Number of classes (for multiclass handling)

    Returns:
        Tuple of (fitted_model, metric_name)
    """
    # Detect number of classes if not provided
    if n_classes is None and problem_type == "classification":
        n_classes = len(np.unique(y))

    is_multiclass = n_classes is not None and n_classes > 2

    if problem_type == "classification":
        best_score = float("inf")
        best_C = 0.01
        # Explicit StratifiedKFold with shuffle
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for C in [0.001, 0.01, 0.1, 1.0]:
            try:
                # Use multinomial for multiclass, auto otherwise
                model = LogisticRegression(
                    C=C,
                    random_state=42,
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="multinomial" if is_multiclass else "auto",
                    class_weight="balanced" if is_multiclass else None,
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
                print(f"      Warning: C={C} failed: {e}")
                continue

        metric_name = "roc_auc_ovr" if is_multiclass else "roc_auc"
        print(f"   Meta-model tuning: best C={best_C} (OOF log_loss={best_score:.4f})")
        print(f"   Using metric: {metric_name} for {'multiclass' if is_multiclass else 'binary'} classification")
        final_model = LogisticRegression(
            C=best_C,
            random_state=42,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial" if is_multiclass else "auto",
            class_weight="balanced" if is_multiclass else None,
        )
        return final_model, metric_name

    # Regression
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
            print(f"      Warning: alpha={alpha} failed: {e}")
            continue

    print(f"   Meta-model tuning: best alpha={best_alpha} (OOF RMSE={best_score:.4f})")
    return Ridge(alpha=best_alpha, random_state=42), "neg_rmse"


def diagnose_stacking_issues(
    meta_model: LogisticRegression | Ridge,
    model_names: list[str],
    meta_X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Diagnose potential issues with stacking meta-model.

    Detects common problems:
    - Very small coefficients (suggests misalignment)
    - Large negative coefficients (model hurting ensemble)
    - Intercept dominance (base models not informative)
    - High variance in coefficients (unstable stacking)

    Args:
        meta_model: Fitted meta-model (LogisticRegression or Ridge)
        model_names: Names of base models
        meta_X: Meta-features (stacked OOF predictions)
        y: Target labels

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics: dict[str, Any] = {
        "warnings": [],
        "coefficients": {},
        "intercept": None,
        "is_healthy": True,
    }

    if not hasattr(meta_model, "coef_"):
        return diagnostics

    coefs = meta_model.coef_.flatten()

    # For multi-class LogisticRegression, coef_ has shape (n_classes, n_features)
    # We average across classes to get a summary per model
    if meta_model.coef_.ndim > 1 and meta_model.coef_.shape[0] > 1:
        coefs = np.mean(np.abs(meta_model.coef_), axis=0)

    # Map coefficients to model names
    # Note: for classification, meta_X has n_classes columns per model
    n_features_per_model = len(coefs) // len(model_names) if len(model_names) > 0 else 1
    if n_features_per_model > 1:
        # Aggregate per-model (sum or mean of class coefficients)
        model_coefs = []
        for i in range(len(model_names)):
            start = i * n_features_per_model
            end = start + n_features_per_model
            model_coefs.append(np.mean(np.abs(coefs[start:end])))
        coefs = np.array(model_coefs)

    diagnostics["coefficients"] = dict(zip(model_names, coefs.tolist()))

    # Check for intercept
    if hasattr(meta_model, "intercept_"):
        intercept = meta_model.intercept_
        if isinstance(intercept, np.ndarray):
            intercept = float(np.mean(intercept))
        diagnostics["intercept"] = intercept

    print("\n   STACKING DIAGNOSTICS:")
    print(f"      Model coefficients: {diagnostics['coefficients']}")
    if diagnostics["intercept"] is not None:
        print(f"      Intercept: {diagnostics['intercept']:.4f}")

    # Warning: Very small coefficients suggest misalignment
    if np.abs(coefs).max() < 0.05:
        warning = (
            "CRITICAL: Very small coefficients detected! "
            "This usually means OOF predictions are misaligned with target. "
            "Check: train_ids alignment, sampling consistency, class order."
        )
        diagnostics["warnings"].append(warning)
        diagnostics["is_healthy"] = False
        print(f"      Warning: {warning}")

    # Warning: Large negative coefficients
    if np.min(coefs) < -0.5:
        worst_model = model_names[int(np.argmin(coefs))]
        warning = (
            f"Model '{worst_model}' has large negative coefficient ({np.min(coefs):.4f}). "
            "This model may be hurting the ensemble."
        )
        diagnostics["warnings"].append(warning)
        print(f"      Warning: {warning}")

    # Warning: Intercept dominance
    if diagnostics["intercept"] is not None:
        if abs(diagnostics["intercept"]) > 10 * np.abs(coefs).max():
            warning = (
                "Intercept dominates over coefficients. "
                "Base models may not be informative - meta-model is falling back to prior."
            )
            diagnostics["warnings"].append(warning)
            diagnostics["is_healthy"] = False
            print(f"      Warning: {warning}")

    # Warning: High coefficient variance (unstable)
    if len(coefs) > 1 and np.std(coefs) > 2 * np.mean(np.abs(coefs)):
        warning = (
            "High variance in coefficients. "
            "Consider reducing model diversity or checking for duplicate models."
        )
        diagnostics["warnings"].append(warning)
        print(f"      Warning: {warning}")

    if not diagnostics["warnings"]:
        print("      OK: No issues detected - stacking looks healthy")

    return diagnostics


def constrained_meta_learner(
    oof_stack: np.ndarray,
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
) -> tuple[np.ndarray, float]:
    """Learn non-negative weights that sum to 1.

    Args:
        oof_stack: Stacked OOF predictions (n_models, n_samples, ...)
        y_true: True labels
        problem_type: 'classification' or 'regression'
        metric_name: Metric name for scoring

    Returns:
        Tuple of (optimal_weights, best_score)
    """
    n_models = oof_stack.shape[0]

    def objective(weights: np.ndarray) -> float:
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_models) / n_models
        blended = np.average(oof_stack, axis=0, weights=weights)
        return score_predictions(blended, y_true, problem_type, metric_name)

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
        print(f"   Warning: Constrained optimization failed: {e}")
        opt_weights = dirichlet_weight_search(
            oof_stack, y_true, problem_type, metric_name, n_samples=300
        )
        best_score = objective(opt_weights)
        return opt_weights, best_score


def dirichlet_weight_search(
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
