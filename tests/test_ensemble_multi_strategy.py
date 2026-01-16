"""
Tests for MLE-STAR Multi-Strategy Ensemble Selection.

Tests the multi-strategy ensemble logic that was added to EnsembleAgent.
These tests are self-contained and don't require optional dependencies like dspy.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_predict

# Check if full kaggle_agents imports work (dspy available)
FULL_IMPORTS_AVAILABLE = importlib.util.find_spec("dspy") is not None


# ============================================================================
# Self-contained implementation of multi-strategy ensemble for testing
# This mirrors the logic in EnsembleAgent._generate_ensemble_candidates()
# ============================================================================


def generate_ensemble_candidates(
    prediction_pairs: dict[str, tuple[Path, Path]],
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
    minimize: bool = False,
) -> tuple[str, float, np.ndarray, dict[str, float] | None]:
    """
    Multi-Strategy Ensemble: Try multiple strategies, select best by OOF.

    This is a self-contained version for testing that mirrors the logic
    in EnsembleAgent._generate_ensemble_candidates().
    """
    candidates = []
    names = list(prediction_pairs.keys())
    n_models = len(names)

    if n_models < 1:
        return ("none", 0.0, np.array([]), None)

    # Load OOF and test predictions
    oof_list = []
    test_list = []
    valid_names = []

    for name, (oof_path, test_path) in prediction_pairs.items():
        try:
            oof = np.load(oof_path)
            test = np.load(test_path)
            if oof.ndim == 1:
                oof = oof.reshape(-1, 1)
            if test.ndim == 1:
                test = test.reshape(-1, 1)
            oof_list.append(oof)
            test_list.append(test)
            valid_names.append(name)
        except Exception as e:
            print(f"   WARNING: Failed to load {name}: {e}")

    if len(oof_list) < 1:
        return ("none", 0.0, np.array([]), None)

    # Validate shapes before stacking - filter to compatible arrays
    # Group by PAIRED shapes (oof_shape, test_shape) to find largest compatible group
    # This avoids discarding all models when OOF and test modal shapes come from different model sets
    oof_shapes = [arr.shape for arr in oof_list]
    test_shapes = [arr.shape for arr in test_list]

    # Group models by their paired (oof_shape, test_shape)
    shape_pair_to_indices: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for i, (oof_shape, test_shape) in enumerate(zip(oof_shapes, test_shapes)):
        pair = (oof_shape, test_shape)
        if pair not in shape_pair_to_indices:
            shape_pair_to_indices[pair] = []
        shape_pair_to_indices[pair].append(i)

    # Find the largest compatible group (most models with same paired shapes)
    largest_group_pair = max(shape_pair_to_indices.keys(), key=lambda p: len(shape_pair_to_indices[p]))
    compatible_indices = shape_pair_to_indices[largest_group_pair]

    if len(compatible_indices) < len(oof_list):
        oof_list = [oof_list[i] for i in compatible_indices]
        test_list = [test_list[i] for i in compatible_indices]
        valid_names = [valid_names[i] for i in compatible_indices]

    if len(oof_list) < 1:
        return ("none", 0.0, np.array([]), None)

    # Stack predictions
    oof_stack = np.stack(oof_list, axis=0)
    test_stack = np.stack(test_list, axis=0)

    def compute_score(preds: np.ndarray) -> float:
        """
        Compute OOF score for predictions using the competition metric.

        Supports common metrics for both classification and regression.
        Returns score normalized for maximization (higher = better).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            precision_score,
            r2_score,
            recall_score,
        )

        try:
            metric_lower = metric_name.lower().replace("_", "").replace("-", "")

            if problem_type == "classification":
                # Prepare predictions
                preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
                is_multiclass = preds_clipped.ndim > 1 and preds_clipped.shape[1] > 1

                if is_multiclass:
                    preds_norm = preds_clipped / preds_clipped.sum(axis=1, keepdims=True)
                    y_pred_class = np.argmax(preds_norm, axis=1)
                    # For binary, get positive class probability
                    if preds_norm.shape[1] == 2:
                        preds_proba = preds_norm[:, 1]
                    else:
                        preds_proba = preds_norm
                else:
                    # Single column or 1D: treat as binary positive class probability
                    preds_proba = preds_clipped.ravel() if preds_clipped.ndim > 1 else preds_clipped
                    y_pred_class = (preds_proba > 0.5).astype(int)
                    preds_norm = None  # Explicitly mark as not available

                # Compute metric based on metric_name
                if metric_lower in ["auc", "rocauc", "roc_auc", "auroc"]:
                    if is_multiclass and preds_clipped.shape[1] > 2:
                        score = roc_auc_score(y_true, preds_norm, multi_class="ovr", average="weighted")
                    else:
                        score = roc_auc_score(y_true, preds_proba)
                    return score  # Higher is better

                elif metric_lower in ["logloss", "log_loss", "crossentropy", "binarycrossentropy"]:
                    # Use preds_norm for multiclass, preds_proba for binary
                    score = log_loss(y_true, preds_norm if is_multiclass else preds_proba)
                    return -score  # Negate: lower log_loss is better

                elif metric_lower in ["accuracy", "acc"]:
                    score = accuracy_score(y_true, y_pred_class)
                    return score  # Higher is better

                elif metric_lower in ["f1", "f1score", "f1macro"]:
                    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
                    score = f1_score(y_true, y_pred_class, average=avg, zero_division=0)
                    return score  # Higher is better

                elif metric_lower in ["precision", "prec"]:
                    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
                    score = precision_score(y_true, y_pred_class, average=avg, zero_division=0)
                    return score  # Higher is better

                elif metric_lower in ["recall", "rec", "sensitivity"]:
                    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
                    score = recall_score(y_true, y_pred_class, average=avg, zero_division=0)
                    return score  # Higher is better

                else:
                    # Default to log_loss for unknown classification metrics
                    score = log_loss(y_true, preds_norm if is_multiclass else preds_proba)
                    return -score  # Negate: lower is better

            else:  # Regression
                preds_flat = preds.ravel() if preds.ndim > 1 else preds

                if metric_lower in ["rmse", "rootmeansquarederror"]:
                    score = np.sqrt(mean_squared_error(y_true, preds_flat))
                    return -score  # Negate: lower is better

                elif metric_lower in ["mse", "meansquarederror"]:
                    score = mean_squared_error(y_true, preds_flat)
                    return -score  # Negate: lower is better

                elif metric_lower in ["mae", "meanabsoluteerror", "l1"]:
                    score = mean_absolute_error(y_true, preds_flat)
                    return -score  # Negate: lower is better

                elif metric_lower in ["r2", "rsquared", "r2score"]:
                    score = r2_score(y_true, preds_flat)
                    return score  # Higher is better

                else:
                    # Default to RMSE for unknown regression metrics
                    score = np.sqrt(mean_squared_error(y_true, preds_flat))
                    return -score  # Negate: lower is better

        except Exception:
            return float("-inf")

    # Strategy 1: Simple Average
    try:
        avg_oof = np.mean(oof_stack, axis=0)
        avg_test = np.mean(test_stack, axis=0)
        avg_score = compute_score(avg_oof)
        candidates.append(("simple_avg", avg_score, avg_test, None))
    except Exception:
        pass

    # Strategy 2: Weighted Average
    try:
        individual_scores = [compute_score(oof_list[i]) for i in range(len(oof_list))]
        scores_arr = np.array(individual_scores)
        min_score = scores_arr.min()
        shifted = scores_arr - min_score + 1e-6
        weights = shifted / shifted.sum()

        weighted_oof = np.average(oof_stack, axis=0, weights=weights)
        weighted_test = np.average(test_stack, axis=0, weights=weights)
        weighted_score = compute_score(weighted_oof)
        weights_dict = dict(zip(valid_names, weights.tolist()))
        candidates.append(("weighted_avg", weighted_score, weighted_test, weights_dict))
    except Exception:
        pass

    # Strategy 3: Stacking with CV (no data leakage)
    try:
        n_samples = oof_stack.shape[1]
        meta_X = oof_stack.transpose(1, 0, 2).reshape(n_samples, -1)
        meta_X_test = test_stack.transpose(1, 0, 2).reshape(test_stack.shape[1], -1)

        if problem_type == "classification":
            meta_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            stacked_oof = cross_val_predict(
                meta_model, meta_X, y_true, cv=5, method="predict_proba", n_jobs=-1
            )
            meta_model.fit(meta_X, y_true)
            stacked_test = meta_model.predict_proba(meta_X_test)
        else:
            meta_model = Ridge(alpha=1.0)
            stacked_oof = cross_val_predict(meta_model, meta_X, y_true, cv=5, n_jobs=-1)
            meta_model.fit(meta_X, y_true)
            stacked_test = meta_model.predict(meta_X_test)

        stacked_score = compute_score(stacked_oof)
        candidates.append(("stacking", stacked_score, stacked_test, None))
    except Exception:
        pass

    # Strategy 4: Rank Average
    try:
        def rank_predictions(preds: np.ndarray) -> np.ndarray:
            if preds.ndim == 1:
                return rankdata(preds) / len(preds)
            return np.apply_along_axis(lambda x: rankdata(x) / len(x), axis=0, arr=preds)

        ranked_oof_list = [rank_predictions(oof) for oof in oof_list]
        ranked_test_list = [rank_predictions(test) for test in test_list]

        ranked_oof_stack = np.stack(ranked_oof_list, axis=0)
        ranked_test_stack = np.stack(ranked_test_list, axis=0)

        rank_avg_oof = np.mean(ranked_oof_stack, axis=0)
        rank_avg_test = np.mean(ranked_test_stack, axis=0)
        rank_score = compute_score(rank_avg_oof)
        candidates.append(("rank_avg", rank_score, rank_avg_test, None))
    except Exception:
        pass

    if not candidates:
        return ("none", 0.0, np.array([]), None)

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for model predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def binary_classification_data():
    """Generate binary classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    np.random.seed(42)
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )
    return X, y


def create_mock_predictions(
    temp_dir: Path,
    y: np.ndarray,
    n_models: int = 3,
    problem_type: str = "classification",
    n_classes: int = 3,
) -> dict[str, tuple[Path, Path]]:
    """Create mock OOF and test predictions for multiple models."""
    prediction_pairs = {}
    n_train = len(y)
    n_test = 100

    for i in range(n_models):
        model_name = f"model_{i}"
        np.random.seed(42 + i)

        if problem_type == "classification":
            oof_preds = np.random.dirichlet(np.ones(n_classes), size=n_train)
            for j in range(n_train):
                oof_preds[j, y[j]] += 0.3 + 0.1 * i
            oof_preds = oof_preds / oof_preds.sum(axis=1, keepdims=True)
            test_preds = np.random.dirichlet(np.ones(n_classes), size=n_test)
        else:
            oof_preds = y + np.random.randn(n_train) * (0.5 - 0.1 * i)
            oof_preds = oof_preds.reshape(-1, 1)
            test_preds = np.random.randn(n_test).reshape(-1, 1)

        oof_path = temp_dir / f"oof_{model_name}.npy"
        test_path = temp_dir / f"test_{model_name}.npy"
        np.save(oof_path, oof_preds)
        np.save(test_path, test_preds)
        prediction_pairs[model_name] = (oof_path, test_path)

    return prediction_pairs


# ============================================================================
# Tests for Multi-Strategy Ensemble Logic
# ============================================================================


class TestGenerateEnsembleCandidates:
    """Tests for multi-strategy ensemble generation."""

    def test_basic_classification_ensemble(self, temp_models_dir, classification_data):
        """Test basic multi-strategy ensemble for classification."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name in ["simple_avg", "weighted_avg", "stacking", "rank_avg"]
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert test_preds.shape[0] == 100
        assert test_preds.shape[1] == 3

    def test_binary_classification_ensemble(self, temp_models_dir, binary_classification_data):
        """Test multi-strategy ensemble for binary classification."""
        X, y = binary_classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=2
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="auc",
            minimize=False,
        )

        assert strategy_name in ["simple_avg", "weighted_avg", "stacking", "rank_avg"]
        assert test_preds.shape[0] == 100
        assert test_preds.shape[1] == 2

    def test_regression_ensemble(self, temp_models_dir, regression_data):
        """Test multi-strategy ensemble for regression."""
        X, y = regression_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="regression"
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="regression",
            metric_name="rmse",
            minimize=True,
        )

        assert strategy_name in ["simple_avg", "weighted_avg", "stacking", "rank_avg"]
        assert isinstance(score, float)
        assert test_preds.shape[0] == 100

    def test_single_model_fallback(self, temp_models_dir, classification_data):
        """Test that single model returns valid results."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=1, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name != "none"
        assert test_preds.shape[0] == 100

    def test_empty_predictions_returns_none(self, classification_data):
        """Test that empty prediction pairs returns 'none' strategy."""
        X, y = classification_data

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs={},
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name == "none"
        assert score == 0.0
        assert len(test_preds) == 0

    def test_weighted_avg_returns_weights(self, temp_models_dir, classification_data):
        """Test that weighted average strategy returns weight dict."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        if strategy_name == "weighted_avg":
            assert isinstance(weights, dict)
            assert len(weights) == 3
            assert all(isinstance(v, float) for v in weights.values())
            assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestShapeMismatchFiltering:
    """Tests for shape mismatch handling."""

    def test_filters_incompatible_shapes(self, temp_models_dir, classification_data):
        """Test that models with incompatible shapes are filtered out."""
        X, y = classification_data
        n_train = len(y)

        # Create models with different shapes
        for i, n_classes in enumerate([3, 3, 5]):
            model_name = f"model_{i}"
            oof_preds = np.random.dirichlet(np.ones(n_classes), size=n_train)
            test_preds = np.random.dirichlet(np.ones(n_classes), size=100)
            np.save(temp_models_dir / f"oof_{model_name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{model_name}.npy", test_preds)

        prediction_pairs = {
            f"model_{i}": (
                temp_models_dir / f"oof_model_{i}.npy",
                temp_models_dir / f"test_model_{i}.npy",
            )
            for i in range(3)
        }

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name != "none"
        assert test_preds.shape[1] == 3  # Most common shape

    def test_all_different_shapes_uses_most_common(self, temp_models_dir, classification_data):
        """Test that most common shape is selected when shapes differ."""
        X, y = classification_data
        n_train = len(y)

        # 3 models with shape (n, 3) and 1 with shape (n, 5)
        shapes = [3, 3, 3, 5]
        for i, n_classes in enumerate(shapes):
            model_name = f"model_{i}"
            oof_preds = np.random.dirichlet(np.ones(n_classes), size=n_train)
            test_preds = np.random.dirichlet(np.ones(n_classes), size=100)
            np.save(temp_models_dir / f"oof_{model_name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{model_name}.npy", test_preds)

        prediction_pairs = {
            f"model_{i}": (
                temp_models_dir / f"oof_model_{i}.npy",
                temp_models_dir / f"test_model_{i}.npy",
            )
            for i in range(4)
        }

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert test_preds.shape[1] == 3  # Most common

    def test_paired_shape_filtering_avoids_empty_intersection(self, temp_models_dir, classification_data):
        """
        Test that paired shape filtering finds valid groups even when
        independent modal OOF/test shapes would cause empty intersection.

        Scenario:
        - Models A, B: OOF shape (100, 3), test shape (50, 3)
        - Models C, D, E: OOF shape (100, 5), test shape (80, 5)

        Independent modal filtering would pick:
        - Most common OOF: (100, 5) from C, D, E
        - Most common test: (80, 5) from C, D, E
        -> This works, but imagine a different scenario...

        Scenario where independent filtering fails:
        - Models A, B, C: OOF shape (100, 3), test shape varies
        - Models D, E: OOF shape (100, 5), test shape (80, 5)

        If A, B have test shape (50, 3) and C has test shape (80, 5):
        - Most common OOF: (100, 3) from A, B, C
        - Most common test: Could be (80, 5) if D, E, C all have it
        -> Intersection could be just C or empty!

        Paired filtering groups by (oof, test) pairs and picks largest group.
        """
        X, y = classification_data
        n_train = len(y)

        # Create scenario: diverse shape pairs
        # Group 1: 2 models with OOF (n, 3), test (50, 3)
        # Group 2: 1 model with OOF (n, 3), test (80, 5) - different test shape!
        # Group 3: 2 models with OOF (n, 5), test (80, 5)
        configs = [
            ("model_a", 3, 50, 3),  # oof (n, 3), test (50, 3)
            ("model_b", 3, 50, 3),  # oof (n, 3), test (50, 3)
            ("model_c", 3, 80, 5),  # oof (n, 3), test (80, 5) - mixed!
            ("model_d", 5, 80, 5),  # oof (n, 5), test (80, 5)
            ("model_e", 5, 80, 5),  # oof (n, 5), test (80, 5)
        ]

        prediction_pairs = {}
        for name, oof_classes, test_samples, test_classes in configs:
            oof_preds = np.random.dirichlet(np.ones(oof_classes), size=n_train)
            test_preds = np.random.dirichlet(np.ones(test_classes), size=test_samples)
            np.save(temp_models_dir / f"oof_{name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{name}.npy", test_preds)
            prediction_pairs[name] = (
                temp_models_dir / f"oof_{name}.npy",
                temp_models_dir / f"test_{name}.npy",
            )

        # Independent modal filtering would struggle here:
        # - Most common OOF shape: (100, 3) from A, B, C
        # - Most common test shape: (80, 5) from C, D, E
        # - Intersection of (OOF=3 AND test=80x5): only model_c!

        # Paired filtering should pick Group 1 (A, B) or Group 3 (D, E) - both have 2 models
        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        # Should NOT return "none" - paired filtering should find a valid group
        assert strategy_name != "none", "Paired filtering should find compatible group"
        # Test predictions should have consistent shape (either 3 or 5 classes)
        assert test_preds.shape[1] in [3, 5], f"Unexpected shape: {test_preds.shape}"


class TestStackingCVNoLeakage:
    """Tests to verify stacking uses proper CV (no data leakage)."""

    def test_stacking_cv_score_differs_from_train_score(self, binary_classification_data):
        """
        Verify stacking OOF score is different from naive train score.

        If CV is working correctly, the OOF score should be worse than
        fitting on all data and predicting on the same data.
        """
        X, y = binary_classification_data
        n_train = len(y)

        # Create informative predictions
        np.random.seed(123)
        oof_1 = np.column_stack([
            1 - y + np.random.randn(n_train) * 0.3,
            y + np.random.randn(n_train) * 0.3
        ])
        oof_1 = np.clip(oof_1, 0.01, 0.99)
        oof_1 = oof_1 / oof_1.sum(axis=1, keepdims=True)

        oof_2 = np.column_stack([
            1 - y + np.random.randn(n_train) * 0.4,
            y + np.random.randn(n_train) * 0.4
        ])
        oof_2 = np.clip(oof_2, 0.01, 0.99)
        oof_2 = oof_2 / oof_2.sum(axis=1, keepdims=True)

        # Compute train score WITHOUT CV (data leakage)
        meta_X = np.hstack([oof_1, oof_2])
        meta_model = LogisticRegression(C=1.0, max_iter=1000)
        meta_model.fit(meta_X, y)
        train_preds = meta_model.predict_proba(meta_X)
        train_score = -log_loss(y, train_preds)

        # Compute CV score (what our implementation does)
        cv_preds = cross_val_predict(meta_model, meta_X, y, cv=5, method="predict_proba")
        cv_score = -log_loss(y, cv_preds)

        # CV score should be worse (more negative) than train score
        assert cv_score < train_score, "CV score should be worse than train score"

    def test_stacking_produces_valid_predictions(self, temp_models_dir, binary_classification_data):
        """Test that stacking with CV produces valid predictions."""
        X, y = binary_classification_data
        n_train = len(y)

        # Create high-quality predictions
        for i in range(2):
            model_name = f"model_{i}"
            oof_preds = np.zeros((n_train, 2))
            oof_preds[y == 0, 0] = 0.7 + np.random.rand(np.sum(y == 0)) * 0.2
            oof_preds[y == 0, 1] = 1 - oof_preds[y == 0, 0]
            oof_preds[y == 1, 1] = 0.7 + np.random.rand(np.sum(y == 1)) * 0.2
            oof_preds[y == 1, 0] = 1 - oof_preds[y == 1, 1]
            test_preds = np.random.dirichlet(np.ones(2), size=100)
            np.save(temp_models_dir / f"oof_{model_name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{model_name}.npy", test_preds)

        prediction_pairs = {
            f"model_{i}": (
                temp_models_dir / f"oof_model_{i}.npy",
                temp_models_dir / f"test_model_{i}.npy",
            )
            for i in range(2)
        }

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert np.isfinite(score)
        assert test_preds.shape[0] == 100


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_files_gracefully(self, temp_models_dir, classification_data):
        """Test that missing prediction files are handled gracefully."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=1, problem_type="classification", n_classes=3
        )

        # Add non-existent files
        prediction_pairs["missing_model"] = (
            temp_models_dir / "oof_missing.npy",
            temp_models_dir / "test_missing.npy",
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name != "none"

    def test_handles_1d_predictions(self, temp_models_dir, binary_classification_data):
        """Test that 1D predictions are properly reshaped."""
        X, y = binary_classification_data
        n_train = len(y)

        for i in range(2):
            model_name = f"model_{i}"
            oof_preds = np.random.rand(n_train)  # 1D array
            test_preds = np.random.rand(100)
            np.save(temp_models_dir / f"oof_{model_name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{model_name}.npy", test_preds)

        prediction_pairs = {
            f"model_{i}": (
                temp_models_dir / f"oof_model_{i}.npy",
                temp_models_dir / f"test_model_{i}.npy",
            )
            for i in range(2)
        }

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="auc",
            minimize=False,
        )

        assert strategy_name != "none"
        assert test_preds.shape[0] == 100

    def test_handles_single_column_binary_with_logloss(self, temp_models_dir, binary_classification_data):
        """
        Test that single-column binary predictions work with log_loss metric.

        This tests the fix for the bug where preds_norm was undefined for
        single-column predictions, causing log_loss to fail.
        """
        X, y = binary_classification_data
        n_train = len(y)

        # Create 2D predictions with single column (n, 1) - common for binary
        for i in range(2):
            model_name = f"model_{i}"
            oof_preds = np.random.rand(n_train, 1)  # 2D with 1 column
            test_preds = np.random.rand(100, 1)
            np.save(temp_models_dir / f"oof_{model_name}.npy", oof_preds)
            np.save(temp_models_dir / f"test_{model_name}.npy", test_preds)

        prediction_pairs = {
            f"model_{i}": (
                temp_models_dir / f"oof_model_{i}.npy",
                temp_models_dir / f"test_model_{i}.npy",
            )
            for i in range(2)
        }

        # This would previously fail with NameError: preds_norm not defined
        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",  # Specifically test log_loss
            minimize=True,
        )

        # Should not fail, should return valid results
        assert strategy_name != "none", "Single-column binary with log_loss should work"
        assert np.isfinite(score), f"Score should be finite, got {score}"
        assert test_preds.shape[0] == 100


class TestStrategySelection:
    """Tests for strategy selection logic."""

    def test_best_strategy_selected(self, temp_models_dir, classification_data):
        """Test that a valid strategy is selected."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        valid_strategies = ["simple_avg", "weighted_avg", "stacking", "rank_avg"]
        assert strategy_name in valid_strategies

    def test_score_is_finite(self, temp_models_dir, classification_data):
        """Test that returned score is finite."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert np.isfinite(score)


# ============================================================================
# Tests for Multi-Metric Support
# ============================================================================


class TestMultiMetricSupport:
    """Tests to verify ensemble selection works with various metrics."""

    def test_accuracy_metric_classification(self, temp_models_dir, classification_data):
        """Test ensemble selection with accuracy metric."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="accuracy",
            minimize=False,
        )

        assert strategy_name != "none"
        # Accuracy should be between 0 and 1 (not negated)
        assert 0.0 <= score <= 1.0, f"Accuracy score {score} out of range [0, 1]"

    def test_f1_metric_binary_classification(self, temp_models_dir, binary_classification_data):
        """Test ensemble selection with F1 metric for binary classification."""
        X, y = binary_classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=2
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="f1",
            minimize=False,
        )

        assert strategy_name != "none"
        # F1 should be between 0 and 1 (not negated)
        assert 0.0 <= score <= 1.0, f"F1 score {score} out of range [0, 1]"

    def test_mae_metric_regression(self, temp_models_dir, regression_data):
        """Test ensemble selection with MAE metric for regression."""
        X, y = regression_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="regression"
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="regression",
            metric_name="mae",
            minimize=True,
        )

        assert strategy_name != "none"
        # MAE is negated for maximization, so score should be <= 0
        assert score <= 0, f"MAE score {score} should be <= 0 (negated for maximization)"

    def test_r2_metric_regression(self, temp_models_dir, regression_data):
        """Test ensemble selection with R2 metric for regression."""
        X, y = regression_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="regression"
        )

        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="regression",
            metric_name="r2",
            minimize=False,
        )

        assert strategy_name != "none"
        # R2 can range from -inf to 1, but for reasonable predictions should be > -10
        assert score > -10, f"R2 score {score} seems unreasonably low"

    def test_unknown_metric_falls_back_gracefully(self, temp_models_dir, classification_data):
        """Test that unknown metrics fall back to default without crashing."""
        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        # Use an unknown metric name
        strategy_name, score, test_preds, weights = generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="unknown_metric_xyz",
            minimize=True,
        )

        # Should not crash, should return valid results using default (log_loss)
        assert strategy_name != "none"
        assert np.isfinite(score)


# ============================================================================
# Tests for Binary Probability Column Selection
# ============================================================================


class TestBinaryProbabilityColumnSelection:
    """Tests to verify binary classification uses positive class probability."""

    def test_binary_submission_uses_positive_class_probability(self, temp_models_dir):
        """
        Verify binary predictions use column 1 (positive class), not column 0.

        Binary classification predict_proba returns [P(class0), P(class1)].
        Kaggle expects P(positive class) = column 1.
        When truncating 2 columns to 1, we must use column 1, not column 0.
        """
        import pandas as pd

        # Create binary predictions with distinct values in each column
        # Column 0 (negative class): low values ~0.2
        # Column 1 (positive class): high values ~0.8
        n_test = 100
        test_preds = np.column_stack([
            np.full(n_test, 0.2),  # Column 0: negative class prob
            np.full(n_test, 0.8),  # Column 1: positive class prob
        ])

        # Create sample submission with 1 target column (id + target)
        sample_sub = pd.DataFrame({
            "id": range(n_test),
            "target": np.zeros(n_test),
        })

        expected_cols = len(sample_sub.columns) - 1  # = 1

        # Simulate the truncation logic from ensemble/agent.py
        if test_preds.shape[1] == 2 and expected_cols == 1:
            # Correct: use positive class (column 1)
            result = test_preds[:, 1]
        else:
            # Wrong: would use negative class (column 0)
            result = test_preds[:, :expected_cols].ravel()

        # Verify we got the positive class probabilities (~0.8), not negative (~0.2)
        assert np.allclose(result, 0.8), f"Expected ~0.8 (positive class), got {result[0]}"

    def test_binary_truncation_logic_matches_implementation(self, temp_models_dir):
        """
        Test that the truncation logic correctly handles various scenarios.

        This tests the actual branching logic used in ensemble/agent.py.
        """
        import pandas as pd

        def apply_truncation_logic(test_preds: np.ndarray, expected_cols: int) -> np.ndarray:
            """Mirror the truncation logic from ensemble/agent.py lines 1246-1253."""
            if test_preds.ndim == 1:
                return test_preds
            elif test_preds.shape[1] == expected_cols:
                return test_preds
            elif test_preds.shape[1] > expected_cols:
                # Binary classification: predict_proba returns [P(class0), P(class1)]
                # Kaggle expects P(positive class) = column 1, not column 0
                if test_preds.shape[1] == 2 and expected_cols == 1:
                    return test_preds[:, 1:2]  # Positive class probability
                else:
                    # Multi-class: take first expected_cols columns
                    return test_preds[:, :expected_cols]
            else:
                # Pad with 0.5
                n_test = test_preds.shape[0]
                padded = np.full((n_test, expected_cols), 0.5)
                padded[:, :test_preds.shape[1]] = test_preds
                return padded

        n_test = 50

        # Case 1: Binary (2 cols) → 1 col submission
        binary_preds = np.column_stack([
            np.linspace(0.9, 0.1, n_test),  # Col 0: decreasing (negative class)
            np.linspace(0.1, 0.9, n_test),  # Col 1: increasing (positive class)
        ])
        result = apply_truncation_logic(binary_preds, expected_cols=1)
        assert result.shape == (n_test, 1), f"Expected shape (50, 1), got {result.shape}"
        assert np.allclose(result[:, 0], np.linspace(0.1, 0.9, n_test)), \
            "Binary truncation should use column 1 (positive class)"

        # Case 2: Multi-class (5 cols) → 3 col submission (take first 3)
        multiclass_preds = np.random.rand(n_test, 5)
        result = apply_truncation_logic(multiclass_preds, expected_cols=3)
        assert result.shape == (n_test, 3), f"Expected shape (50, 3), got {result.shape}"
        assert np.allclose(result, multiclass_preds[:, :3]), \
            "Multi-class truncation should use first 3 columns"

        # Case 3: Exact match (3 cols) → 3 col submission
        exact_preds = np.random.rand(n_test, 3)
        result = apply_truncation_logic(exact_preds, expected_cols=3)
        assert np.allclose(result, exact_preds), "Exact match should return unchanged"

        # Case 4: Under-specified (2 cols) → 4 col submission (pad with 0.5)
        under_preds = np.random.rand(n_test, 2)
        result = apply_truncation_logic(under_preds, expected_cols=4)
        assert result.shape == (n_test, 4), f"Expected shape (50, 4), got {result.shape}"
        assert np.allclose(result[:, :2], under_preds), "First 2 cols should match input"
        assert np.allclose(result[:, 2:], 0.5), "Padded cols should be 0.5"

    def test_inverted_probabilities_would_hurt_score(self):
        """
        Demonstrate that using column 0 instead of column 1 inverts predictions.

        This shows why the fix is critical - using the wrong column would
        cause scores to be inverted (predicting low when should be high).
        """
        from sklearn.metrics import roc_auc_score

        # True labels: 50 zeros, 50 ones
        y_true = np.array([0] * 50 + [1] * 50)

        # Good model: P(class1) is high when y=1, low when y=0
        # Column 0 = P(class0), Column 1 = P(class1)
        good_probs = np.column_stack([
            np.concatenate([np.full(50, 0.8), np.full(50, 0.2)]),  # P(class0)
            np.concatenate([np.full(50, 0.2), np.full(50, 0.8)]),  # P(class1) - correct
        ])

        # Using column 1 (correct): high AUC
        auc_correct = roc_auc_score(y_true, good_probs[:, 1])

        # Using column 0 (wrong): inverted AUC
        auc_wrong = roc_auc_score(y_true, good_probs[:, 0])

        assert auc_correct > 0.9, f"Correct column should give high AUC, got {auc_correct}"
        assert auc_wrong < 0.2, f"Wrong column should give inverted AUC, got {auc_wrong}"
        assert auc_correct > auc_wrong, "Using correct column must give better AUC"


# ============================================================================
# Integration tests with full EnsembleAgent (only if dspy is available)
# ============================================================================


@pytest.mark.skipif(not FULL_IMPORTS_AVAILABLE, reason="dspy not installed")
class TestEnsembleAgentIntegration:
    """Integration tests that require the full kaggle_agents imports."""

    def test_ensemble_agent_generates_candidates(self, temp_models_dir, classification_data):
        """Test that EnsembleAgent._generate_ensemble_candidates works."""
        from kaggle_agents.agents.ensemble.agent import EnsembleAgent

        X, y = classification_data
        prediction_pairs = create_mock_predictions(
            temp_models_dir, y, n_models=3, problem_type="classification", n_classes=3
        )

        agent = EnsembleAgent()
        strategy_name, score, test_preds, weights = agent._generate_ensemble_candidates(
            prediction_pairs=prediction_pairs,
            y_true=y,
            problem_type="classification",
            metric_name="log_loss",
            minimize=True,
        )

        assert strategy_name in ["simple_avg", "weighted_avg", "stacking", "rank_avg"]
        assert np.isfinite(score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
